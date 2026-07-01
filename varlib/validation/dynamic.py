"""
varlib.validation.dynamic
==========================
Capa de aplicación de ventana deslizante (rolling / expanding window)
sobre cualquier estimador puntual de VaR.

Motivación
----------
Cada modelo de la librería (ParametricVaR, HistoricalVaR,
GARCHParametricVaR, ...) es un estimador puntual: dado un tramo de
rendimientos devuelve un único VaR para el instante final de ese tramo.
En una entidad financiera real el VaR no se calcula una sola vez, sino
día a día, reestimándolo con una ventana móvil de historia reciente
(reporting diario y backtesting regulatorio).

Este módulo separa esa responsabilidad: el modelo sigue siendo un
estimador puntual y la lógica de "deslizar la ventana en el tiempo" vive
en una sola función reutilizable, dynamic_var, en lugar de
replicarse dentro de cada modelo o dentro del Backtester.

Ausencia de look-ahead bias
---------------------------
La garantía fundamental es que el VaR asignado a la fecha t se estima
usando exclusivamente datos anteriores a t (hasta t-1).
La implementación usa cortes posicionales returns.iloc[inicio:t], que
son exclusivos en el extremo derecho, de modo que el dato de la propia
fecha t nunca entra en su estimación.

Ventana deslizante vs. expansiva
-----------------------------
- expanding=False (defecto): ventana deslizante de longitud fija
  window.  El VaR de t usa returns.iloc[t-window:t].
- expanding=True: ventana expansiva.  El VaR de t usa todos
  los datos disponibles returns.iloc[0:t], exigiendo al menos
  window observaciones de calentamiento antes de emitir el primero.

Reestimación en cada paso
--------------------------
El modelo se reajusta (.fit) sobre su ventana en cada fecha evaluada, de
modo que el VaR reacciona a la información más reciente. Para los modelos
GARCH esto implica una estimación por Máxima Verosimilitud por paso, un
esquema de filtrado online que actualice la volatilidad condicional sin
reestimar los parámetros queda como posible mejora futura.

Convención de signo
-------------------
Se mantiene la convención del resto de la librería: el VaR es una pérdida
positiva y existe violación cuando retorno_t < -VaR[t].
"""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from varlib.models.base import BaseVaR


def dynamic_var(
    model: BaseVaR,
    returns: pd.Series,
    window: int,
    *,
    step: int = 1,
    expanding: bool = False,
    min_obs: int = 30,
    compute_es: bool = False,
    on_error: str = "previous",
) -> pd.Series | pd.DataFrame:
    """
    Calcula la serie temporal de VaR con ventana deslizante.

    Aplica cualquier estimador puntual de VaR (instancia de
    varlib.models.base.BaseVaR) sobre una ventana móvil que se
    desplaza a lo largo de returns, devolviendo un VaR por cada fecha
    evaluada.  El VaR de la fecha t se estima usando solo datos
    anteriores a t (sin look-ahead bias).

    Parameters
    ----------
    model : BaseVaR
        Instancia (ajustada o no) de cualquier subclase de BaseVaR.
        Se clona internamente con copy.deepcopy en cada paso, por lo
        que el objeto original no se modifica.
    returns : pd.Series
        Serie completa de rendimientos, indexada por fecha.  Los NaN
        se eliminan al inicio.
    window : int
        Longitud de la ventana deslizante.  Si expanding=True actúa como
        número mínimo de observaciones de calentamiento antes de emitir
        el primer VaR.  Debe ser >= min_obs.
    step : int, optional
        Stride o paso de evaluación: cada cuántos días se emite un VaR.
        step=1 (defecto) corresponde al reporting diario.  En las
        fechas no evaluadas la serie devuelta contiene NaN.
    expanding : bool, optional
        Si True usa ventana expansiva (todos los datos hasta t-1),
        si False (defecto) usa ventana deslizante de longitud window.
    min_obs : int, optional
        Número mínimo de observaciones para considerar una ventana
        válida.  Por defecto 30.
    compute_es : bool, optional
        Si True también calcula el ES (Expected Shortfall, también CVaR) y
        devuelve un pd.DataFrame con columnas VaR y ES.
    on_error : {'previous', 'nan'}, optional
        Comportamiento ante una ventana inválida o un fallo de ajuste:

        - 'previous' (defecto): arrastra el último VaR válido
          calculado (o NaN si aún no hay ninguno).
        - 'nan': deja NaN en esa fecha.

    Returns
    -------
    pd.Series or pd.DataFrame
        Si compute_es=False (defecto) devuelve un pd.Series
        llamado "VaR" alineado al índice de returns.  Si
        compute_es=True devuelve un pd.DataFrame con columnas
        "VaR" y "ES".  En ambos casos las fechas de
        calentamiento y las no evaluadas por step contienen NaN.

    Raises
    ------
    TypeError
        Si model no es una instancia de BaseVaR o returns no
        es un pd.Series.
    ValueError
        Si window o step no son enteros válidos, o si window < min_obs,
        o on_error no es reconocido.

    Notes
    -----
    No existe look-ahead bias: el corte returns.iloc[inicio:t] es
    exclusivo en t, por lo que el dato de la fecha estimada nunca
    interviene en su propia estimación.

    Examples
    --------
    from varlib.models.parametric import ParametricVaR
    from varlib.validation.dynamic import dynamic_var
    var = dynamic_var(ParametricVaR(0.99), returns, window=250)
    var.tail(3)

    GARCH con ventana deslizante:

    from varlib.models.garch_parametric import GARCHParametricVaR
    var = dynamic_var(GARCHParametricVaR(0.99, dist='t'), returns, window=500)
    """
    if not isinstance(model, BaseVaR):
        raise TypeError(
            f"[model] debe ser una instancia de BaseVaR. "
            f"Recibido: '{type(model).__name__}'"
        )
    if not isinstance(returns, pd.Series):
        raise TypeError(
            f"[returns] debe ser un pd.Series. "
            f"Recibido: '{type(returns).__name__}'"
        )
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"[window] debe ser un entero >= 1. Recibido: '{window}'")
    if not isinstance(step, int) or step < 1:
        raise ValueError(f"[step] debe ser un entero >= 1. Recibido: '{step}'")
    if window < min_obs:
        raise ValueError(
            f"[window] '{window}' debe ser >= [min_obs] '{min_obs}'."
        )
    if on_error not in ("previous", "nan"):
        raise ValueError(
            f"[on_error] debe ser 'previous' o 'nan'. Recibido: '{on_error}'"
        )

    clean = returns.dropna()
    n = len(clean)

    var_vals = np.full(n, np.nan)
    es_vals = np.full(n, np.nan) if compute_es else None

    # Primera fecha con suficiente historia previa.
    t_first = window

    n_fits = 0              # nº de ventanas ajustadas (.fit)
    n_not_converged = 0     # de ellas, cuántas no convergieron
    n_failures = 0          # ajustes/cálculos que lanzaron excepción
    last_var = np.nan
    last_es = np.nan

    for t in range(t_first, n, step):
        if expanding:
            window_data = clean.iloc[0:t]
        else:
            window_data = clean.iloc[t - window:t]

        if len(window_data) < min_obs:
            # Ventana insuficiente: aplicar política de error.
            if on_error == "previous":
                var_vals[t] = last_var
                if compute_es:
                    es_vals[t] = last_es
            continue

        try:
            candidate = deepcopy(model)
            n_fits += 1
            # Capturamos los warnings del ajuste (sin mostrarlos uno a uno)
            # para contar las no-convergencias y avisar agregado al final.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate.fit(window_data)
            if any(issubclass(w.category, RuntimeWarning) for w in caught):
                n_not_converged += 1

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                var_t = float(candidate.compute_var())
                es_t = (float(candidate.compute_es()) if compute_es else np.nan)

            var_vals[t] = var_t
            last_var = var_t
            if compute_es:
                es_vals[t] = es_t
                last_es = es_t

        except Exception:
            # El ajuste o el cálculo del VaR lanzó una excepción (datos
            # degenerados, error numérico de arch, etc.): se cuenta el fallo.
            n_failures += 1
            if on_error == "previous":
                var_vals[t] = last_var
                if compute_es:
                    es_vals[t] = last_es

    if n_not_converged > 0:
        warnings.warn(
            f"{type(model).__name__}: el optimizador no convergió en "
            f"{n_not_converged} de {n_fits} ventanas.",
            RuntimeWarning,
            stacklevel=2,
        )
    if n_failures > 0:
        warnings.warn(
            f"{type(model).__name__}: el ajuste/cálculo del VaR falló (excepción) "
            f"en {n_failures} ventana(s). Se arrastró el VaR previo (on_error='previous').",
            RuntimeWarning,
            stacklevel=2,
        )

    var_series = pd.Series(var_vals, index=clean.index, name="VaR")

    if compute_es:
        return pd.DataFrame(
            {"VaR": var_series, "ES": pd.Series(es_vals, index=clean.index)}
        )
    
    return var_series
