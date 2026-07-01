"""
varlib.validation.comparison
============================
Comparación sistemática de modelos VaR mediante backtesting.

Dos funciones, una por modo de ventana, paralelas a los dos métodos del
Backtester (run_expanding / run_rolling):

- compare_models_expanding(backtesters, train, test): ejecuta cada
  Backtester con ventana EXPANSIVA sobre el test (warm-up = train).
- compare_models_rolling(backtesters, returns, window, ...): ejecuta cada
  Backtester con ventana DESLIZANTE fija de window días sobre la serie.

La unidad de comparación es un Backtester ya configurado: cada uno lleva su
propio modelo y su política de ejecución (step), de modo que esa configuración
vive donde corresponde, en el Backtester, y no en la firma de la comparación.
Las dos funciones solo aportan: aplicar el mismo modo de ventana y los mismos
datos a todos, y apilar sus summary() en un único DataFrame, una fila por modelo.

El DataFrame resultante tiene el mismo esquema que produce Backtester.summary(),
por lo que se puede pasar tal cual a ReportExporter.

Identificación por etiqueta
---------------------------
Cada Backtester se identifica por la etiqueta que le asigna el usuario en
el diccionario de entrada (no por el nombre de la clase del modelo). Esto
permite distinguir variantes que comparten clase, p.ej. un GARCHParametricVaR
con distribución normal frente a otro con t de Student.

Niveles de comparación típicos
-------------------------------
- Distribuciones dentro de una misma metodología (p.ej. GARCH-paramétrico
  normal vs t vs skew-t).
- Metodologías distintas entre sí (histórico, paramétrico, Monte Carlo,
  GARCH-paramétrico, GARCH-Monte Carlo).
"""

from copy import deepcopy
from typing import Callable

import pandas as pd

from varlib.validation.backtesting import Backtester


def _compare(
    backtesters: dict,
    run: Callable[[Backtester], Backtester],
) -> pd.DataFrame:
    """
    Motor común de las dos funciones de comparación.

    Valida el diccionario de Backtesters y, para cada uno: lo copia (sin
    mutar el que pasó el usuario), delega la ejecución en run (el método de
    ventana concreto con sus datos) y recoge su summary() reetiquetado.

    Parameters
    ----------
    backtesters : dict
        Diccionario {etiqueta: Backtester}.
    run : callable
        Recibe un Backtester y ejecuta sobre él el modo de ventana deseado
        (run_expanding o run_rolling con sus datos).

    Raises
    ------
    ValueError
        Si backtesters no es un diccionario o está vacío.
    TypeError
        Si algún valor no es una instancia de Backtester.
    """
    if not isinstance(backtesters, dict) or len(backtesters) == 0:
        raise ValueError(
            "[backtesters] debe ser un diccionario no vacío {etiqueta: Backtester}."
        )

    rows = []
    for label, bt in backtesters.items():
        if not isinstance(bt, Backtester):
            raise TypeError(
                f"[backtesters] el valor de '{label}' debe ser una instancia de "
                f"Backtester. Recibido: '{type(bt).__name__}'"
            )

        # Copia para no mutar (run_* deja el Backtester en estado post-run).
        b = deepcopy(bt)
        run(b)

        row = b.summary()   # dict con las métricas del backtest
        # Sustituir el nombre de la clase por la etiqueta del usuario
        row["modelo"] = str(label)
        rows.append(row)

    # Una fila por modelo: DataFrame a partir de la lista de dicts.
    return pd.DataFrame(rows)


def compare_models_expanding(
    backtesters: dict,
    train: pd.Series,
    test: pd.Series,
) -> pd.DataFrame:
    """
    Compara varios modelos VaR con ventana EXPANSIVA y reúne sus
    estadísticos en un único DataFrame.

    Paralelo a Backtester.run_expanding: en cada día del test el modelo se
    reajusta con todo el historial disponible hasta el día anterior (la
    ventana crece). El train define el warm-up inicial y el test el periodo
    evaluado.

    Parameters
    ----------
    backtesters : dict
        Diccionario {etiqueta: Backtester}, donde cada Backtester envuelve un
        modelo (instancia de BaseVaR) y su política de ejecución (step). La
        etiqueta es libre y sirve para distinguir variantes de una misma clase.
    train : pd.Series
        Histórico inicial de entrenamiento (warm-up), común a todos.
    test : pd.Series
        Rendimientos sobre los que se evalúan las violaciones, común a todos.

    Returns
    -------
    pd.DataFrame
        Una fila por modelo, en el orden del diccionario. Columnas
        heredadas de Backtester.summary(): modelo (la etiqueta),
        confianza, n_obs, n_violations, violation_rate,
        expected_rate, LR_uc, p_uc, reject_uc, LR_ind,
        p_ind, reject_ind, LR_cc, p_cc, reject_cc.
        Apto para pasar directamente a ReportExporter.

    Raises
    ------
    ValueError
        Si backtesters no es un diccionario o está vacío.
    TypeError
        Si algún valor no es una instancia de Backtester.

    Examples
    --------
    from varlib.models.garch_parametric import GARCHParametricVaR
    from varlib.validation.backtesting import Backtester
    from varlib.validation.comparison import compare_models_expanding

    tabla = compare_models_expanding({
        "GARCH-Normal": Backtester(GARCHParametricVaR(dist="normal")),
        "GARCH-t":      Backtester(GARCHParametricVaR(dist="t")),
    }, train, test)
    """
    return _compare(
        backtesters,
        lambda b: b.run_expanding(train, test),
    )


def compare_models_rolling(
    backtesters: dict,
    returns: pd.Series,
    window: int,
    *,
    eval_start=None,
) -> pd.DataFrame:
    """
    Compara varios modelos VaR con ventana DESLIZANTE y reúne sus
    estadísticos en un único DataFrame.

    Paralelo a Backtester.run_rolling: en cada día t el modelo se reajusta
    con los window días previos. No necesita partición train/test: cada
    predicción es out-of-sample por construcción (la ventana solo mira hacia
    atrás).

    Parameters
    ----------
    backtesters : dict
        Diccionario {etiqueta: Backtester}, donde cada Backtester envuelve un
        modelo (instancia de BaseVaR) y su política de ejecución (step). La
        etiqueta es libre y sirve para distinguir variantes de una misma clase.
    returns : pd.Series
        Serie completa de retornos, común a todos.
    window : int
        Tamaño de la ventana deslizante (>= 1), común a todos.
    eval_start : etiqueta de índice or None, optional
        Fecha desde la que se empiezan a contar violaciones.
        - None (defecto): evalúa desde el día window en adelante.
        - Una fecha: evalúa desde ahí, usando los window días anteriores
          como warm-up. Sirve para alinear la evaluación con un backtest
          expansivo sobre el mismo periodo.

    Returns
    -------
    pd.DataFrame
        Una fila por modelo, en el orden del diccionario. Columnas
        heredadas de Backtester.summary(): modelo (la etiqueta),
        confianza, n_obs, n_violations, violation_rate,
        expected_rate, LR_uc, p_uc, reject_uc, LR_ind,
        p_ind, reject_ind, LR_cc, p_cc, reject_cc.
        Apto para pasar directamente a ReportExporter.

    Raises
    ------
    ValueError
        Si backtesters no es un diccionario o está vacío.
    TypeError
        Si algún valor no es una instancia de Backtester.

    Examples
    --------
    from varlib.models.parametric import ParametricVaR
    from varlib.models.garch_parametric import GARCHParametricVaR
    from varlib.validation.backtesting import Backtester
    from varlib.validation.comparison import compare_models_rolling

    tabla = compare_models_rolling({
        "Param-Normal": Backtester(ParametricVaR(dist="normal")),
        "GARCH-t":      Backtester(GARCHParametricVaR(dist="t")),
    }, returns, window=250, eval_start=test.index[0])
    """
    return _compare(
        backtesters,
        lambda b: b.run_rolling(returns, window, eval_start=eval_start),
    )
