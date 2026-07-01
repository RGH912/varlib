"""
varlib.validation.backtesting
==============================
Backtesting estadístico del VaR mediante tests de cobertura.

Metodología
-----------
Se evalúa si el modelo VaR produce la tasa de violaciones correcta
sobre la muestra de test.  El VaR es una pérdida positiva, por lo que
una violación ocurre cuando:

    r[t] < -VaR[t]   (la pérdida realizada supera el VaR estimado)

El backtesting calcula primero un VaR dinámico con ventana deslizante:
para cada observación t del test, el modelo se reajusta con la
ventana [t-window, t-1] y genera el VaR para t.  Si window=None
(defecto) se usa una ventana expansiva (todos los datos hasta t-1
comenzando con el conjunto train completo).

Tests implementados
-------------------
Test de Kupiec (LR_uc)
    Contrasta H0: p = 1-conf, donde p es la tasa de violaciones
    observada.  Bajo H0 el estadístico sigue una chi^2(1).

    LR_uc = -2 * [log L(p0) - log L(p_hat)]
    L(p) = (1-p)^(T-V) * p^V

Test de Christoffersen (LR_ind + LR_cc)
    Adicionalmente contrasta la independencia temporal de las
    violaciones.  Construye la matriz de transición 2*2
    (estado 0 = no-violación, estado 1 = violación, nij = i -> j):

        n00, n01   (no-violación -> no-violación, no-violación -> violación)
        n10, n11   (violación -> no-violación,    violación -> violación)

    LR_ind = -2 * log [L(pi_hat) / L(pi01, pi11)]
    LR_cc  = LR_uc + LR_ind  ~ chi^2(2)

Interpretación
--------------
- p-value >= significance (0.05 por defecto) -> no se rechaza H0 -> modelo
  estadísticamente válido.
- Se reportan los cuatro estadísticos: LR_uc, LR_ind, LR_cc con sus
  p-valores, la tasa de violaciones observada y el número de excedencias.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

from varlib.models.base import BaseVaR
from varlib.validation.dynamic import dynamic_var


class Backtester:
    """
    Validación estadística de modelos VaR mediante backtesting.

    Ajusta el modelo sobre los datos de entrenamiento y genera un VaR
    dinámico sobre el conjunto de test, luego aplica los tests de
    Kupiec y Christoffersen para evaluar la cobertura y la
    independencia de las violaciones.

    Parameters
    ----------
    model : BaseVaR
        Instancia de cualquier subclase de BaseVaR (no ajustada).
        Se clona internamente en cada paso del backtesting.
    step : int, optional
        Paso de evaluación (stride) que se traslada a la capa
        varlib.validation.dynamic.dynamic_var.  1 (defecto)
        evalúa el VaR en cada día del test.
    significance : float, optional
        Nivel de significación de los tests estadísticos (Kupiec,
        Christoffersen). Se rechaza H0 si p_value < significance.
        Por defecto 0.05.

    Attributes
    ----------
    violations_ : np.ndarray of bool
        Vector binario de violaciones sobre el test (post-run).
    var_series_ : pd.Series
        VaR estimado para cada observación del test (post-run).
    n_violations_ : int
        Número total de violaciones (post-run).
    violation_rate_ : float
        Tasa de violaciones observada: n_violations / n_test (post-run).
    test_returns_ : pd.Series
        Serie de retornos de test usada en el último run (post-run).

    Examples
    --------
    from varlib.models.parametric import ParametricVaR
    from varlib.validation.backtesting import Backtester
    bt = Backtester(ParametricVaR(confidence=0.95))
    bt.run_expanding(train_returns, test_returns)
    print(bt.kupiec_test())
    print(bt.summary())
    """

    def __init__(
        self,
        model: BaseVaR,
        step: int = 1,
        significance: float = 0.05,
    ) -> None:
        if not isinstance(model, BaseVaR):
            raise TypeError(
                f"[model] debe ser una instancia de BaseVaR. "
                f"Recibido: '{type(model).__name__}'"
            )
        if not 0.0 < significance < 1.0:
            raise ValueError(
                f"[significance] debe estar en (0, 1). Recibido: '{significance}'"
            )

        self._model_template = model
        self.confidence: float = model.confidence
        self.step: int = step
        self.significance: float = significance

        # Atributos post-run
        self.violations_: np.ndarray | None = None
        self.var_series_: pd.Series | None = None
        self.n_violations_: int | None = None
        self.violation_rate_: float | None = None
        self.test_returns_: pd.Series | None = None
        self._fitted: bool = False

    # ── Ejecución del backtesting ─────────────────────────────────────────────

    def run_expanding(
        self,
        train: pd.Series,
        test: pd.Series,
    ) -> "Backtester":
        """
        Backtesting con ventana EXPANSIVA.

        En cada día t del test, el modelo se reajusta con todo el historial
        disponible hasta t-1 (la ventana crece). El train define el warm-up
        inicial y el test el periodo evaluado.

        Parameters
        ----------
        train : pd.Series
            Histórico inicial de entrenamiento (>= 30 observaciones).
        test : pd.Series
            Retornos sobre los que se evalúan las violaciones.

        Returns
        -------
        Backtester
            La propia instancia (permite encadenamiento).
        """
        train = train.dropna()
        test  = test.dropna()
        if len(train) < 30:
            raise ValueError(
                f"[train] debe tener al menos 30 observaciones. "
                f"Recibido: '{len(train)}'"
            )
        if len(test) < 1:
            raise ValueError("[test] debe tener al menos 1 observación.")

        all_returns = pd.concat([train, test], ignore_index=False)
        return self._evaluate(all_returns, window=len(train), expanding=True, eval_returns=test)

    def run_rolling(
        self,
        returns: pd.Series,
        window: int,
        eval_start = None,
    ) -> "Backtester":
        """
        Backtesting con ventana DESLIZANTE fija de window días.

        En cada día t el modelo se reajusta con los window días previos. No
        necesita partición train/test: cada predicción es out-of-sample por
        construcción (la ventana solo mira hacia atrás).

        Parameters
        ----------
        returns : pd.Series
            Serie completa de retornos.
        window : int
            Tamaño de la ventana deslizante (>= 1).
        eval_start : etiqueta de índice or None, optional
            Fecha desde la que se empiezan a contar violaciones.
            - None (defecto): evalúa desde el día window en adelante (toda la
              serie tras el warm-up inicial).
            - Una fecha: evalúa desde ahí, usando los window días anteriores
              como warm-up. Los datos previos a eval_start - window NO se usan
              (la ventana fija no los alcanza). Sirve para alinear la evaluación
              con un backtest expansivo sobre el mismo periodo.

        Returns
        -------
        Backtester
            La propia instancia (permite encadenamiento).
        """
        returns = returns.dropna()
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"[window] debe ser un entero >= 1. Recibido: '{window}'"
            )

        if eval_start is None:
            series       = returns
            warmup       = window
            eval_returns = returns.iloc[window:]
        else:
            pos          = returns.index.get_loc(eval_start)
            start        = max(0, pos - window)
            warmup       = pos - start          # = window si hay historia suficiente
            series       = returns.iloc[start:]
            eval_returns = returns.iloc[pos:]

        if len(eval_returns) < 1:
            raise ValueError(
                "No hay observaciones para evaluar: revisa 'window'/'eval_start'."
            )
        return self._evaluate(series, window=warmup, expanding=False, eval_returns=eval_returns)

    def _evaluate(
        self,
        series: pd.Series,
        window: int,
        expanding: bool,
        eval_returns: pd.Series,
    ) -> "Backtester":
        """
        Motor común de los dos modos: delega en dynamic_var la serie de VaR
        (sin look-ahead: el VaR de t se estima solo con datos hasta t-1), la
        alinea al periodo evaluado y cuenta las violaciones.
        """
        var_full = dynamic_var(
            self._model_template,
            series,
            window=window,
            step=self.step,
            expanding=expanding,
            on_error="previous",
        )

        self.var_series_ = var_full.reindex(eval_returns.index)
        self.var_series_.name = "VaR"

        var_vals = self.var_series_.values
        ret_vals = eval_returns.values
        # VaR es pérdida positiva: hay violación si el retorno cae por debajo
        # de -VaR (equivalente a que la pérdida supere el VaR).
        self.violations_     = ret_vals < -var_vals
        self.n_violations_   = int(self.violations_.sum())
        self.violation_rate_ = self.n_violations_ / len(eval_returns)
        self.test_returns_   = eval_returns
        self._fitted         = True
        return self

    # ── Tests estadísticos ────────────────────────────────────────────────────

    def kupiec_test(self) -> dict:
        """
        Test de Kupiec (LR_uc): contrasta cobertura incondicional.

        H0: la tasa de violaciones observada es igual a la teórica (1-conf).

        LR_uc = -2 * [V * log(p0/p_hat) + (T-V) * log((1-p0)/(1-p_hat))]
        Bajo H0: LR_uc ~ chi^2(1).

        Returns
        -------
        dict
            Claves: n_obs, n_violations, violation_rate,
            expected_rate, LR_uc, p_value, reject_H0.
        """
        self._check_run()

        T = len(self.violations_)
        V = self.n_violations_
        p0 = 1.0 - self.confidence   # tasa esperada de violaciones
        p_hat = V / T if T > 0 else 0.0

        # Log-likelihood bajo H0 (p = p0) vs. H1 (p = p_hat)
        if V == 0:
            lr_uc = -2.0 * T * np.log(1.0 - p0)
        elif V == T:
            lr_uc = -2.0 * T * np.log(p0)
        else:
            ll_h0 = V * np.log(p0) + (T - V) * np.log(1.0 - p0)
            ll_h1 = V * np.log(p_hat) + (T - V) * np.log(1.0 - p_hat)
            lr_uc = -2.0 * (ll_h0 - ll_h1)

        p_value = float(stats.chi2.sf(lr_uc, df=1))

        return {
            "n_obs":           T,
            "n_violations":    V,
            "violation_rate":  p_hat,
            "expected_rate":   p0,
            "LR_uc":           float(lr_uc),
            "p_value":         p_value,
            "reject_H0":       p_value < self.significance,
        }

    def christoffersen_test(self) -> dict:
        """
        Test de Christoffersen (LR_ind y LR_cc).

        LR_ind contrasta la independencia temporal de las violaciones.
        LR_cc = LR_uc + LR_ind contrasta cobertura condicional conjunta.

        Bajo H0:
            LR_ind ~ chi^2(1)
            LR_cc  ~ chi^2(2)

        Returns
        -------
        dict
            Claves: n00, n01, n10, n11,
            pi01, pi11, pi_hat,
            LR_ind, p_value_ind, reject_ind,
            LR_cc, p_value_cc, reject_cc.
        """
        self._check_run()

        # Construir matriz de transición
        # Emparejamos cada día con el siguiente recortando por extremos opuestos:
        # hits[:-1] -> días t     (todos menos el último)
        # hits[1:]  -> días t+1   (todos menos el primero)
        # Así la posición i de cada uno son días consecutivos
        hits = self.violations_.astype(int)
        n00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))  # 0->0
        n01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))  # 0->1
        n10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))  # 1->0
        n11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))  # 1->1

        pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi_hat = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00+n01+n10+n11) > 0 else 0.0

        # Log-likelihood ratio de independencia
        def _safe_log(x: float) -> float:
            return np.log(x) if x > 0 else 0.0

        ll_ind_h1 = (
            n00 * _safe_log(1.0 - pi01)
            + n01 * _safe_log(pi01)
            + n10 * _safe_log(1.0 - pi11)
            + n11 * _safe_log(pi11)
        )
        ll_ind_h0 = (
            (n00 + n10) * _safe_log(1.0 - pi_hat)
            + (n01 + n11) * _safe_log(pi_hat)
        )
        lr_ind = -2.0 * (ll_ind_h0 - ll_ind_h1)

        p_value_ind = float(stats.chi2.sf(lr_ind, df=1))

        # LR combinado (cobertura condicional): LR_uc + LR_ind, ambos sin redondear.
        lr_cc  = self.kupiec_test()["LR_uc"] + lr_ind
        p_value_cc = float(stats.chi2.sf(lr_cc, df=2))

        # Estadísticos sin redondear (el formateo es cosa de la presentación).
        return {
            "n00":          n00,
            "n01":          n01,
            "n10":          n10,
            "n11":          n11,
            "pi01":         pi01,
            "pi11":         pi11,
            "pi_hat":       pi_hat,
            "LR_ind":       float(lr_ind),
            "p_value_ind":  p_value_ind,
            "reject_ind":   p_value_ind < self.significance,
            "LR_cc":        float(lr_cc),
            "p_value_cc":   p_value_cc,
            "reject_cc":    p_value_cc < self.significance,
        }

    # ── Tabla resumen ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Resumen con todas las métricas de backtesting.

        Returns
        -------
        dict
            Claves: modelo, confianza, n_obs,
            n_violations, violation_rate, expected_rate,
            LR_uc, p_uc, reject_uc,
            LR_ind, p_ind, reject_ind,
            LR_cc, p_cc, reject_cc.
        """
        self._check_run()

        kup  = self.kupiec_test()
        chri = self.christoffersen_test()

        return {
            "modelo":          type(self._model_template).__name__,
            "confianza":       self.confidence,
            "n_obs":           kup["n_obs"],
            "n_violations":    kup["n_violations"],
            "violation_rate":  kup["violation_rate"],
            "expected_rate":   kup["expected_rate"],
            "LR_uc":           kup["LR_uc"],
            "p_uc":            kup["p_value"],
            "reject_uc":       kup["reject_H0"],
            "LR_ind":          chri["LR_ind"],
            "p_ind":           chri["p_value_ind"],
            "reject_ind":      chri["reject_ind"],
            "LR_cc":           chri["LR_cc"],
            "p_cc":            chri["p_value_cc"],
            "reject_cc":       chri["reject_cc"],
        }

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _check_run(self) -> None:
        """
        Lanza RuntimeError si no se ha llamado a run() previamente.
        """
        if not self._fitted:
            raise RuntimeError(
                "Backtester no ha sido ejecutado. Llama primero a "
                ".run_expanding(train, test) o .run_rolling(returns, window)."
            )

    def __repr__(self) -> str:
        model_name = type(self._model_template).__name__
        status = "ejecutado" if self._fitted else "sin ejecutar"
        if self._fitted:
            v_info = (
                f", violations={self.n_violations_}/{len(self.violations_)}"
                f" ({self.violation_rate_:.2%})"
            )
        else:
            v_info = ""
        return (
            f"Backtester(model={model_name}, "
            f"confidence={self.confidence}, "
            f"status={status}{v_info})"
        )
