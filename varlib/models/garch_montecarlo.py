"""
varlib.models.garch_montecarlo
================================
VaR por simulación de Monte Carlo con volatilidad dinámica GARCH(p,q).

Motivación
----------
GARCHParametricVaR usa las fórmulas analíticas del modelo GARCH:
es exacto pero asume normalidad (o distribución paramétrica fija) en
la cola.  GARCHMonteCarloVaR propaga la dinámica GARCH paso a paso
simulando la trayectoria completa de la volatilidad condicional, lo que
permite:

- Capturar el clustering de volatilidad en horizontes multi-día sin
  simplificaciones.
- Obtener VaR y ES empíricos directamente de la distribución simulada
  sin supuestos adicionales sobre su forma.
- Propagación correcta de la incertidumbre a largo plazo.

Algoritmo de simulación
-----------------------
Se parte del estado filtrado en T (último punto de la muestra):

    sigma^2[T+1] = omega + alpha*epsilon^2[T] + beta*sigma^2[T]

Para cada simulación i = 1...N y cada paso k = 1...h:

    z[k]     ~ F_inn(0, 1)               [innovación]
    epsilon[k]     = sigma[k] * z[k]
    r[k]     = mu + epsilon[k]                 [retorno del paso k]
    sigma^2[k+1]  = omega + alpha*epsilon^2[k] + beta*sigma^2[k] [actualizar varianza]

Retorno acumulado de la simulación i:  R[i] = sum[k=1]^{h} r[k]

El VaR y ES (Expected Shortfall, también CVaR) se obtienen como percentiles empíricos de {R[i]}.

Implementación vectorizada
--------------------------
El bucle externo es sobre los h pasos del horizonte (típicamente
1-22 días), pero cada paso opera en paralelo sobre las N simulaciones
usando arrays de NumPy.  Para N=50 000 y h=10 el coste es despreciable.

Distribución de innovaciones
----------------------------
Las innovaciones son siempre Normales estándar N(0,1). La asimetría y las
colas pesadas del retorno acumulado surgen de la propia dinámica GARCH y
del clustering de volatilidad, no de la distribución de las innovaciones.

Limitación: solo GARCH(1,1) en la simulación
--------------------------------------------
La estimación admite órdenes (p, q) cualesquiera (se ajustan todos los
coeficientes alpha[i], beta[j]), pero la recursión de simulación usa
únicamente el primer rezago de cada suma (alpha[1] y beta[1]), de modo que
la propagación de la varianza es efectivamente GARCH(1,1). Con p > 1 o
q > 1 el VaR/ES simulado NO es correcto. Soportar órdenes mayores en la
simulación (manteniendo el historial de los p residuos y las q varianzas)
queda como mejora futura.
"""

import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
from arch import arch_model

from varlib.models.base import BaseVaR


_SCALE = 100.0   # escala porcentual para estabilidad numérica del MLE


class GARCHMonteCarloVaR(BaseVaR):
    """
    VaR Monte Carlo con dinámica de volatilidad GARCH(p,q).

    Combina la estimación GARCH (por MLE con arch) con la simulación
    de trayectorias completas de retornos, propagando la varianza condicional
    paso a paso durante el horizonte deseado.

    Parameters
    ----------
    confidence : float, optional
        Nivel de confianza. Por defecto 0.95.
    horizon : int, optional
        Horizonte temporal en días. Por defecto 1.
    p : int, optional
        Orden ARCH (términos epsilon^2). Por defecto 1. Nota: la simulación
        solo propaga el primer rezago. Usar p > 1 da un VaR/ES incorrecto
        (véase la limitación en el docstring del módulo).
    q : int, optional
        Orden GARCH (términos sigma^2). Por defecto 1. Nota: la simulación
        solo propaga el primer rezago. Usar q > 1 da un VaR/ES incorrecto
        (véase la limitación en el docstring del módulo).
    n_simulations : int, optional
        Número de trayectorias a simular. Por defecto 10_000.
        Se recomienda >= 50 000 para la cola del 1 %.
    random_state : int or None, optional
        Semilla para reproducibilidad. None produce resultados
        distintos en cada ejecución.

    Attributes
    ----------
    mu_ : float
        Media condicional estimada, escala decimal (post-fit).
    omega_ : float
        Parámetro omega en escala decimal^2 (post-fit).
    alpha_ : list[float]
        Coeficientes alpha[i] (post-fit).
    beta_ : list[float]
        Coeficientes beta[j] (post-fit).
    persistence_ : float
        sum(alpha) + sum(beta) (post-fit).
    conditional_variance_ : float
        Varianza condicional en T, escala decimal^2 (post-fit).
    sigma2_next_ : float
        Varianza condicional 1-step ahead sigma^2[T+1], escala decimal^2
        (post-fit). Punto de inicio de las simulaciones.
    simulated_returns_ : np.ndarray or None
        Retornos acumulados simulados, escala decimal (post-simulate).

    Examples
    --------
    from varlib.models.garch_montecarlo import GARCHMonteCarloVaR
    model = GARCHMonteCarloVaR(confidence=0.95, horizon=1,
                                n_simulations=50_000, random_state=42)
    model.fit(train_returns)
    model.simulate_paths()
    model.compute_var()
    model.compute_es()
    """

    def __init__(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        n_simulations: int = 10_000,
        random_state: int | None = None,
        p: int = 1,
        q: int = 1,
    ) -> None:
        super().__init__(confidence, horizon)

        if not isinstance(p, int) or p < 1:
            raise ValueError(f"[p] debe ser un entero >= 1. Recibido: '{p}'")
        if not isinstance(q, int) or q < 1:
            raise ValueError(f"[q] debe ser un entero >= 1. Recibido: '{q}'")
        if not isinstance(n_simulations, int) or n_simulations < 1:
            raise ValueError(
                f"[n_simulations] debe ser un entero >= 1. Recibido: '{n_simulations}'"
            )

        # La simulación solo implementa la recursión GARCH(1,1) (véase la
        # limitación en el docstring del módulo): lanzar warning.
        if p > 1 or q > 1:
            warnings.warn(
                f"GARCHMonteCarloVaR: por ahora la simulación solo funciona con "
                f"GARCH(1,1). Otros órdenes aún no están implementados (queda como "
                f"mejora futura). Con p={p}, q={q} se usará igualmente un (1,1).",
                UserWarning,
                stacklevel=2,
            )

        self.p: int = p
        self.q: int = q
        self.n_simulations: int = n_simulations
        self.random_state: int | None = random_state

        # Atributos post-fit
        self.garch_result_            = None
        self.mu_: float | None        = None
        self.omega_: float | None     = None
        self.alpha_: list[float]      = []
        self.beta_: list[float]       = []
        self.persistence_: float | None       = None
        self.conditional_variance_: float | None = None
        self.sigma2_next_: float | None          = None

        # Atributos post-simulate
        self.simulated_returns_: np.ndarray | None = None

    # ── Ajuste ────────────────────────────────────────────────────────────────

    def fit(self, returns: pd.Series) -> "GARCHMonteCarloVaR":
        """
        Ajusta el modelo GARCH(p,q) por MLE y prepara las condiciones
        iniciales para la simulación.

        Parameters
        ----------
        returns : pd.Series
            Serie de rendimientos históricos.

        Returns
        -------
        GARCHMonteCarloVaR
            La propia instancia (permite encadenamiento).
        """
        self._validate_returns(returns, min_obs=50)
        clean  = returns.dropna()
        scaled = clean * _SCALE

        am = arch_model(
            scaled,
            vol="Garch",
            p=self.p,
            q=self.q,
            dist="normal",
            mean="Constant",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off")

        if res.convergence_flag != 0:
            warnings.warn(
                f"GARCHMonteCarloVaR: el optimizador no convergió "
                f"(flag={res.convergence_flag}). "
                "Los resultados pueden ser poco fiables.",
                RuntimeWarning,
                stacklevel=2,
            )

        params = res.params

        # Parámetros en escala decimal
        self.mu_    = float(params.get("mu", 0.0)) / _SCALE
        self.omega_ = float(params["omega"]) / (_SCALE ** 2)
        self.alpha_ = [float(params[f"alpha[{i}]"]) for i in range(1, self.p + 1)]
        self.beta_  = [float(params[f"beta[{i}]"])  for i in range(1, self.q + 1)]

        self.persistence_ = sum(self.alpha_) + sum(self.beta_)

        # sigma^2[T] y epsilon[T] para inicializar la primera trayectoria
        sigma_T_pct = float(res.conditional_volatility.iloc[-1])   # %
        eps_T_pct = float(res.resid.iloc[-1])                      # %

        self.conditional_variance_ = (sigma_T_pct / _SCALE) ** 2

        # sigma^2[T+1] = omega_pct + alpha*epsilon^2[T] + beta*sigma^2[T]  (en escala %^2)
        # para GARCH(p,q) genérico tomamos solo el lag 1 de cada suma, solo hacemos GARCH(1,1)
        omega_pct   = self.omega_ * (_SCALE ** 2)
        sigma2_T_pct = sigma_T_pct ** 2
        sigma2_next_pct = (
            omega_pct + self.alpha_[0] * eps_T_pct ** 2
            + self.beta_[0]  * sigma2_T_pct
        )
        self.sigma2_next_ = sigma2_next_pct / (_SCALE ** 2)   # decimal^2

        self.garch_result_       = res
        self._returns            = clean
        self._fitted             = True
        self.simulated_returns_  = None   # invalidar simulaciones previas
        return self

    # ── Simulación ────────────────────────────────────────────────────────────

    def simulate_paths(self) -> "GARCHMonteCarloVaR":
        """
        Genera N trayectorias GARCH de longitud h y almacena los
        retornos acumulados en simulated_returns_.

        El bucle externo itera sobre los h pasos del horizonte, y cada
        paso es completamente vectorizado sobre las N simulaciones.

        Returns
        -------
        GARCHMonteCarloVaR
            La propia instancia (permite encadenamiento).

        Raises
        ------
        RuntimeError
            Si no se ha llamado a fit previamente.
        """
        self._check_fitted()

        N   = self.n_simulations
        h   = self.horizon
        rng = np.random.default_rng(self.random_state)

        # Innovaciones Normales estándar: shape (h, N) -> columna = una simulación
        Z = rng.standard_normal((h, N))

        # Parámetros en escala porcentual para la recursión GARCH
        omega_pct = self.omega_ * (_SCALE ** 2)
        alpha     = self.alpha_[0]
        beta      = self.beta_[0]
        mu_pct    = self.mu_ * _SCALE

        # Condición inicial: sigma^2[T+1] en %^2  (igual para todas las sims)
        sigma2 = np.full(N, self.sigma2_next_ * (_SCALE ** 2))   # shape (N,)

        cumulative_pct = np.zeros(N)   # retorno acumulado en %

        for k in range(h):
            eps           = np.sqrt(sigma2) * Z[k]     # residuos en %
            cumulative_pct += mu_pct + eps             # r[k] = mu + epsilon[k]

            if k < h - 1:
                # Actualizar varianza para el siguiente paso
                sigma2 = omega_pct + alpha * eps ** 2 + beta * sigma2
                # Clamp para evitar varianzas negativas por errores numéricos
                np.clip(sigma2, 1e-12, None, out=sigma2)

        # Convertir a escala decimal y almacenar
        self.simulated_returns_ = cumulative_pct / _SCALE
        return self

    def _ensure_simulated(self) -> None:
        """
        Lanza la simulación automáticamente si todavía no se ha hecho.
        """
        self._check_fitted()
        if self.simulated_returns_ is None:
            self.simulate_paths()

    # ── VaR y ES ────────────────────────────────────────────────────────────

    def compute_var(self) -> float:
        """
        Calcula el VaR como percentil empírico de los retornos simulados.

        Llama a simulate_paths automáticamente si es necesario.

        Returns
        -------
        float
            VaR como pérdida positiva (escala decimal).
        """
        self._ensure_simulated()
        q = 1.0 - self.confidence
        # Pérdida positiva = negativo del cuantil de retorno simulado.
        return -float(np.quantile(self.simulated_returns_, q))

    def compute_es(self) -> float:
        """
        Calcula el ES como pérdida media (positiva) de la cola de
        retornos simulados.

        Returns
        -------
        float
            ES como pérdida positiva (escala decimal).
        """
        self._ensure_simulated()
        q = np.quantile(self.simulated_returns_, 1.0 - self.confidence)
        tail = self.simulated_returns_[self.simulated_returns_ <= q]
        if len(tail) == 0:
            return float(-q)
        return float(-tail.mean())

    # ── Resumen ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Resumen completo: parámetros GARCH + metadatos de simulación.

        Returns
        -------
        dict
            Extiende el resumen base con parámetros GARCH, estadísticos
        de la distribución simulada y metadatos de la simulación.
        """
        base = super().summary()

        # super().summary() invoca compute_var(), que simula si aún no se había
        # hecho, así que aquí simulated_returns_ siempre está disponible.
        s = self.simulated_returns_
        base.update({
            "p":                  self.p,
            "q":                  self.q,
            "mu":                 self.mu_,
            "omega":              self.omega_,
            "alpha":              self.alpha_,
            "beta":               self.beta_,
            "persistence":        self.persistence_,
            "conditional_vol_T":  (
                np.sqrt(self.conditional_variance_)
                if self.conditional_variance_ is not None else None
            ),
            "sigma2_next":        self.sigma2_next_,
            "n_simulations":      self.n_simulations,
            "random_state":       self.random_state,
            "simulated":          s is not None,
            "sim_mean":           float(s.mean()),
            "sim_std":            float(s.std()),
            "sim_skew":           float(stats.skew(s)),
            "sim_kurt":           float(stats.kurtosis(s)),
        })
        return base

    # ── Representación ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "ajustado" if self._fitted else "sin ajustar"
        simulated = (
            f", sims={len(self.simulated_returns_):,}"
            if self.simulated_returns_ is not None else ""
        )
        return (
            f"GARCHMonteCarloVaR("
            f"confidence={self.confidence}, "
            f"horizon={self.horizon}, "
            f"p={self.p}, q={self.q}, "
            f"n_simulations={self.n_simulations:,}, "
            f"status={status}{simulated})"
        )
