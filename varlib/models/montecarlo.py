"""
varlib.models.montecarlo
========================
VaR por simulación de Monte Carlo con volatilidad constante.

Modelo
------
La librería trabaja con rendimientos LOGARÍTMICOS, que son aditivos en el
tiempo: el rendimiento acumulado a horizonte h es la suma de h rendimientos
diarios. Bajo el supuesto de rendimientos i.i.d. Normales se simula:

    r_acum = mu*h + sigma*sqrt(h)*Z ,       Z ~ N(0,1)

con mu y sigma estimados de la muestra histórica.

El VaR y el ES (Expected Shortfall, también CVaR) se obtienen como percentiles de la distribución
empírica de las n_simulations trayectorias:

    VaR  = -Q[1-alpha]( r_acum )
    ES = -E[ r_acum | r_acum <= Q[1-alpha](r_acum) ]

Nota sobre la corrección de Itô
--------------------------------
NO se aplica la corrección -sigma^2/2 del GBM. Esa corrección convierte una
deriva aritmética (la del retorno simple) en logarítmica, pero aquí mu ya es
la media de los rendimientos LOGARÍTMICOS, de modo que restarla la contaría
dos veces. Así el Monte Carlo parte de la misma deriva (mu*h) que el resto
de modelos (paramétrico, histórico).
"""

import numpy as np
import pandas as pd

from varlib.models.base import BaseVaR, annualize_volatility


class MonteCarloVaR(BaseVaR):
    """
    VaR por simulación de Monte Carlo con volatilidad constante (GBM).

    Parameters
    ----------
    confidence : float, optional
        Nivel de confianza. Por defecto 0.95.
    horizon : int, optional
        Horizonte temporal en días. Por defecto 1.
    n_simulations : int, optional
        Número de trayectorias a simular. Un mínimo de 10 000 es
        recomendable para la cola del 1%. Por defecto 10_000.
    random_state : int or None, optional
        Semilla del generador de números aleatorios para reproducibilidad.
        None (defecto) produce resultados distintos en cada ejecución.

    Attributes
    ----------
    mu_ : float
        Media muestral de los rendimientos diarios (post-fit).
    sigma_ : float
        Desviación típica muestral (ddof=1) de los rendimientos (post-fit).
    simulated_returns_ : np.ndarray, shape (n_simulations,)
        Rendimientos acumulados simulados al horizonte h (post-simulate).

    Examples
    --------
    from varlib.models.montecarlo import MonteCarloVaR
    model = MonteCarloVaR(confidence=0.95, horizon=1,
                          n_simulations=50_000, random_state=42)
    model.fit(train_returns)
    model.simulate()
    model.compute_var()
    model.compute_es()
    """

    def __init__(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        n_simulations: int = 10_000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(confidence, horizon)

        if not isinstance(n_simulations, int) or n_simulations < 1:
            raise ValueError(
                f"[n_simulations] debe ser un entero >= 1. "
                f"Recibido: '{n_simulations}'"
            )

        self.n_simulations: int = n_simulations
        self.random_state: int | None = random_state

        # Atributos post-fit / post-simulate
        self.mu_: float | None = None
        self.sigma_: float | None = None
        self.simulated_returns_: np.ndarray | None = None

    # ── Ajuste ────────────────────────────────────────────────────────────────

    def fit(self, returns: pd.Series) -> "MonteCarloVaR":
        """
        Estima mu y sigma a partir de los rendimientos históricos.

        No lanza las simulaciones todavía. Llama a simulate
        para generarlas (o usa directamente compute_var, que
        llama a simulate automáticamente si es necesario).

        Parameters
        ----------
        returns : pd.Series
            Serie de rendimientos históricos.

        Returns
        -------
        MonteCarloVaR
            La propia instancia (permite encadenamiento).
        """
        self._validate_returns(returns)
        clean = returns.dropna()

        self._returns  = clean
        self.mu_       = float(clean.mean())
        self.sigma_    = float(clean.std(ddof=1))
        self._fitted   = True
        # Invalidar simulaciones previas si se reajusta
        self.simulated_returns_ = None
        return self

    # ── Simulación ────────────────────────────────────────────────────────────

    def simulate(self) -> "MonteCarloVaR":
        """
        Genera las trayectorias de Monte Carlo.

        Simula n_simulations rendimientos logarítmicos acumulados a
        horizonte h (aditivos, bajo retornos i.i.d. Normales):

            r_acum = mu*h + sigma*sqrt(h)*Z,   Z ~ N(0,1)

        Los resultados se almacenan en simulated_returns_.

        Returns
        -------
        MonteCarloVaR
            La propia instancia (permite encadenamiento).

        Raises
        ------
        RuntimeError
            Si no se ha llamado a fit previamente.
        """
        self._check_fitted()

        rng = np.random.default_rng(self.random_state)
        Z = rng.standard_normal(self.n_simulations)

        # Se asumen rendimientos LOGARÍTMICOS (aditivos): media mu*h y desviación
        # sigma*sqrt(h) por la regla de la raíz del tiempo.
        drift = self.mu_ * self.horizon
        diffusion = self.sigma_ * np.sqrt(self.horizon) * Z

        self.simulated_returns_ = drift + diffusion
        return self

    def _ensure_simulated(self) -> None:
        """
        Lanza la simulación si todavía no se ha ejecutado.
        """
        self._check_fitted()
        if self.simulated_returns_ is None:
            self.simulate()

    # ── VaR y ES ────────────────────────────────────────────────────────────

    def compute_var(self) -> float:
        """
        Calcula el VaR como percentil de los retornos simulados.

        Llama a simulate automáticamente si no se ha hecho antes.

        Returns
        -------
        float
            VaR como pérdida positiva.
        """
        self._ensure_simulated()
        q = float(np.quantile(self.simulated_returns_, 1.0 - self.confidence))
        return -q   # pérdida positiva = negativo del cuantil de retorno

    def compute_es(self) -> float:
        """
        Calcula el ES como pérdida media (positiva) de la cola de
        retornos simulados.

        Returns
        -------
        float
            ES como pérdida positiva.
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
        Resumen del modelo Monte Carlo con metadatos de la simulación.

        Returns
        -------
        dict
            Extiende el resumen base con: mu, sigma,
            annualized_vol, n_simulations, random_state,
            simulated (bool).
        """
        base = super().summary()
        base.update({
            "mu":             self.mu_,
            "sigma":          self.sigma_,
            "annualized_vol": annualize_volatility(self.sigma_),
            "n_simulations":  self.n_simulations,
            "random_state":   self.random_state,
            "simulated":      self.simulated_returns_ is not None,
        })
        return base
