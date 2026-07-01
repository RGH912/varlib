"""
varlib.models.historical
========================
VaR histórico (no paramétrico) basado en la distribución empírica
de los rendimientos.

Modelo
------
No se asume ninguna distribución teórica. El VaR (pérdida positiva) se
obtiene como el percentil empírico de la muestra, cambiado de signo:

    VaR(alpha, h) = -Q[1-alpha](r)  * sqrt(h)

donde Q[1-alpha](r) es el cuantil (1-alpha) de los rendimientos históricos
(negativo) y sqrt(h) aplica la regla de la raíz del tiempo al horizonte h.

El ES (Expected Shortfall), también conocido como CVaR, se calcula de forma empírica:

    ES(alpha, h) = -E[r | r <= Q[1-alpha](r)] * sqrt(h)

Ventajas frente al paramétrico
-------------------------------
- No asume normalidad: captura fat tails y asimetría reales.
- Simple e interpretable: el VaR corresponde a un día histórico real.

Limitaciones
------------
- Sensible al tamaño de la muestra (mínimo recomendado: 250 obs.).
- Asigna el mismo peso a todas las observaciones, sin distinguir entre
  las más recientes y las más antiguas.
"""

import numpy as np
import pandas as pd

from varlib.models.base import BaseVaR


class HistoricalVaR(BaseVaR):
    """
    VaR histórico no paramétrico basado en la distribución empírica.

    Todas las observaciones reciben el mismo peso: el VaR es el percentil
    (1 - confidence) de la muestra de rendimientos.

    Parameters
    ----------
    confidence : float, optional
        Nivel de confianza. Por defecto 0.95.
    horizon : int, optional
        Horizonte en días. Por defecto 1.

    Attributes
    ----------
    quantile_level_ : float
        Nivel de cuantil usado: 1 - confidence.

    Examples
    --------
    from varlib.models.historical import HistoricalVaR
    model = HistoricalVaR(confidence=0.95, horizon=1)
    model.fit(train_returns).compute_var()
    """

    def __init__(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> None:
        super().__init__(confidence, horizon)

        # Atributos post-fit
        self.quantile_level_: float | None = None

    # ── Ajuste ────────────────────────────────────────────────────────────────

    def fit(self, returns: pd.Series) -> "HistoricalVaR":
        """
        Almacena la distribución empírica de rendimientos.

        Parameters
        ----------
        returns : pd.Series
            Serie de rendimientos históricos. Se recomienda un mínimo
            de 250 observaciones para una estimación robusta.

        Returns
        -------
        HistoricalVaR
            La propia instancia (permite encadenamiento).
        """
        self._validate_returns(returns, min_obs=30)
        clean = returns.dropna()

        self._returns = clean
        self.quantile_level_ = 1.0 - self.confidence
        self._fitted = True
        return self

    # ── VaR y ES ────────────────────────────────────────────────────────────

    def compute_var(self) -> float:
        """
        Calcula el VaR histórico como percentil empírico.

        Usa np.quantile (interpolación lineal) sobre la muestra y
        escala al horizonte: VaR(h) = VaR(1) * sqrt(h).

        Returns
        -------
        float
            VaR como pérdida positiva.
        """
        self._check_fitted()
        # Pérdida positiva = negativo del cuantil (1 - confidence) de retorno.
        return -self._var_1d() * np.sqrt(self.horizon)

    def compute_es(self) -> float:
        """
        Calcula el ES (Expected Shortfall) histórico.

        Pérdida media (positiva) de los rendimientos que caen por debajo
        del cuantil a horizonte 1 día, escalada luego por sqrt(h).

        Returns
        -------
        float
            ES como pérdida positiva.
        """
        self._check_fitted()
        var_1d = self._var_1d()   # cuantil de retorno (negativo)

        tail = self._returns.values[self._returns.values <= var_1d]
        if len(tail) == 0:
            return -var_1d * np.sqrt(self.horizon)

        return -float(tail.mean()) * np.sqrt(self.horizon)

    # ── Utilidad interna ──────────────────────────────────────────────────────

    def _var_1d(self) -> float:
        """
        Percentil empírico (1 - confidence) a horizonte 1 día.
        """
        return float(np.quantile(self._returns.values, self.quantile_level_))
