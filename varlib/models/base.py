"""
varlib.models.base
==================
Clase abstracta base de la que heredan todos los modelos VaR de la librería.

Define el contrato común (interfaz) que garantiza que todos los modelos
expongan los mismos métodos y se puedan usar de forma intercambiable
en el backtesting y los reportes.

Convención de signo
-------------------
compute_var() y compute_es() devuelven la PÉRDIDA como número
POSITIVO. Con nivel de confianza (1 - alpha) y horizonte h se define:

    P(L > VaR) = alpha,   con L = -(P[t+h] - P[t]) la pérdida realizada

    VaR = 1.65 %  ->  con 95 % de confianza la pérdida no superará el 1.65 %

Hay violación cuando la pérdida supera el VaR, es decir:
    violación <-> retorno_t < -VaR

Convención de escala
--------------------
Los modelos reciben rendimientos LOGARÍTMICOS, así que el VaR y el ES se
expresan también en términos de rendimiento logarítmico (una pérdida
logarítmica). Para la pérdida porcentual real del activo se convierte con
1 - exp(-VaR). A horizontes cortos y valores pequeños log ≈ simple, por lo
que ambos casi coinciden (la diferencia solo importa en colas extremas o
horizontes largos).
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def annualize_volatility(vol, periods_per_year: int = 252):
    """
    Anualiza una volatilidad por la regla de la raíz del tiempo.

    La volatilidad (desviación típica) escala con sqrt(periodos), a
    diferencia de un retorno medio, que escala linealmente. Para datos
    diarios de bolsa periods_per_year = 252.

    Parameters
    ----------
    vol : float, np.ndarray or pd.Series
        Volatilidad en frecuencia base (p.ej. diaria), escala decimal.
    periods_per_year : int, optional
        Periodos de negociación por año (252 diario, 52 semanal, 12
        mensual). Por defecto 252.

    Returns
    -------
    Igual tipo que vol
        Volatilidad anualizada (vol * sqrt(periods_per_year)).
    """
    return vol * np.sqrt(periods_per_year)


class BaseVaR(ABC):
    """
    Clase abstracta base para todos los modelos VaR.

    Todos los modelos concretos deben heredar de esta clase e implementar
    fit y compute_var. El resto de métodos tienen
    implementaciones por defecto que los subclases pueden sobrescribir.

    Parameters
    ----------
    confidence : float, optional
        Nivel de confianza del VaR, p.ej. 0.95 o 0.99.
        Debe estar en el intervalo abierto (0, 1). Por defecto 0.95.
    horizon : int, optional
        Horizonte temporal en días. Se aplica la regla de la raíz del
        tiempo (sqrt(h)) para escalar el VaR diario. Por defecto 1.

    Attributes
    ----------
    confidence : float
    horizon : int
    _fitted : bool
        True una vez que se ha llamado a fit.
    _returns : pd.Series or None
        Serie de rendimientos almacenada durante el ajuste.
    n_obs_ : int or None
        Número de observaciones usadas en el ajuste (None si no ajustado).
        Property derivada de _returns, común a todos los modelos.

    Raises
    ------
    ValueError
        Si confidence no está en (0, 1) o horizon < 1.
    RuntimeError
        Si se llama a compute_var, compute_es o
        summary antes de fit.
    """

    def __init__(self, confidence: float = 0.95, horizon: int = 1) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError(
                f"[confidence] debe estar en (0, 1). Recibido: '{confidence}'"
            )
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError(
                f"[horizon] debe ser un entero >= 1. Recibido: '{horizon}'"
            )

        self.confidence: float = confidence
        self.horizon: int = horizon
        self._fitted: bool = False
        self._returns: pd.Series | None = None

    @property
    def n_obs_(self) -> int | None:
        """
        Número de observaciones usadas en el ajuste (None si no ajustado).
        """
        return len(self._returns) if self._returns is not None else None

    # ── Interfaz abstracta ────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, returns: pd.Series) -> "BaseVaR":
        """
        Ajusta el modelo a una serie de rendimientos históricos.

        Parameters
        ----------
        returns : pd.Series
            Serie de rendimientos (logarítmicos o simples). Debe tener
            al menos 30 observaciones para estimaciones estables.

        Returns
        -------
        BaseVaR
            La propia instancia, para permitir encadenamiento:
            model.fit(returns).compute_var().
        """

    @abstractmethod
    def compute_var(self) -> float:
        """
        Calcula el Value at Risk al nivel de confianza configurado.

        Returns
        -------
        float
            VaR como pérdida positiva. Ejemplo: 0.0165 significa que,
            con la confianza dada, la pérdida no superará el 1.65 % del
            valor de la cartera en el horizonte indicado.

        Raises
        ------
        RuntimeError
            Si el modelo no ha sido ajustado previamente.
        """

    # ── Implementaciones por defecto ──────────────────────────────────────────

    def compute_es(self) -> float:
        """
        Calcula el ES (Expected Shortfall) de forma empírica.

        El ES (Expected Shortfall), también conocido como CVaR, es la
        pérdida media condicionada a que se supere el VaR: la pérdida
        esperada dado que se ha producido una violación.

        Los subclases con fórmulas cerradas (p.ej. Normal) pueden
        sobrescribir este método para mayor precisión.

        Returns
        -------
        float
            ES como pérdida positiva (peor pérdida esperada en la cola).

        Raises
        ------
        RuntimeError
            Si el modelo no ha sido ajustado previamente.
        """
        self._check_fitted()
        var = self.compute_var()                 # positivo (pérdida)
        threshold = -var                          # umbral de retorno
        tail = self._returns[self._returns <= threshold]
        if len(tail) == 0:
            # Sin observaciones en la cola: el ES coincide con el VaR
            return var
        return float(-tail.mean())

    def summary(self) -> dict:
        """
        Devuelve un diccionario resumen con los resultados del modelo.

        Returns
        -------
        dict
            Claves mínimas garantizadas por todos los modelos:
            model, confidence, horizon, var, es,
            n_obs.

        Raises
        ------
        RuntimeError
            Si el modelo no ha sido ajustado previamente.
        """
        self._check_fitted()
        return {
            "model":      self.__class__.__name__,
            "confidence": self.confidence,
            "horizon":    self.horizon,
            "var":        self.compute_var(),
            "es":         self.compute_es(),
            "n_obs":      self.n_obs_,
        }

    # ── Utilidades internas ───────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        """
        Lanza RuntimeError si el modelo todavía no ha sido ajustado.
        """
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} no ha sido ajustado aún. "
                "Llama primero a .fit(returns)."
            )

    def _validate_returns(self, returns: pd.Series, min_obs: int = 30) -> None:
        """
        Valida la serie de rendimientos antes de ajustar.

        Parameters
        ----------
        returns : pd.Series
            Serie de rendimientos a validar.
        min_obs : int
            Número mínimo de observaciones requeridas.

        Raises
        ------
        TypeError
            Si returns no es un pd.Series.
        ValueError
            Si la serie tiene demasiados NaN o menos de min_obs datos.
        """
        if not isinstance(returns, pd.Series):
            raise TypeError(
                f"[returns] debe ser un pd.Series. "
                f"Recibido: '{type(returns).__name__}'"
            )
        n_valid = returns.dropna().shape[0]
        if n_valid < min_obs:
            raise ValueError(
                f"La serie de rendimientos tiene solo {n_valid} observaciones "
                f"válidas (mínimo requerido: {min_obs})."
            )
        if returns.isna().any():
            # Aviso pero no error: el subclase decide cómo manejarlos
            import warnings
            n_nan = returns.isna().sum()
            warnings.warn(
                f"La serie contiene {n_nan} valores NaN que serán ignorados.",
                UserWarning,
                stacklevel=3,
            )

    # ── Representación ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "ajustado" if self._fitted else "sin ajustar"
        return (
            f"{self.__class__.__name__}("
            f"confidence={self.confidence}, "
            f"horizon={self.horizon}, "
            f"status={status})"
        )
