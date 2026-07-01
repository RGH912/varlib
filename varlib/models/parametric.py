"""
varlib.models.parametric
========================
VaR paramétrico con volatilidad constante y distribución estandarizada
configurable (Normal, t de Student o t asimétrica de Hansen).

Modelo
------
Los rendimientos se modelan como media más un término con distribución
estandarizada y escala constante:

    r[t] = mu + sigma * z[t]

donde z[t] sigue una distribución estandarizada (media 0, varianza 1)
elegida por el usuario. La estimación de mu y sigma depende de la
distribución:

- Normal      : mu y sigma son los estadísticos muestrales (media y
                desviación típica).
- t / skew-t  : mu, sigma y los parámetros de forma se estiman CONJUNTAMENTE
                por Máxima Verosimilitud (vía arch), coherente con el GARCH.

El VaR (pérdida positiva) al nivel de confianza (1 - alpha) y horizonte h es:

    VaR(alpha, h) = -(mu*h + q[alpha] * sigma * sqrt(h))

donde q[alpha] < 0 es el cuantil (1 - alpha) de la distribución de los
rendimientos estandarizados:

- Normal : q[alpha] = Phi^-1(1 - alpha).
- t      : cuantil de la t de Student estandarizada a varianza 1, con nu
           grados de libertad estimados por Máxima Verosimilitud.
- skew-t : cuantil de la t asimétrica de Hansen (nu, lambda) estimada por MLE.

El ES (Expected Shortfall), también conocido como CVaR, se obtiene de forma cerrada para la Normal y
por cuadratura del cuantil para t y skew-t:

    ES(alpha, h) = -(mu*h + ES_alpha * sigma * sqrt(h)),   ES_alpha = (1/alpha) integral[0]^alpha q(u) du

Ambos se devuelven como pérdidas positivas.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from arch import arch_model
from arch.univariate import SkewStudent, StudentsT

from varlib.models.base import BaseVaR, annualize_volatility


# Factor de escala: rendimientos -> porcentaje para estabilidad numérica del MLE
_SCALE = 100.0

# Distribuciones admitidas (alias -> nombre interno)
_VALID_DISTS: tuple[str, ...] = ("normal", "t", "skewt")


class ParametricVaR(BaseVaR):
    """
    VaR paramétrico con volatilidad constante y distribución configurable.

    Estima la media y la desviación típica de los rendimientos históricos
    y aplica el cuantil de la distribución estandarizada elegida. Con
    dist='normal' equivale al paramétrico Normal clásico. Con 't' o
    'skewt' captura colas pesadas (y asimetría en el caso skew-t).

    Parameters
    ----------
    confidence : float, optional
        Nivel de confianza. Por defecto 0.95.
    horizon : int, optional
        Horizonte en días. Por defecto 1.
    dist : str, optional
        Distribución estandarizada: 'normal' (defecto), 't',
        'skewt'. Para 't'/'skewt' los parámetros de forma se estiman por
        Máxima Verosimilitud sobre los residuos estandarizados.

    Attributes
    ----------
    mu_ : float
        Media muestral de los rendimientos (post-fit).
    sigma_ : float
        Desviación típica muestral (ddof=1) de los rendimientos (post-fit).
    z_ : float
        Cuantil (1 - confidence) de la distribución de los rendimientos
        estandarizados (post-fit). Siempre negativo para confidence > 0.5.
    nu_ : float or None
        Grados de libertad estimados si dist='t' o 'skewt' (post-fit).
        None para la Normal.
    lambda_ : float or None
        Parámetro de asimetría estimado si dist='skewt' (post-fit).
        None en otro caso.

    Examples
    --------
    from varlib.models.parametric import ParametricVaR
    model = ParametricVaR(confidence=0.95, horizon=1, dist='t')
    model.fit(returns)
    model.compute_var()
    model.compute_es()
    model.summary()
    """

    def __init__(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        dist: str = "normal",
    ) -> None:
        super().__init__(confidence, horizon)

        dist_lower = dist.lower()
        if dist_lower not in _VALID_DISTS:
            raise ValueError(
                f"[dist] '{dist}' no reconocido. "
                f"Opciones válidas: {list(_VALID_DISTS)}"
            )
        self.dist: str = dist_lower

        # Atributos que se rellenan en fit()
        self.mu_: float | None = None
        self.sigma_: float | None = None
        self.z_: float | None = None
        self.nu_: float | None = None
        self.lambda_: float | None = None

        # Distribución estandarizada ajustada (solo t/skewt)
        self._dist_obj = None
        self._dist_params: np.ndarray | None = None

    # ── Ajuste ────────────────────────────────────────────────────────────────

    def fit(self, returns: pd.Series) -> "ParametricVaR":
        """
        Para la Normal estima mu y sigma con los estadísticos muestrales.
        Para t / skew-t estima mu, sigma y los parámetros de forma de la
        distribución estandarizada conjuntamente por Máxima Verosimilitud
        (arch).

        Parameters
        ----------
        returns : pd.Series
            Serie de rendimientos históricos (logarítmicos recomendados).
            Se eliminan NaN antes de la estimación.

        Returns
        -------
        ParametricVaR
            La propia instancia (permite encadenamiento).

        Raises
        ------
        TypeError
            Si returns no es un pd.Series.
        ValueError
            Si la serie tiene menos de 30 observaciones válidas.

        Examples
        --------
        model = ParametricVaR(confidence=0.99, horizon=10, dist='skewt')
        model.fit(train_returns).compute_var()
        """
        self._validate_returns(returns)
        clean = returns.dropna()

        self._returns = clean

        alpha = 1.0 - self.confidence

        if self.dist == "normal":
            self.mu_    = float(clean.mean())
            self.sigma_ = float(clean.std(ddof=1))
            self.z_     = float(norm.ppf(alpha))
        else:
            # Estimación CONJUNTA por máxima verosimilitud (media, escala y
            # parámetros de forma a la vez) con arch: un modelo de media y
            # varianza constantes con la distribución elegida.
            scaled = clean * _SCALE   # porcentaje para mejor condicionamiento
            am = arch_model(scaled, mean="Constant", vol="Constant", dist=self.dist)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off")
            params = res.params

            # Parámetros de localización y escala -> escala decimal.
            self.mu_    = float(params["mu"]) / _SCALE
            self.sigma_ = float(np.sqrt(params["sigma2"])) / _SCALE

            # arch nombra 'nu' a los gl de la t, y 'eta'/'lambda' a los de la skew-t.
            # Los parámetros de forma (nu, lambda) son adimensionales: no dependen
            # de la escala, así que NO se dividen por _SCALE (a diferencia de mu/sigma).
            dist_obj = StudentsT() if self.dist == "t" else SkewStudent()

            if self.dist == "t":
                self.nu_     = float(params["nu"])
                self._dist_params = np.array([self.nu_])
            else:
                self.nu_     = float(params["eta"])
                self.lambda_ = float(params["lambda"])
                self._dist_params = np.array([self.nu_, self.lambda_])

            self._dist_obj = dist_obj
            self.z_ = float(np.asarray(dist_obj.ppf(np.array([alpha]), self._dist_params))[0])

        self._fitted = True
        return self

    # ── VaR y ES ────────────────────────────────────────────────────────────

    def compute_var(self) -> float:
        """
        Calcula el VaR paramétrico.

        Fórmula:
            VaR = -(mu*h + q[alpha] * sigma * sqrt(h))

        donde q[alpha] es el cuantil de la distribución de los rendimientos
        estandarizados configurada.

        Returns
        -------
        float
            VaR como pérdida positiva.

        Raises
        ------
        RuntimeError
            Si no se ha llamado a fit previamente.
        """
        self._check_fitted()
        # Pérdida positiva = negativo del cuantil de retorno (mu*h + q*sigma*sqrt(h)).
        return -(self.mu_ * self.horizon + self.z_ * self.sigma_ * np.sqrt(self.horizon))

    def compute_es(self) -> float:
        """
        Calcula el ES (Expected Shortfall).

        Pérdida media en la cola, devuelta como número POSITIVO.

        - Normal : fórmula cerrada ES = -(mu*h) + sigma*sqrt(h) * phi(z[alpha]) / (1 - conf).
        - t/skewt: ES estandarizado por cuadratura del cuantil sobre la
          cola, ES_alpha = (1/alpha) integral[0]^alpha q(u) du, y luego ES = -(mu*h + ES_alpha*sigma*sqrt(h)).

        Returns
        -------
        float
            ES como pérdida positiva (pérdida esperada en la cola).

        Raises
        ------
        RuntimeError
            Si no se ha llamado a fit previamente.
        """
        self._check_fitted()
        alpha = 1.0 - self.confidence
        scale = self.sigma_ * np.sqrt(self.horizon)

        if self.dist == "normal":
            return -(self.mu_ * self.horizon - scale * norm.pdf(self.z_) / alpha)

        # ES estandarizado: media del cuantil sobre una rejilla uniforme
        # de la cola (0, alpha], que aproxima (1/alpha) integral[0]^alpha q(u) du.
        u = np.linspace(1e-6, alpha, 1000)
        es_std = float(np.asarray(self._dist_obj.ppf(u, self._dist_params)).mean())
        return -(self.mu_ * self.horizon + es_std * scale)

    # ── Resumen ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Devuelve el resumen del modelo con parámetros estimados.

        Returns
        -------
        dict
            Extiende el resumen base con: dist, mu, sigma, z, nu, lambda,
            annualized_vol.
        """
        base = super().summary()
        base.update({
            "dist":          self.dist,
            "mu":            self.mu_,
            "sigma":         self.sigma_,
            "z":             self.z_,
            "nu":            self.nu_,
            "lambda":        self.lambda_,
            "annualized_vol": annualize_volatility(self.sigma_),
        })
        return base

    # ── Representación ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "ajustado" if self._fitted else "sin ajustar"
        return (
            f"ParametricVaR("
            f"confidence={self.confidence}, "
            f"horizon={self.horizon}, "
            f"dist='{self.dist}', "
            f"status={status})"
        )
