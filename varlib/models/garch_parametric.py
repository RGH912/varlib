"""
varlib.models.garch_parametric
================================
VaR paramétrico con varianza condicional dinámica estimada mediante
un modelo GARCH(p,q).

Motivación
----------
El VaR paramétrico clásico asume volatilidad constante (sigma^2
histórica). En series financieras reales la volatilidad muestra
clustering: períodos tranquilos se alternan con períodos turbulentos.
El modelo GARCH captura este comportamiento mediante una varianza
condicional que evoluciona en el tiempo:

    sigma^2[t] = omega + sum alpha[i]*epsilon^2[t-i]  +  sum beta[j]*sigma^2[t-j]
                i=1..p               j=1..q

Estimación
----------
Se usa la librería arch (v>=6) que estima los parámetros por
Máxima Verosimilitud (MLE) con el método L-BFGS-B, garantizando
las restricciones de positividad y estacionariedad.

Escala numérica
---------------
Internamente los rendimientos se escalan *100 (forma porcentual)
para mejorar la estabilidad numérica del optimizador - práctica
estándar recomendada por la propia documentación de arch.
Todos los parámetros y resultados se devuelven en escala decimal.

Distribuciones de innovaciones soportadas
------------------------------------------
- 'normal'     : Normal estándar (defecto)
- 't'          : t-Student con nu grados de libertad estimados
- 'skewt'      : t-Student asimétrica de Hansen

VaR a horizonte h
-----------------
Para h > 1 se suman las varianzas condicionales pronosticadas:

    sigma^2(1:h) = sum[k=1]^{h}  E[sigma^2[T+k] | F[T]]

que la librería arch calcula analíticamente por recursión.
El VaR total es:

    VaR(alpha,h) = -(mu*h + z[alpha] * sqrt(sigma^2(1:h)))
"""

import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
from arch import arch_model
from arch.univariate import SkewStudent

from varlib.models.base import BaseVaR, annualize_volatility


# Factor de escala: rendimientos -> porcentaje para estabilidad numérica
_SCALE = 100.0

# Distribuciones admitidas (alias -> nombre interno de arch)
_VALID_DISTS: tuple[str, ...] = ("normal", "t", "skewt")


class GARCHParametricVaR(BaseVaR):
    """
    VaR paramétrico usando varianza condicional GARCH(p,q).

    Ajusta un modelo GARCH(p,q) a la serie de rendimientos con la
    librería arch, extrae la varianza condicional en T y calcula
    el VaR utilizando el pronóstico de volatilidad dinámica.

    Parameters
    ----------
    confidence : float, optional
        Nivel de confianza. Por defecto 0.95.
    horizon : int, optional
        Horizonte temporal en días. Por defecto 1.
    p : int, optional
        Orden ARCH (términos epsilon^2). Por defecto 1.
    q : int, optional
        Orden GARCH (términos sigma^2). Por defecto 1.
    dist : str, optional
        Distribución de las innovaciones. Opciones: 'normal'
        (defecto), 't', 'skewt'.
    mean : str, optional
        Especificación de la media condicional. 'constant' (defecto)
        o 'zero'. Para series de rendimientos log 'constant'
        es lo habitual.

    Attributes
    ----------
    garch_result_ : arch.univariate.base.ARCHModelResult
        Objeto resultado completo de arch (post-fit).
    mu_ : float
        Media condicional estimada en escala decimal (post-fit).
    omega_ : float
        Parámetro omega en escala decimal^2 (post-fit).
    alpha_ : list[float]
        Coeficientes alpha[i], i=1..p (post-fit).
    beta_ : list[float]
        Coeficientes beta[j], j=1..q (post-fit).
    nu_ : float or None
        Grados de libertad estimados si dist='t' o 'skewt' (post-fit).
        En la skew-t corresponde al parámetro 'eta' de arch. None para
        distribución normal.
    lambda_ : float or None
        Parámetro de asimetría estimado si dist='skewt' (post-fit).
        None en otro caso.
    persistence_ : float
        Persistencia de la volatilidad: sum(alpha) + sum(beta) (post-fit).
        Debe ser < 1 para estacionariedad.
    unconditional_vol_ : float or None
        Volatilidad incondicional diaria: sqrt(omega/(1-sum(alpha)-sum(beta))) (post-fit).
        None si el proceso no es estacionario (persistencia >= 1).
    conditional_variance_ : float
        Varianza condicional en T (último dato), escala decimal^2 (post-fit).
    conditional_vol_ : float
        Volatilidad condicional en T, escala decimal (post-fit).

    Examples
    --------
    from varlib.models.garch_parametric import GARCHParametricVaR
    model = GARCHParametricVaR(confidence=0.95, horizon=1, dist='t')
    model.fit(train_returns)
    model.compute_var()
    model.forecast_volatility(steps=10)
    """

    def __init__(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "constant",
    ) -> None:
        super().__init__(confidence, horizon)

        if not isinstance(p, int) or p < 1:
            raise ValueError(f"[p] debe ser un entero >= 1. Recibido: '{p}'")
        if not isinstance(q, int) or q < 1:
            raise ValueError(f"[q] debe ser un entero >= 1. Recibido: '{q}'")

        dist_lower = dist.lower()
        if dist_lower not in _VALID_DISTS:
            raise ValueError(
                f"[dist] '{dist}' no reconocido. "
                f"Opciones válidas: {list(_VALID_DISTS)}"
            )

        self.p: int = p
        self.q: int = q
        self.dist: str = dist_lower
        self.mean: str = mean.lower()

        # Atributos post-fit
        self.garch_result_         = None
        self.mu_: float | None     = None
        self.omega_: float | None  = None
        self.alpha_: list[float]   = []
        self.beta_: list[float]    = []
        self.nu_: float | None     = None
        self.lambda_: float | None = None
        self.persistence_: float | None       = None
        self.unconditional_vol_: float | None = None
        self.conditional_variance_: float | None = None
        self.conditional_vol_: float | None      = None

    # ── Ajuste ────────────────────────────────────────────────────────────────

    def fit(self, returns: pd.Series) -> "GARCHParametricVaR":
        """
        Ajusta el modelo GARCH(p,q) por Máxima Verosimilitud.

        Internamente escala los rendimientos *100 para estabilidad
        numérica, luego convierte todos los parámetros de vuelta a
        escala decimal.

        Parameters
        ----------
        returns : pd.Series
            Serie de rendimientos (logarítmicos recomendados).
            Mínimo recomendado: 250 observaciones.

        Returns
        -------
        GARCHParametricVaR
            La propia instancia (permite encadenamiento).

        Raises
        ------
        RuntimeError
            Si el optimizador de arch no converge.
        """
        self._validate_returns(returns, min_obs=50)
        clean = returns.dropna()
        scaled = clean * _SCALE   # porcentaje para mejor condicionamiento

        am = arch_model(
            scaled,
            vol="Garch",
            p=self.p,
            q=self.q,
            dist=self.dist,
            mean="Constant" if self.mean == "constant" else "Zero",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off")

        if res.convergence_flag != 0:
            warnings.warn(
                f"GARCHParametricVaR: el optimizador no convergió "
                f"(flag={res.convergence_flag}). "
                "Los resultados pueden ser poco fiables.",
                RuntimeWarning,
                stacklevel=2,
            )

        # ── Extraer y convertir parámetros a escala decimal ───────────────────
        params = res.params

        self.mu_    = float(params.get("mu", 0.0)) / _SCALE
        self.omega_ = float(params["omega"]) / (_SCALE ** 2)

        self.alpha_ = [float(params[f"alpha[{i}]"]) for i in range(1, self.p + 1)]
        self.beta_ = [float(params[f"beta[{i}]"])  for i in range(1, self.q + 1)]

        # Parámetros de forma para distribuciones de cola pesada.
        # arch nombra 'nu' a los gl de la t, y 'eta'/'lambda' a los de la skew-t.
        if self.dist == "t":
            self.nu_ = float(params["nu"])
        elif self.dist == "skewt":
            self.nu_     = float(params["eta"])
            self.lambda_ = float(params["lambda"])
        # normal: nu_ y lambda_ siguen None desde el __init__

        # Estadísticos derivados
        sum_alpha = sum(self.alpha_)
        sum_beta  = sum(self.beta_)
        self.persistence_ = sum_alpha + sum_beta

        if self.persistence_ < 1.0:
            self.unconditional_vol_ = np.sqrt(self.omega_ / (1.0 - self.persistence_))
        else:
            self.unconditional_vol_ = None   # proceso no estacionario: no existe

        # Varianza condicional en T (último valor de la serie filtrada)
        # res.conditional_volatility está en escala porcentual (%)
        last_vol_pct = float(res.conditional_volatility.iloc[-1])
        self.conditional_variance_ = (last_vol_pct / _SCALE) ** 2
        self.conditional_vol_ = last_vol_pct / _SCALE

        self.garch_result_ = res
        self._returns  = clean
        self._fitted   = True
        return self

    # ── VaR y ES ────────────────────────────────────────────────────────────

    def compute_var(self) -> float:
        """
        Calcula el VaR paramétrico GARCH.

        Usa el pronóstico de varianza condicional a horizon pasos:

            VaR(alpha,h) = -(mu*h + z[alpha] * sqrt[sum[k=1]^{h} E(sigma^2[T+k]|F[T])])

        Para h=1 se usa directamente sigma^2[T+1] (varianza 1-step ahead).
        Para h>1 se suman las varianzas pronosticadas individuales.

        Returns
        -------
        float
            VaR como pérdida positiva (escala decimal).
        """
        self._check_fitted()

        sigma_h = self._forecast_sigma(self.horizon)
        z = self._compute_quantile()
        # Pérdida positiva = negativo del cuantil de retorno (mu*h + z*sigma_h).
        return -(self.mu_ * self.horizon + z * sigma_h)

    def compute_es(self) -> float:
        """
        Calcula el ES (Expected Shortfall, también CVaR, pérdida positiva) con fórmula cerrada según la
        distribución.

        - Normal : ES = -(mu*h - sigma[h] * phi(z[alpha])/(1-conf))
        - t/skewt: ES estandarizado por cuadratura del cuantil sobre la
          cola, ES_alpha = (1/alpha) integral[0]^alpha q(u) du.

        Returns
        -------
        float
            ES como pérdida positiva.
        """
        self._check_fitted()

        sigma_h = self._forecast_sigma(self.horizon)
        z       = self._compute_quantile()
        alpha   = 1.0 - self.confidence

        if self.dist == "normal":
            # Fórmula cerrada Normal (negada -> pérdida positiva)
            return -(self.mu_ * self.horizon - sigma_h * stats.norm.pdf(z) / alpha)

        # ES estandarizado: media del cuantil estandarizado (varianza 1) sobre
        # una rejilla uniforme de la cola (0, alpha], que aproxima
        # (1/alpha) integral[0]^alpha q(u) du. Mismo enfoque que ParametricVaR.
        u = np.linspace(1e-6, alpha, 1000)
        if self.dist == "t":
            nu    = self.nu_ if self.nu_ is not None else 10.0
            q_std = stats.t.ppf(u, df=nu) * np.sqrt((nu - 2.0) / nu)
        else:  # skewt
            q_std = np.asarray(SkewStudent().ppf(u, [self.nu_, self.lambda_]))
        es_std = float(q_std.mean())
        return -(self.mu_ * self.horizon + es_std * sigma_h)

    # ── Volatilidad condicional y pronóstico ───────────────────────────────────

    @property
    def conditional_vol_series_(self) -> pd.Series:
        """
        Serie de volatilidad condicional diaria in-sample, en escala decimal.

        Encapsula la conversión de arch (que ajusta sobre retornos*100, por lo
        que su conditional_volatility viene en %): aquí se devuelve ya en
        decimal y con el índice de los retornos de entrenamiento. Para
        anualizar, usa annualize_volatility (de varlib.models.base).

        Returns
        -------
        pd.Series
            Volatilidad condicional diaria (decimal), una por fecha de train.
        """
        self._check_fitted()
        return self.garch_result_.conditional_volatility / _SCALE

    def forecast_volatility(self, steps: int = 10) -> np.ndarray:
        """
        Pronóstico de la volatilidad condicional diaria.

        Calcula E[sigma^2[T+k]|F[T]] para k=1..steps mediante la recursión
        GARCH analítica (implementada en arch). En escala decimal y diaria,
        coherente con conditional_vol_series_ (usa annualize_volatility si lo
        necesitas para mostrarla).

        Parameters
        ----------
        steps : int
            Número de días a pronosticar. Por defecto 10.

        Returns
        -------
        np.ndarray, shape (steps,)
            Volatilidad diaria (decimal) para cada paso.

        Examples
        --------
        vols = model.forecast_volatility(steps=30)
        print(f"Vol. mañana: {vols[0]:.2%}, en 30 días: {vols[-1]:.2%}")
        """
        self._check_fitted()

        fc = self.garch_result_.forecast(horizon=steps, reindex=False)
        # fc.variance: DataFrame (1 * steps), valores en %^2
        var_pct2 = fc.variance.iloc[-1].values        # shape (steps,)
        var_dec  = var_pct2 / (_SCALE ** 2)           # decimal^2
        return np.sqrt(var_dec)                       # vol diaria (decimal)

    # ── Utilidades internas ───────────────────────────────────────────────────

    def _forecast_sigma(self, h: int) -> float:
        """
        Desviación típica pronosticada a horizonte h en escala decimal.

        Para h=1 devuelve sigma[T+1].
        Para h>1 devuelve sqrt(sum[k=1]^{h} sigma^2[T+k]).
        """
        fc = self.garch_result_.forecast(horizon=h, reindex=False)
        var_pct2 = fc.variance.iloc[-1].values   # shape (h,)
        total_var_dec = np.sum(var_pct2) / (_SCALE ** 2)
        return float(np.sqrt(total_var_dec))

    def _compute_quantile(self) -> float:
        """
        Cuantil z[alpha] de la distribución de innovaciones configurada.
        """
        alpha = 1.0 - self.confidence

        if self.dist == "normal":
            return float(stats.norm.ppf(alpha))

        elif self.dist == "t":
            nu = self.nu_ if self.nu_ is not None else 10.0
            # t estandarizada en arch: Var=1 -> escalar para unit-variance
            raw_q = float(stats.t.ppf(alpha, df=nu))
            return raw_q * float(np.sqrt((nu - 2.0) / nu))

        elif self.dist == "skewt":
            # Cuantil estandarizado (varianza 1) de la skew-t de arch,
            # usando los parámetros ajustados eta (nu_) y lambda (lambda_).
            dist = SkewStudent()
            return float(np.asarray(dist.ppf(np.array([alpha]), [self.nu_, self.lambda_]))[0])

        return float(stats.norm.ppf(alpha))   # fallback

    # ── Resumen ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Resumen completo del modelo GARCH con todos sus parámetros.

        Returns
        -------
        dict
            Extiende el resumen base con todos los parámetros GARCH,
            estadísticos de ajuste (AIC, BIC) y métricas de volatilidad.
        """
        base = super().summary()
        base.update({
            "p":                  self.p,
            "q":                  self.q,
            "dist":               self.dist,
            "mu":                 self.mu_,
            "omega":              self.omega_,
            "alpha":              self.alpha_,
            "beta":               self.beta_,
            "nu":                 self.nu_,
            "lambda":             self.lambda_,
            "persistence":        self.persistence_,
            "unconditional_vol":  self.unconditional_vol_,
            "conditional_vol_T":  self.conditional_vol_,
            "annualized_cond_vol": (
                annualize_volatility(self.conditional_vol_)
                if self.conditional_vol_ is not None else None
            ),
            "annualized_uncond_vol": (
                annualize_volatility(self.unconditional_vol_)
                if self.unconditional_vol_ is not None else None
            ),
            "aic":    self.garch_result_.aic if self.garch_result_ else None,
            "bic":    self.garch_result_.bic if self.garch_result_ else None,
        })
        return base

    # ── Representación ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "ajustado" if self._fitted else "sin ajustar"
        return (
            f"GARCHParametricVaR("
            f"confidence={self.confidence}, "
            f"horizon={self.horizon}, "
            f"p={self.p}, q={self.q}, "
            f"dist='{self.dist}', "
            f"status={status})"
        )
