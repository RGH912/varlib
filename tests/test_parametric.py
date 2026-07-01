"""
Tests para ParametricVaR (Normal con volatilidad constante).

Incluye:
- Resultado exacto frente a la fórmula mu + z[alpha]*sigma*sqrt(h) y al ES cerrado.
- Invariantes de signo y monotonía respecto a la confianza.
- Escalado sqrt(h) del horizonte.
- Errores y casos límite.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from varlib.models.parametric import ParametricVaR


class TestParametricVaR:

    # ── Ciclo de vida y atributos ─────────────────────────────────────────────

    def test_fit_sets_state(self, train_returns):
        """
        fit() devuelve self, marca el flag y deja los atributos listos.
        """
        model = ParametricVaR()
        assert model.fit(train_returns) is model
        assert model._fitted
        assert model.mu_ is not None
        assert model.sigma_ is not None and model.sigma_ > 0
        assert model.z_ < 0
        assert model.n_obs_ == len(train_returns.dropna())

    # ── Resultado exacto frente a la fórmula ──────────────────────────────────

    def test_var_matches_closed_form(self, train_returns):
        """
        VaR == -(mu*h + z[alpha]*sigma*sqrt(h)), vs scipy y vs params internos.
        """
        conf, h = 0.95, 1
        model = ParametricVaR(confidence=conf, horizon=h).fit(train_returns)
        clean = train_returns.dropna()
        z = norm.ppf(1 - conf)
        expected = -(clean.mean() + z * clean.std(ddof=1))
        np.testing.assert_allclose(model.compute_var(), expected, rtol=1e-12)
        # Coherente con los parámetros internos almacenados.
        expected_internal = -(model.mu_ * h + model.z_ * model.sigma_ * np.sqrt(h))
        np.testing.assert_allclose(model.compute_var(), expected_internal, rtol=1e-12)

    def test_es_matches_closed_form(self, train_returns):
        """
        ES == -(mu*h - sigma*sqrt(h)*phi(z[alpha])/(1-conf)).
        """
        conf = 0.95
        model = ParametricVaR(confidence=conf, horizon=1).fit(train_returns)
        clean = train_returns.dropna()
        z = norm.ppf(1 - conf)
        expected = -(clean.mean() - clean.std(ddof=1) * norm.pdf(z) / (1 - conf))
        np.testing.assert_allclose(model.compute_es(), expected, rtol=1e-12)

    # ── Invariantes ───────────────────────────────────────────────────────────

    def test_compute_var_is_positive(self, train_returns):
        assert ParametricVaR(confidence=0.95).fit(train_returns).compute_var() > 0

    def test_es_geq_var(self, train_returns):
        model = ParametricVaR(confidence=0.95).fit(train_returns)
        assert model.compute_es() >= model.compute_var()

    def test_var_more_extreme_with_confidence(self, train_returns):
        v90 = ParametricVaR(confidence=0.90).fit(train_returns).compute_var()
        v95 = ParametricVaR(confidence=0.95).fit(train_returns).compute_var()
        v99 = ParametricVaR(confidence=0.99).fit(train_returns).compute_var()
        assert v90 < v95 < v99   # mayor confianza = mayor pérdida (VaR positivo)

    def test_risk_term_scales_with_sqrt_horizon(self, train_returns):
        m1  = ParametricVaR(confidence=0.95, horizon=1).fit(train_returns)
        m10 = ParametricVaR(confidence=0.95, horizon=10).fit(train_returns)
        # VaR = -(mu*h + z*sigma*sqrt(h)); la parte de riesgo (sin drift) escala con sqrt(h).
        risk_1  = m1.compute_var()  + m1.mu_ * 1
        risk_10 = m10.compute_var() + m10.mu_ * 10
        np.testing.assert_allclose(risk_10 / risk_1, np.sqrt(10), rtol=1e-9)

    # ── Errores y casos límite ────────────────────────────────────────────────

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ParametricVaR(confidence=1.2)

    def test_horizon_invalid_raises(self):
        with pytest.raises(ValueError):
            ParametricVaR(horizon=0)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            ParametricVaR().compute_var()

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError):
            ParametricVaR().fit(pd.Series([0.01, 0.02]))

    def test_wrong_input_type_raises(self):
        with pytest.raises(TypeError):
            ParametricVaR().fit([0.01] * 50)

    # ── Resumen ───────────────────────────────────────────────────────────────

    def test_summary_keys(self, train_returns):
        s = ParametricVaR().fit(train_returns).summary()
        for key in ("model", "confidence", "horizon", "var", "es",
                    "mu", "sigma", "z", "annualized_vol", "n_obs"):
            assert key in s

    def test_summary_var_matches_compute(self, train_returns):
        model = ParametricVaR(confidence=0.95).fit(train_returns)
        np.testing.assert_allclose(model.summary()["var"], model.compute_var(), rtol=1e-12)

    def test_default_dist_is_normal(self):
        assert ParametricVaR().dist == "normal"

    def test_invalid_dist_raises(self):
        with pytest.raises(ValueError):
            ParametricVaR(dist="xyz")


class TestParametricDistributions:
    """
    Distribuciones de innovaciones: Normal, t de Student y skew-t.
    """

    @pytest.mark.parametrize("dist", ["t", "skewt"])
    def test_fit_estimates_nu(self, train_returns, dist):
        model = ParametricVaR(0.95, dist=dist).fit(train_returns)
        assert model.nu_ is not None and model.nu_ > 2

    def test_skewt_estimates_lambda(self, train_returns):
        model = ParametricVaR(0.95, dist="skewt").fit(train_returns)
        assert model.lambda_ is not None
        assert -1.0 < model.lambda_ < 1.0

    def test_normal_has_no_shape_params(self, train_returns):
        model = ParametricVaR(0.95, dist="normal").fit(train_returns)
        assert model.nu_ is None and model.lambda_ is None

    @pytest.mark.parametrize("dist", ["normal", "t", "skewt"])
    def test_var_positive_and_es_geq_var(self, train_returns, dist):
        model = ParametricVaR(0.99, dist=dist).fit(train_returns)
        assert model.compute_var() > 0
        assert model.compute_es() >= model.compute_var()

    def test_t_heavier_tail_than_normal_at_99(self, train_returns):
        """
        A 99% la t (colas pesadas) debe dar un VaR al menos tan extremo
        (mayor pérdida positiva) como la Normal sobre los mismos datos."""
        v_normal = ParametricVaR(0.99, dist="normal").fit(train_returns).compute_var()
        v_t      = ParametricVaR(0.99, dist="t").fit(train_returns).compute_var()
        assert v_t >= v_normal - 1e-9

    def test_summary_dist_fields(self, train_returns):
        s = ParametricVaR(0.95, dist="skewt").fit(train_returns).summary()
        for key in ("dist", "nu", "lambda"):
            assert key in s
        assert s["dist"] == "skewt"
