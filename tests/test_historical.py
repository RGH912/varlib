"""
Tests para HistoricalVaR (percentil empírico clásico).

Incluye:
- Resultado exacto frente a np.percentile sobre datos controlados.
- Escalado sqrt(h), invariantes de signo/confianza y errores.
"""

import numpy as np
import pandas as pd
import pytest

from varlib.models.historical import HistoricalVaR


class TestHistoricalVaR:

    # ── Resultado exacto conocido ─────────────────────────────────────────────

    def test_var_equals_empirical_percentile(self, controlled_returns):
        """
        VaR(95%) == -percentil 5 empírico (pérdida positiva).
        """
        model = HistoricalVaR(confidence=0.95, horizon=1).fit(controlled_returns)
        expected = -np.percentile(controlled_returns.values, 5)
        np.testing.assert_allclose(model.compute_var(), expected, rtol=1e-12)

    def test_var_scales_with_sqrt_horizon_exact(self, controlled_returns):
        m1 = HistoricalVaR(0.95, horizon=1).fit(controlled_returns)
        m9 = HistoricalVaR(0.95, horizon=9).fit(controlled_returns)
        np.testing.assert_allclose(m9.compute_var(), m1.compute_var() * 3.0, rtol=1e-12)

    def test_es_is_tail_mean(self, controlled_returns):
        model = HistoricalVaR(0.95, horizon=1).fit(controlled_returns)
        q = np.percentile(controlled_returns.values, 5)   # cuantil de retorno
        tail = controlled_returns.values[controlled_returns.values <= q]
        np.testing.assert_allclose(model.compute_es(), -tail.mean(), rtol=1e-12)

    # ── Ciclo de vida e invariantes ───────────────────────────────────────────

    def test_fit_returns_self(self, train_returns):
        model = HistoricalVaR()
        assert model.fit(train_returns) is model

    def test_compute_var_positive(self, train_returns):
        assert HistoricalVaR().fit(train_returns).compute_var() > 0

    def test_es_geq_var(self, train_returns):
        model = HistoricalVaR().fit(train_returns)
        assert model.compute_es() >= model.compute_var()

    def test_var_more_extreme_with_confidence(self, train_returns):
        v90 = HistoricalVaR(0.90).fit(train_returns).compute_var()
        v99 = HistoricalVaR(0.99).fit(train_returns).compute_var()
        assert v90 < v99   # mayor confianza -> mayor pérdida (VaR positivo)

    # ── Errores y casos límite ────────────────────────────────────────────────

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            HistoricalVaR().compute_var()

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError):
            HistoricalVaR().fit(pd.Series([0.01, 0.02, 0.03]))

    def test_wrong_input_type_raises(self):
        with pytest.raises(TypeError):
            HistoricalVaR().fit(np.arange(50) * 0.001)

    # ── Resumen ───────────────────────────────────────────────────────────────

    def test_summary_keys(self, train_returns):
        s = HistoricalVaR().fit(train_returns).summary()
        for key in ("model", "confidence", "var", "es", "n_obs"):
            assert key in s
