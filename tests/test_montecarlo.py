"""
Tests para MonteCarloVaR (GBM con volatilidad constante).

Incluye:
- Reproducibilidad con random_state fijo y divergencia entre semillas.
- Coherencia con el VaR paramétrico e histórico (misma Normal subyacente).
- Invariantes, errores y manejo de la caché de simulaciones.
"""

import numpy as np
import pytest

from varlib.models.montecarlo import MonteCarloVaR
from varlib.models.parametric import ParametricVaR
from varlib.models.historical import HistoricalVaR


class TestMonteCarloVaR:

    # ── Ciclo de vida / simulación ────────────────────────────────────────────

    def test_fit_returns_self(self, train_returns):
        model = MonteCarloVaR(n_simulations=1000, random_state=0)
        assert model.fit(train_returns) is model

    def test_simulated_returns_shape(self, train_returns):
        model = MonteCarloVaR(n_simulations=5000, random_state=0).fit(train_returns)
        model.simulate()
        assert model.simulated_returns_.shape == (5000,)

    def test_compute_var_auto_simulates(self, train_returns):
        """
        compute_var sin llamar a simulate() debe simular sola.
        """
        model = MonteCarloVaR(n_simulations=2000, random_state=0).fit(train_returns)
        assert model.simulated_returns_ is None
        var = model.compute_var()
        assert var > 0 and model.simulated_returns_ is not None

    # ── Reproducibilidad ──────────────────────────────────────────────────────

    def test_reproducibility_same_seed(self, train_returns):
        v1 = MonteCarloVaR(n_simulations=5000, random_state=42).fit(train_returns).compute_var()
        v2 = MonteCarloVaR(n_simulations=5000, random_state=42).fit(train_returns).compute_var()
        assert v1 == v2

    def test_different_seeds_differ(self, train_returns):
        v1 = MonteCarloVaR(n_simulations=5000, random_state=1).fit(train_returns).compute_var()
        v2 = MonteCarloVaR(n_simulations=5000, random_state=2).fit(train_returns).compute_var()
        assert v1 != v2

    # ── Invariantes ───────────────────────────────────────────────────────────

    def test_compute_var_positive(self, train_returns):
        assert MonteCarloVaR(n_simulations=2000, random_state=0).fit(train_returns).compute_var() > 0

    def test_es_geq_var(self, train_returns):
        model = MonteCarloVaR(n_simulations=5000, random_state=0).fit(train_returns)
        assert model.compute_es() >= model.compute_var()

    # ── Coherencia entre métodos ──────────────────────────────────────────────

    def test_var_close_to_parametric(self, train_returns):
        par = ParametricVaR(0.95).fit(train_returns).compute_var()
        mc  = MonteCarloVaR(n_simulations=100_000, random_state=0).fit(train_returns).compute_var()
        assert abs(mc - par) / abs(par) < 0.05

    def test_three_methods_agree_on_normal(self, large_normal_returns):
        """
        Con muestra normal grande, los tres VaR(95%) deben coincidir.
        """
        par  = ParametricVaR(0.95).fit(large_normal_returns).compute_var()
        hist = HistoricalVaR(0.95).fit(large_normal_returns).compute_var()
        mc   = MonteCarloVaR(0.95, n_simulations=100_000, random_state=0).fit(large_normal_returns).compute_var()
        np.testing.assert_allclose(hist, par, rtol=0.05)
        np.testing.assert_allclose(mc,   par, rtol=0.05)

    # ── Errores y caché ───────────────────────────────────────────────────────

    def test_invalid_n_simulations_raises(self):
        with pytest.raises(ValueError):
            MonteCarloVaR(n_simulations=0)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            MonteCarloVaR().compute_var()

    def test_not_fitted_simulate_raises(self):
        with pytest.raises(RuntimeError):
            MonteCarloVaR().simulate()

    def test_refitting_clears_cache(self, train_returns):
        model = MonteCarloVaR(n_simulations=1000, random_state=0).fit(train_returns)
        model.compute_var()
        model.fit(train_returns)
        assert model.simulated_returns_ is None

    def test_summary_keys(self, train_returns):
        s = MonteCarloVaR(n_simulations=1000, random_state=0).fit(train_returns).summary()
        for key in ("mu", "sigma", "n_simulations", "random_state", "simulated", "n_obs"):
            assert key in s
