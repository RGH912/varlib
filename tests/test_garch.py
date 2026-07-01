"""
Tests para GARCHParametricVaR y GARCHMonteCarloVaR.
"""

import pytest

from varlib.models.garch_parametric import GARCHParametricVaR
from varlib.models.garch_montecarlo import GARCHMonteCarloVaR


class TestGARCHParametricVaR:

    def test_fit_sets_state(self, volatile_returns):
        """
        fit() devuelve self y deja los parámetros GARCH estimados.
        """
        model = GARCHParametricVaR()
        assert model.fit(volatile_returns) is model
        assert model.omega_ is not None and model.omega_ > 0
        assert len(model.alpha_) == 1 and len(model.beta_) == 1
        assert 0 < model.alpha_[0] < 1
        assert 0 < model.beta_[0] < 1

    def test_stationarity(self, volatile_returns):
        assert GARCHParametricVaR().fit(volatile_returns).persistence_ < 1.0

    def test_var_positive(self, volatile_returns):
        assert GARCHParametricVaR().fit(volatile_returns).compute_var() > 0

    def test_es_geq_var(self, volatile_returns):
        model = GARCHParametricVaR().fit(volatile_returns)
        assert model.compute_es() >= model.compute_var()

    def test_var_more_extreme_with_confidence(self, volatile_returns):
        model95 = GARCHParametricVaR(0.95).fit(volatile_returns)
        model99 = GARCHParametricVaR(0.99).fit(volatile_returns)
        assert model95.compute_var() < model99.compute_var()

    def test_forecast_volatility_shape(self, volatile_returns):
        fc = GARCHParametricVaR().fit(volatile_returns).forecast_volatility(steps=10)
        assert fc.shape == (10,) and (fc > 0).all()

    def test_t_dist_has_nu(self, volatile_returns):
        model = GARCHParametricVaR(dist="t").fit(volatile_returns)
        assert model.nu_ is not None and model.nu_ > 2

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            GARCHParametricVaR(p=0)

    def test_invalid_dist_raises(self):
        with pytest.raises(ValueError):
            GARCHParametricVaR(dist="invalid")

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            GARCHParametricVaR().compute_var()

    def test_summary_keys(self, volatile_returns):
        s = GARCHParametricVaR().fit(volatile_returns).summary()
        for key in ("omega", "alpha", "beta", "persistence",
                    "conditional_vol_T", "aic", "bic"):
            assert key in s

    @pytest.mark.parametrize("dist", ["t", "skewt"])
    def test_es_heavy_tails(self, volatile_returns, dist):
        """
        ES con distribuciones de cola: rama de simulación/cuantil.
        """
        model = GARCHParametricVaR(confidence=0.99, dist=dist).fit(volatile_returns)
        es = model.compute_es()
        assert es >= model.compute_var() > 0

    def test_multiday_horizon_var(self, volatile_returns):
        """
        Horizonte > 1 suma varianzas pronosticadas (rama _forecast_sigma).
        """
        model = GARCHParametricVaR(confidence=0.95, horizon=10).fit(volatile_returns)
        assert model.compute_var() > 0


class TestGARCHMonteCarloVaR:

    def test_fit_returns_self(self, volatile_returns):
        model = GARCHMonteCarloVaR(n_simulations=500, random_state=0)
        assert model.fit(volatile_returns) is model

    def test_correct_param_mapping(self, volatile_returns):
        model = GARCHMonteCarloVaR(0.95, 1, 2000, 42).fit(volatile_returns)
        assert model.p == 1 and model.q == 1 and model.n_simulations == 2000

    def test_sigma2_next_positive(self, volatile_returns):
        model = GARCHMonteCarloVaR(n_simulations=500, random_state=0).fit(volatile_returns)
        assert model.sigma2_next_ > 0

    def test_simulated_returns_shape(self, volatile_returns):
        model = GARCHMonteCarloVaR(n_simulations=3000, random_state=0).fit(volatile_returns)
        model.simulate_paths()
        assert model.simulated_returns_.shape == (3000,)

    def test_var_positive(self, volatile_returns):
        model = GARCHMonteCarloVaR(n_simulations=2000, random_state=0).fit(volatile_returns)
        assert model.compute_var() > 0

    def test_es_geq_var(self, volatile_returns):
        model = GARCHMonteCarloVaR(n_simulations=2000, random_state=0).fit(volatile_returns)
        assert model.compute_es() >= model.compute_var()

    def test_reproducibility(self, volatile_returns):
        v1 = GARCHMonteCarloVaR(n_simulations=2000, random_state=7).fit(volatile_returns).compute_var()
        v2 = GARCHMonteCarloVaR(n_simulations=2000, random_state=7).fit(volatile_returns).compute_var()
        assert v1 == v2

    def test_mc_close_to_parametric(self, volatile_returns):
        gp  = GARCHParametricVaR(0.95).fit(volatile_returns).compute_var()
        gmc = GARCHMonteCarloVaR(0.95, n_simulations=50_000, random_state=0).fit(volatile_returns).compute_var()
        assert abs(gmc - gp) / abs(gp) < 0.10

    def test_refitting_clears_simulations(self, volatile_returns):
        model = GARCHMonteCarloVaR(n_simulations=500, random_state=0).fit(volatile_returns)
        model.compute_var()
        model.fit(volatile_returns)
        assert model.simulated_returns_ is None

    def test_not_fitted_simulate_raises(self):
        with pytest.raises(RuntimeError):
            GARCHMonteCarloVaR().simulate_paths()

    def test_invalid_n_simulations_raises(self):
        with pytest.raises(ValueError):
            GARCHMonteCarloVaR(n_simulations=0)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            GARCHMonteCarloVaR(p=0)

    def test_invalid_q_raises(self):
        with pytest.raises(ValueError):
            GARCHMonteCarloVaR(q=0)

    def test_multiday_horizon(self, volatile_returns):
        """
        Horizonte > 1 ejercita la recursión de varianza paso a paso.
        """
        model = GARCHMonteCarloVaR(horizon=5, n_simulations=2000, random_state=0).fit(volatile_returns)
        assert model.compute_var() > 0

    def test_summary_with_simulation_stats(self, volatile_returns):
        model = GARCHMonteCarloVaR(n_simulations=2000, random_state=0).fit(volatile_returns)
        model.compute_var()        # fuerza la simulación
        s = model.summary()
        for key in ("sim_mean", "sim_std", "sim_skew", "sim_kurt", "persistence"):
            assert key in s
