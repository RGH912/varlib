"""
Tests para Backtester (Kupiec + Christoffersen) y su VaR dinámico.

Incluye:
- Claves devueltas por kupiec_test / christoffersen_test / summary.
- Coherencia de violations / var_series / tasas.
- Ventana expansiva vs. deslizante (ambas producen resultados válidos).
- Casos límite: ventana mayor que los datos, no ejecutado.
"""

import numpy as np
import pandas as pd
import pytest

from varlib.models.parametric  import ParametricVaR
from varlib.models.historical  import HistoricalVaR
from varlib.validation.backtesting import Backtester


class TestBacktesterRun:

    def test_run_returns_self(self, train_returns, test_returns):
        bt = Backtester(ParametricVaR(0.95))
        assert bt.run_expanding(train_returns, test_returns) is bt

    def test_violations_shape_and_dtype(self, train_returns, test_returns):
        bt = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns)
        assert bt.violations_.shape == (len(test_returns),)
        assert bt.violations_.dtype == bool

    def test_var_series_shape(self, train_returns, test_returns):
        bt = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns)
        assert len(bt.var_series_) == len(test_returns)

    def test_var_series_all_positive(self, train_returns, test_returns):
        bt = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns)
        assert (bt.var_series_.values > 0).all()

    def test_n_violations_consistent(self, train_returns, test_returns):
        bt = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns)
        assert bt.n_violations_ == bt.violations_.sum()

    def test_violation_rate_in_range(self, train_returns, test_returns):
        bt = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns)
        assert 0.0 <= bt.violation_rate_ <= 1.0


class TestWindowModes:

    def test_expanding_window_runs(self, train_returns, test_returns):
        bt = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns)
        assert bt.var_series_.notna().all()
        assert (bt.var_series_.values > 0).all()

    def test_rolling_window_runs(self, train_returns, test_returns):
        returns = pd.concat([train_returns, test_returns])
        bt = Backtester(ParametricVaR(0.95)).run_rolling(
            returns, 100, eval_start=test_returns.index[0]
        )
        assert bt.var_series_.notna().all()
        assert (bt.var_series_.values > 0).all()

    def test_rolling_and_expanding_differ(self, train_returns, test_returns):
        returns = pd.concat([train_returns, test_returns])
        exp  = Backtester(ParametricVaR(0.95)).run_expanding(
            train_returns, test_returns).var_series_
        roll = Backtester(ParametricVaR(0.95)).run_rolling(
            returns, 60, eval_start=test_returns.index[0]).var_series_
        # Ventanas distintas -> al menos algún VaR distinto
        assert not np.allclose(exp.values, roll.values)

    def test_window_larger_than_data_still_runs(self, train_returns, test_returns):
        """
        Ventana mayor que el historial disponible no debe romper.
        """
        returns = pd.concat([train_returns, test_returns])
        big = len(returns) + 1000
        bt = Backtester(ParametricVaR(0.95)).run_rolling(
            returns, big, eval_start=test_returns.index[0]
        )
        assert len(bt.var_series_) == len(test_returns)
        assert bt.var_series_.notna().all()


class TestStatisticalTests:

    def test_kupiec_keys(self, train_returns, test_returns):
        result = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns).kupiec_test()
        for key in ("n_obs", "n_violations", "violation_rate", "expected_rate",
                    "LR_uc", "p_value", "reject_H0"):
            assert key in result

    def test_kupiec_pvalue_range(self, train_returns, test_returns):
        p = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns).kupiec_test()["p_value"]
        assert 0.0 <= p <= 1.0

    def test_christoffersen_keys(self, train_returns, test_returns):
        result = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns).christoffersen_test()
        for key in ("n00", "n01", "n10", "n11", "LR_ind", "p_value_ind", "LR_cc", "p_value_cc"):
            assert key in result

    def test_christoffersen_transition_counts_sum(self, train_returns, test_returns):
        c = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns).christoffersen_test()
        assert c["n00"] + c["n01"] + c["n10"] + c["n11"] == len(test_returns) - 1

    def test_summary_is_dict(self, train_returns, test_returns):
        s = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns).summary()
        assert isinstance(s, dict)
        assert {"modelo", "n_violations", "LR_uc", "reject_cc"} <= s.keys()

    def test_expected_rate_matches_confidence(self, train_returns, test_returns):
        kup = Backtester(HistoricalVaR(0.95)).run_expanding(train_returns, test_returns).kupiec_test()
        assert kup["expected_rate"] == pytest.approx(0.05, abs=1e-6)


class TestBacktesterErrors:

    def test_invalid_model_raises(self):
        with pytest.raises(TypeError):
            Backtester("not_a_model")

    def test_not_run_raises(self):
        with pytest.raises(RuntimeError):
            Backtester(ParametricVaR(0.95)).kupiec_test()

    def test_too_few_train_raises(self, test_returns):
        with pytest.raises(ValueError):
            Backtester(ParametricVaR(0.95)).run_expanding(pd.Series([0.01] * 10), test_returns)

    def test_empty_test_raises(self, train_returns):
        with pytest.raises(ValueError):
            Backtester(ParametricVaR(0.95)).run_expanding(train_returns, pd.Series([], dtype=float))


class TestValidationCoverage:

    def test_violation_rate_near_alpha_on_normal(self, synthetic_returns):
        n_train = int(len(synthetic_returns) * 0.8)
        train = synthetic_returns.iloc[:n_train]
        test  = synthetic_returns.iloc[n_train:]
        bt = Backtester(ParametricVaR(0.95)).run_expanding(train, test)
        assert abs(bt.violation_rate_ - 0.05) < 0.10
