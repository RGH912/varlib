"""
Tests para compare_models_expanding / compare_models_rolling
(comparación sistemática vía backtesting).

La unidad de comparación es un Backtester ya configurado: el diccionario es
{etiqueta: Backtester}.
"""

import numpy as np
import pandas as pd
import pytest

from varlib.models.parametric import ParametricVaR
from varlib.models.historical import HistoricalVaR
from varlib.models.garch_parametric import GARCHParametricVaR
from varlib.validation.backtesting import Backtester
from varlib.validation.comparison import (
    compare_models_expanding,
    compare_models_rolling,
)


# Columnas que debe heredar de Backtester.summary()
_EXPECTED_COLS = [
    "modelo", "confianza", "n_obs", "n_violations", "violation_rate",
    "expected_rate", "LR_uc", "p_uc", "reject_uc",
    "LR_ind", "p_ind", "reject_ind", "LR_cc", "p_cc", "reject_cc",
]


class TestCompareModelsRolling:

    def test_returns_dataframe_one_row_per_model(self, synthetic_returns, test_returns):
        bts = {
            "Parametrico": Backtester(ParametricVaR(0.95)),
            "Historico":   Backtester(HistoricalVaR(0.95)),
        }
        df = compare_models_rolling(bts, synthetic_returns, 100,
                                    eval_start=test_returns.index[0])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_label_used_as_identifier(self, synthetic_returns, test_returns):
        bts = {
            "Modelo A": Backtester(ParametricVaR(0.95)),
            "Modelo B": Backtester(HistoricalVaR(0.95)),
        }
        df = compare_models_rolling(bts, synthetic_returns, 100,
                                    eval_start=test_returns.index[0])
        assert list(df["modelo"]) == ["Modelo A", "Modelo B"]

    def test_label_distinguishes_same_class(self, synthetic_returns, test_returns):
        # Dos modelos de la MISMA clase con etiquetas distintas: ambas filas
        # deben conservar su etiqueta (no el nombre de la clase).
        bts = {
            "Param 95": Backtester(ParametricVaR(0.95)),
            "Param 99": Backtester(ParametricVaR(0.99)),
        }
        df = compare_models_rolling(bts, synthetic_returns, 100,
                                    eval_start=test_returns.index[0])
        assert list(df["modelo"]) == ["Param 95", "Param 99"]
        assert "ParametricVaR" not in df["modelo"].values

    def test_expected_columns_present(self, synthetic_returns, test_returns):
        df = compare_models_rolling({"P": Backtester(ParametricVaR(0.95))},
                                    synthetic_returns, 100,
                                    eval_start=test_returns.index[0])
        for col in _EXPECTED_COLS:
            assert col in df.columns

    def test_per_model_confidence_reported(self, synthetic_returns, test_returns):
        # Cada modelo lleva su confianza; la tabla la refleja por fila.
        df = compare_models_rolling(
            {"P90": Backtester(ParametricVaR(0.90)),
             "P99": Backtester(HistoricalVaR(0.99))},
            synthetic_returns, 100, eval_start=test_returns.index[0],
        )
        assert list(df["confianza"]) == [0.90, 0.99]
        np.testing.assert_allclose(df["expected_rate"].values, [0.10, 0.01], atol=1e-6)

    def test_per_model_runs(self, synthetic_returns, test_returns):
        # Dos Backtesters distintos: ambos producen su fila sin error.
        bts = {
            "normal": Backtester(ParametricVaR(0.95, dist="normal")),
            "t":      Backtester(ParametricVaR(0.95, dist="t")),
        }
        df = compare_models_rolling(bts, synthetic_returns, 100,
                                    eval_start=test_returns.index[0])
        assert list(df["modelo"]) == ["normal", "t"]
        assert len(df) == 2

    def test_does_not_mutate_input_backtester(self, synthetic_returns, test_returns):
        # El Backtester pasado no debe quedar en estado post-run (se copia).
        bt = Backtester(ParametricVaR(0.95))
        compare_models_rolling({"P": bt}, synthetic_returns, 100,
                               eval_start=test_returns.index[0])
        assert bt.var_series_ is None

    def test_order_preserved(self, synthetic_returns, test_returns):
        bts = {
            "C": Backtester(ParametricVaR(0.95)),
            "A": Backtester(HistoricalVaR(0.95)),
            "B": Backtester(ParametricVaR(0.99)),
        }
        df = compare_models_rolling(bts, synthetic_returns, 100,
                                    eval_start=test_returns.index[0])
        assert list(df["modelo"]) == ["C", "A", "B"]

    # ── Errores ───────────────────────────────────────────────────────────────

    def test_empty_dict_raises(self, synthetic_returns):
        with pytest.raises(ValueError):
            compare_models_rolling({}, synthetic_returns, 100)

    def test_not_dict_raises(self, synthetic_returns):
        with pytest.raises(ValueError):
            compare_models_rolling([Backtester(ParametricVaR(0.95))],
                                   synthetic_returns, 100)

    def test_value_not_backtester_raises(self, synthetic_returns):
        # Pasar el modelo directamente (sin envolver en Backtester) es un error
        # claro de uso.
        with pytest.raises(TypeError):
            compare_models_rolling({"malo": ParametricVaR(0.95)},
                                   synthetic_returns, 100)

    # ── Caso de uso real (GARCH) ───────────────────────────────────────────────

    def test_compare_garch_distributions(self, synthetic_returns, test_returns):
        bts = {
            "GARCH-Normal": Backtester(GARCHParametricVaR(0.99, dist="normal")),
            "GARCH-t":      Backtester(GARCHParametricVaR(0.99, dist="t")),
        }
        df = compare_models_rolling(bts, synthetic_returns, 200,
                                    eval_start=test_returns.index[0])
        assert list(df["modelo"]) == ["GARCH-Normal", "GARCH-t"]
        assert len(df) == 2


class TestCompareModelsExpanding:

    def test_returns_dataframe_one_row_per_model(self, train_returns, test_returns):
        bts = {
            "Parametrico": Backtester(ParametricVaR(0.95)),
            "Historico":   Backtester(HistoricalVaR(0.95)),
        }
        df = compare_models_expanding(bts, train_returns, test_returns)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_label_used_as_identifier(self, train_returns, test_returns):
        bts = {
            "Modelo A": Backtester(ParametricVaR(0.95)),
            "Modelo B": Backtester(HistoricalVaR(0.95)),
        }
        df = compare_models_expanding(bts, train_returns, test_returns)
        assert list(df["modelo"]) == ["Modelo A", "Modelo B"]

    def test_expected_columns_present(self, train_returns, test_returns):
        df = compare_models_expanding({"P": Backtester(ParametricVaR(0.95))},
                                      train_returns, test_returns)
        for col in _EXPECTED_COLS:
            assert col in df.columns

    def test_violation_rate_in_range(self, train_returns, test_returns):
        df = compare_models_expanding({"P": Backtester(ParametricVaR(0.95))},
                                      train_returns, test_returns)
        assert len(df) == 1
        assert 0.0 <= df["violation_rate"].iloc[0] <= 1.0

    # ── Errores ───────────────────────────────────────────────────────────────

    def test_empty_dict_raises(self, train_returns, test_returns):
        with pytest.raises(ValueError):
            compare_models_expanding({}, train_returns, test_returns)

    def test_value_not_backtester_raises(self, train_returns, test_returns):
        with pytest.raises(TypeError):
            compare_models_expanding({"malo": ParametricVaR(0.95)},
                                     train_returns, test_returns)
