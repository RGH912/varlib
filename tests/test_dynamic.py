"""
Tests para varlib.validation.dynamic.dynamic_var (capa de ventana deslizante).

Incluye:
- Alineación y tipo de la serie devuelta; convención de signo.
- Ausencia de look-ahead: equivalencia con un cálculo manual que usa solo
  datos anteriores a t.
- Semántica de step (NaN en fechas no evaluadas).
- compute_es=True devuelve DataFrame VaR/ES.
- Validación de argumentos.
"""

import numpy as np
import pandas as pd
import pytest

from varlib.models.base import BaseVaR
from varlib.models.parametric import ParametricVaR
from varlib.validation.dynamic import dynamic_var


class _FailsForLongWindows(BaseVaR):
    """
    Modelo de prueba: fit() falla cuando la ventana es suficientemente
    grande. Permite ejercitar la política on_error de dynamic_var."""

    def __init__(self, fail_when_len_ge):
        super().__init__()
        self.fail_when_len_ge = fail_when_len_ge

    def fit(self, returns):
        if len(returns) >= self.fail_when_len_ge:
            raise RuntimeError("fallo simulado de ajuste")
        self._returns = returns
        self._fitted = True
        return self

    def compute_var(self):
        return 0.02


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(2024)
    n = 400
    vals = 0.0003 + 0.011 * rng.standard_normal(n)
    return pd.Series(vals, index=pd.date_range("2019-01-01", periods=n, freq="B"))


class TestOutput:

    def test_returns_series_aligned(self, returns):
        var = dynamic_var(ParametricVaR(0.95), returns, window=100)
        assert isinstance(var, pd.Series)
        assert var.name == "VaR"
        assert var.index.equals(returns.index)

    def test_warmup_is_nan(self, returns):
        var = dynamic_var(ParametricVaR(0.95), returns, window=100)
        # Las primeras `window` fechas no tienen historia suficiente.
        assert var.iloc[:100].isna().all()
        assert var.iloc[100:].notna().all()

    def test_var_positive(self, returns):
        var = dynamic_var(ParametricVaR(0.95), returns, window=100)
        assert (var.dropna() > 0).all()


class TestNoLookAhead:

    def test_matches_manual_expanding(self, returns):
        """
        Cada VaR de t debe coincidir con ajustar SOLO con datos < t.
        """
        window = 60
        var = dynamic_var(ParametricVaR(0.95), returns, window=window, expanding=True)
        # Comprobamos varios puntos manualmente.
        for t in (window, window + 1, 150, 399):
            manual = ParametricVaR(0.95).fit(returns.iloc[:t]).compute_var()
            np.testing.assert_allclose(var.iloc[t], manual, rtol=1e-12)

    def test_matches_manual_rolling(self, returns):
        window = 80
        var = dynamic_var(ParametricVaR(0.95), returns, window=window, expanding=False)
        for t in (window, 200, 399):
            manual = ParametricVaR(0.95).fit(returns.iloc[t - window:t]).compute_var()
            np.testing.assert_allclose(var.iloc[t], manual, rtol=1e-12)

    def test_future_spike_does_not_affect_past(self, returns):
        """
        Alterar el ÚLTIMO retorno no debe cambiar los VaR anteriores.
        """
        base = dynamic_var(ParametricVaR(0.95), returns, window=100)
        perturbed = returns.copy()
        perturbed.iloc[-1] = -0.5      # shock enorme en t final
        after = dynamic_var(ParametricVaR(0.95), perturbed, window=100)
        # Todos los VaR salvo el último deben ser idénticos.
        np.testing.assert_allclose(base.iloc[:-1].dropna(),
                                   after.iloc[:-1].dropna(), rtol=1e-12)


class TestStep:

    def test_step_leaves_nan_between(self, returns):
        var = dynamic_var(ParametricVaR(0.95), returns, window=100, step=5)
        evaluated = var.iloc[100:]
        # Posiciones evaluadas: 100, 105, 110, ...
        non_nan_positions = np.where(evaluated.notna().values)[0]
        assert set(np.diff(non_nan_positions)) == {5}

    def test_step_one_is_dense(self, returns):
        var = dynamic_var(ParametricVaR(0.95), returns, window=100, step=1)
        assert var.iloc[100:].notna().all()


class TestSlidingWindow:

    def test_var_changes_each_step(self, returns):
        """
        El reajuste en cada paso hace que el VaR cambie día a día.
        """
        var = dynamic_var(ParametricVaR(0.95), returns, window=100).dropna()
        changes = int((var.diff().dropna().abs() > 1e-15).sum())
        # Casi todos los pasos cambian el VaR (ventana deslizante diaria).
        assert changes >= len(var) - 2


class TestES:

    def test_compute_es_returns_dataframe(self, returns):
        out = dynamic_var(ParametricVaR(0.95), returns, window=100, compute_es=True)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["VaR", "ES"]

    def test_es_geq_var(self, returns):
        out = dynamic_var(ParametricVaR(0.95), returns, window=100, compute_es=True).dropna()
        assert (out["ES"] >= out["VaR"] - 1e-12).all()


class TestOnError:

    def test_previous_carries_last_valid(self, returns):
        """
        on_error='previous': tras un fallo, arrastra el último VaR válido.
        """
        model = _FailsForLongWindows(fail_when_len_ge=200)
        var = dynamic_var(model, returns, window=40, expanding=True, on_error="previous")
        assert var.iloc[100] == 0.02           # ventana < 200 -> ajusta bien
        assert var.iloc[250] == 0.02           # ventana >= 200 -> falla, carga previo

    def test_nan_leaves_nan_on_failure(self, returns):
        model = _FailsForLongWindows(fail_when_len_ge=200)
        var = dynamic_var(model, returns, window=40, expanding=True, on_error="nan")
        assert var.iloc[100] == 0.02
        assert np.isnan(var.iloc[250])         # ventana >= 200 -> falla, deja NaN


class TestErrors:

    def test_model_not_basevar_raises(self, returns):
        with pytest.raises(TypeError):
            dynamic_var("not_a_model", returns, window=50)

    def test_returns_not_series_raises(self):
        with pytest.raises(TypeError):
            dynamic_var(ParametricVaR(0.95), [0.01] * 100, window=50)

    def test_window_below_min_obs_raises(self, returns):
        with pytest.raises(ValueError):
            dynamic_var(ParametricVaR(0.95), returns, window=10, min_obs=30)

    @pytest.mark.parametrize("kwargs", [{"step": 0}, {"window": 0}])
    def test_invalid_integer_args_raise(self, returns, kwargs):
        params = {"window": 100}
        params.update(kwargs)
        with pytest.raises(ValueError):
            dynamic_var(ParametricVaR(0.95), returns, **params)

    def test_invalid_on_error_raises(self, returns):
        with pytest.raises(ValueError):
            dynamic_var(ParametricVaR(0.95), returns, window=100, on_error="boom")
