"""
Tests para BaseVaR (clase abstracta) y su contrato común.

Se usa una subclase mínima _DummyVaR para ejercitar la lógica de la
clase base (validación de parámetros, _validate_returns, compute_es
empírico por defecto, summary y los RuntimeError previos a fit) sin
depender de la implementación concreta de ningún modelo.
"""

import numpy as np
import pandas as pd
import pytest

from varlib.models.base import BaseVaR


class _DummyVaR(BaseVaR):
    """
    Modelo mínimo: VaR = pérdida positiva (negativo del percentil),
    ES heredado de la base.
    """

    def fit(self, returns):
        self._validate_returns(returns)
        self._returns = returns.dropna()
        self._fitted = True
        return self

    def compute_var(self):
        self._check_fitted()
        # Pérdida positiva = negativo del cuantil (1 - confidence) de retorno.
        return -float(np.percentile(self._returns.values, (1.0 - self.confidence) * 100))


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(200) * 0.01
    return pd.Series(vals, index=pd.date_range("2020-01-01", periods=200, freq="B"))


class TestBaseValidation:

    @pytest.mark.parametrize("conf", [-0.1, 0.0, 1.0, 1.5])
    def test_confidence_out_of_range_raises(self, conf):
        with pytest.raises(ValueError):
            _DummyVaR(confidence=conf)

    @pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
    def test_confidence_valid_ok(self, conf):
        assert _DummyVaR(confidence=conf).confidence == conf

    @pytest.mark.parametrize("h", [0, -1, 1.5, "2"])
    def test_horizon_invalid_raises(self, h):
        with pytest.raises(ValueError):
            _DummyVaR(horizon=h)

    def test_horizon_valid_ok(self):
        assert _DummyVaR(horizon=10).horizon == 10


class TestValidateReturns:

    def test_wrong_type_raises(self):
        model = _DummyVaR()
        with pytest.raises(TypeError):
            model.fit([0.01, 0.02, 0.03])     # lista, no pd.Series

    def test_too_few_obs_raises(self):
        model = _DummyVaR()
        with pytest.raises(ValueError):
            model.fit(pd.Series(np.arange(10) * 0.001))

    def test_nan_emits_warning_not_error(self, dummy_data):
        data = dummy_data.copy()
        data.iloc[5] = np.nan        # >=30 válidos, solo avisa
        model = _DummyVaR()
        with pytest.warns(UserWarning):
            model.fit(data)
        assert model._fitted
        # El NaN se descarta: n usado = válidos
        assert len(model._returns) == len(data.dropna())


class TestNotFitted:

    @pytest.mark.parametrize("method", ["compute_var", "compute_es", "summary"])
    def test_method_before_fit_raises(self, method):
        with pytest.raises(RuntimeError):
            getattr(_DummyVaR(), method)()


class TestDefaultBehaviour:

    def test_es_empirical_is_tail_mean(self, dummy_data):
        """
        El compute_es por defecto = -media de retornos <= -VaR (pérdida).
        """
        model = _DummyVaR(confidence=0.95).fit(dummy_data)
        var = model.compute_var()               # pérdida positiva
        tail = dummy_data[dummy_data <= -var]    # retornos por debajo de -VaR
        expected = float(-tail.mean())           # pérdida media positiva
        np.testing.assert_allclose(model.compute_es(), expected, rtol=1e-12)

    def test_es_geq_var(self, dummy_data):
        # Con VaR/ES positivos, el ES (peor pérdida) es >= VaR.
        model = _DummyVaR(confidence=0.95).fit(dummy_data)
        assert model.compute_es() >= model.compute_var()

    def test_summary_minimum_keys(self, dummy_data):
        s = _DummyVaR().fit(dummy_data).summary()
        for key in ("model", "confidence", "horizon", "var", "es", "n_obs"):
            assert key in s
        assert s["model"] == "_DummyVaR"

    def test_repr_reflects_status(self, dummy_data):
        model = _DummyVaR()
        assert "sin ajustar" in repr(model)
        model.fit(dummy_data)
        assert "ajustado" in repr(model)

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseVaR()        # fit y compute_var son abstractos
