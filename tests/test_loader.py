"""
Tests para DataLoader SIN acceso a red ni a yfinance.

La descarga real (download) no se prueba aquí (requeriría red). En su
lugar se inyecta una serie de precios sintética en loader.prices_ y se
ejercita todo el preprocesamiento: cálculo de rendimientos, caché,
partición train/test, summary, validaciones y los RuntimeError previos a
la descarga.
"""

import numpy as np
import pytest

from varlib.data.loader import DataLoader


@pytest.fixture
def loader(price_series):
    """
    DataLoader con precios inyectados (sin llamar a download).
    """
    ld = DataLoader("aapl", start="2022-01-01", end="2023-01-01")
    ld.prices_ = price_series.copy()
    return ld


class TestInit:

    def test_non_string_ticker_raises(self):
        # El constructor solo valida el tipo; un ticker no-cadena lanza ValueError.
        with pytest.raises(ValueError):
            DataLoader(123, start="2022-01-01", end="2023-01-01")

    def test_ticker_with_spaces_raises(self):
        # Validación estricta: cualquier espacio (incl. sobrantes o multi-ticker)
        # se rechaza en el constructor.
        with pytest.raises(ValueError):
            DataLoader("  msft ", start="2022-01-01", end="2023-01-01")
        with pytest.raises(ValueError):
            DataLoader("AAPL MSFT", start="2022-01-01", end="2023-01-01")

    def test_ticker_with_comma_raises(self):
        with pytest.raises(ValueError):
            DataLoader("AAPL,MSFT", start="2022-01-01", end="2023-01-01")

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError):
            DataLoader("AAPL", start="2022-01-01", end="2023-01-01", interval="7d")

    def test_ticker_normalised(self):
        # Un ticker sin espacios se normaliza a mayúsculas.
        ld = DataLoader("msft", start="2022-01-01", end="2023-01-01")
        assert ld.ticker == "MSFT"

    def test_repr_status(self):
        ld = DataLoader("AAPL", start="2022-01-01", end="2023-01-01")
        assert "sin descargar" in repr(ld)


class TestBeforeDownload:

    def test_log_returns_before_download_raises(self):
        ld = DataLoader("AAPL", start="2022-01-01", end="2023-01-01")
        with pytest.raises(RuntimeError):
            ld.get_log_returns()

    def test_summary_before_download_raises(self):
        ld = DataLoader("AAPL", start="2022-01-01", end="2023-01-01")
        with pytest.raises(RuntimeError):
            ld.summary()

    def test_split_before_download_raises(self):
        ld = DataLoader("AAPL", start="2022-01-01", end="2023-01-01")
        with pytest.raises(RuntimeError):
            ld.split(100)


class TestReturns:

    def test_log_returns_match_formula(self, loader, price_series):
        lr = loader.get_log_returns()
        expected = np.log(price_series / price_series.shift(1)).dropna()
        np.testing.assert_allclose(lr.values, expected.values, rtol=1e-12)
        assert len(lr) == len(price_series) - 1

    def test_simple_returns_match_formula(self, loader, price_series):
        sr = loader.get_simple_returns()
        expected = price_series.pct_change().dropna()
        np.testing.assert_allclose(sr.values, expected.values, rtol=1e-12)

    def test_returns_are_cached(self, loader):
        first  = loader.get_log_returns()
        second = loader.get_log_returns()
        assert first is second        # mismo objeto desde la caché


class TestSplit:

    def test_split_warmup(self, loader):
        train, test = loader.split(warmup=100)
        n = len(loader.get_log_returns())
        assert len(train) == 100
        assert len(train) + len(test) == n

    def test_split_is_temporal(self, loader):
        train, test = loader.split(warmup=100)
        assert train.index[-1] < test.index[0]

    def test_split_warmup_too_large_raises(self, loader):
        n = len(loader.get_log_returns())
        with pytest.raises(ValueError):
            loader.split(warmup=n)            # no deja test

    def test_split_warmup_zero_raises(self, loader):
        with pytest.raises(ValueError):
            loader.split(warmup=0)

    def test_split_invalid_return_type_raises(self, loader):
        with pytest.raises(ValueError):
            loader.split(return_type="quadratic")

    def test_split_simple_returns(self, loader):
        train, test = loader.split(warmup=100, return_type="simple")
        assert len(train) > 0 and len(test) > 0


class TestSummary:

    def test_summary_keys(self, loader):
        s = loader.summary()
        for key in ("ticker", "start", "end", "n_prices", "n_returns",
                    "mean_return", "std_return", "skewness", "kurtosis"):
            assert key in s

    def test_summary_n_returns_consistent(self, loader):
        s = loader.summary()
        assert s["n_returns"] == s["n_prices"] - 1
