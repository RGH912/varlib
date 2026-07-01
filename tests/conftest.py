"""
conftest.py
===========
Fixtures compartidos por todos los tests de varlib.

Se generan series sintéticas deterministas con semilla fija para que
los tests sean reproducibles sin necesidad de descarga de datos reales.
"""

import matplotlib       # backend no interactivo, sin Tk: 
matplotlib.use("Agg")   # necesario para los tests de plots                        

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    """
    Generador de números aleatorios con semilla fija.
    """
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_returns(rng):
    """
    Serie de 500 retornos log-normales sintéticos (diarios).

    mu  = 0.0005  (~12.6% anual)
    sig = 0.015   (~23.8% anual vol)
    """
    n   = 500
    mu  = 0.0005
    sig = 0.015
    vals = mu + sig * rng.standard_normal(n)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(vals, index=dates, name="ret")


@pytest.fixture(scope="session")
def train_returns(synthetic_returns):
    """
    80% inicial de la serie sintética para entrenamiento.
    """
    n_train = int(len(synthetic_returns) * 0.8)
    return synthetic_returns.iloc[:n_train]


@pytest.fixture(scope="session")
def test_returns(synthetic_returns):
    """
    20% final de la serie sintética para test.
    """
    n_train = int(len(synthetic_returns) * 0.8)
    return synthetic_returns.iloc[n_train:]


@pytest.fixture(scope="session")
def controlled_returns():
    """
    Serie determinista de 100 valores con percentiles exactos conocidos.

    Se usa para tests de resultado exacto: como los valores son fijos,
    np.percentile sobre ellos da un número conocido con el que comparar
    la salida del modelo histórico mediante assert_allclose.
    """
    vals = np.linspace(-0.10, 0.099, 100)   # 100 valores equiespaciados
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    return pd.Series(vals, index=dates, name="ctrl")


@pytest.fixture(scope="session")
def large_normal_returns():
    """
    Muestra grande N(mu, sigma) para tests de coherencia entre métodos.

    Con suficientes observaciones, el VaR histórico, paramétrico y Monte
    Carlo deben converger (todos describen la misma Normal subyacente).
    Semilla fija -> reproducible.
    """
    rng = np.random.default_rng(123)
    n   = 20_000
    mu, sig = 0.0004, 0.012
    vals = mu + sig * rng.standard_normal(n)
    dates = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.Series(vals, index=dates, name="normal_ret")


@pytest.fixture(scope="session")
def price_series():
    """
    Serie sintética de precios positivos (sin red), para DataLoader.

    Permite probar el cálculo de rendimientos y la partición inyectando
    estos precios en DataLoader.prices_ sin descargar de yfinance.
    """
    rng = np.random.default_rng(7)
    n = 300
    log_ret = 0.0003 + 0.01 * rng.standard_normal(n)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="PRICE")


@pytest.fixture(scope="session")
def volatile_returns(rng):
    """
    Serie de 300 retornos con clustering de volatilidad (GARCH-like).
    """
    n = 300
    omega, alpha, beta = 1e-5, 0.1, 0.88
    mu = 0.0003

    sigma2 = np.empty(n)
    eps    = np.empty(n)
    ret    = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)

    z = rng.standard_normal(n)
    for t in range(n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1] if t > 0 else sigma2[0]
        eps[t]    = np.sqrt(sigma2[t]) * z[t]
        ret[t]    = mu + eps[t]

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(ret, index=dates, name="garch_ret")
