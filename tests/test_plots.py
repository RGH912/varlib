"""
Tests para VaRPlotter.

Las figuras se generan en memoria. Se verifica que cada método devuelve
una matplotlib.figure.Figure y que save escribe el archivo en disco.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from varlib.reporting.plots import VaRPlotter
from varlib.models.parametric import ParametricVaR
from varlib.validation.backtesting import Backtester


@pytest.fixture
def plotter():
    return VaRPlotter()


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestFigures:

    def test_var_scalar(self, plotter, train_returns):
        # VaR estático: escalar (pérdida positiva).
        fig = plotter.plot_var(train_returns, 0.02, 0.03)
        assert isinstance(fig, Figure)

    def test_var_series(self, plotter, train_returns, test_returns):
        # VaR dinámico: serie (la del Backtester).
        bt = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns)
        fig = plotter.plot_var(bt.test_returns_, bt.var_series_)
        assert isinstance(fig, Figure)

    def test_volatility(self, plotter):
        idx = pd.date_range("2022-01-01", periods=100, freq="B")
        vol = pd.Series(np.abs(np.random.default_rng(0).standard_normal(100)) * 0.2, index=idx)
        fig = plotter.plot_volatility(vol, forecast=np.linspace(0.2, 0.25, 10))
        assert isinstance(fig, Figure)

    def test_simulation_histogram(self, plotter):
        sims = np.random.default_rng(0).standard_normal(10_000) * 0.01
        fig = plotter.plot_simulation_histogram(sims, var_value=0.02, es_value=0.025)
        assert isinstance(fig, Figure)


class TestSave:

    def test_save_creates_file(self, plotter, train_returns, tmp_path):
        fig = plotter.plot_var(train_returns, 0.02)
        out = tmp_path / "fig.png"
        VaRPlotter.save(fig, str(out))
        assert out.exists() and out.stat().st_size > 0


class TestPlotFixes:
    """
    Arreglos: plot rolling, alineación por índice, vol anualizada y
    estilo no-global."""

    def test_dynamic_var_returns_figure(self, plotter):
        idx = pd.date_range("2021-01-01", periods=300, freq="B")
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.standard_normal(300) * 0.01, index=idx)
        var_series = pd.Series(np.full(300, 0.02), index=idx)
        var_series.iloc[:50] = np.nan                 # warm-up sin VaR
        fig = plotter.plot_var(returns, var_series)
        assert isinstance(fig, Figure)

    def test_dynamic_var_with_es(self, plotter):
        idx = pd.date_range("2021-01-01", periods=120, freq="B")
        returns = pd.Series(np.random.default_rng(3).standard_normal(120) * 0.01, index=idx)
        var  = pd.Series(np.full(120, 0.02), index=idx)
        es = pd.Series(np.full(120, 0.03), index=idx)
        assert isinstance(plotter.plot_var(returns, var, es), Figure)

    def test_violations_aligns_by_index(self, plotter):
        # var_series con el mismo índice pero desordenado y con fechas de más:
        # debe alinear por fecha, no comparar posicionalmente.
        idx = pd.date_range("2021-01-01", periods=100, freq="B")
        returns = pd.Series(np.random.default_rng(1).standard_normal(100) * 0.01, index=idx)
        var = pd.Series(np.full(120, 0.02),
                        index=pd.date_range("2021-01-01", periods=120, freq="B"))
        var_shuffled = var.sample(frac=1.0, random_state=2)
        fig = plotter.plot_var(returns, var_shuffled)
        assert isinstance(fig, Figure)

    def test_volatility_annualize_flag(self, plotter):
        idx = pd.date_range("2022-01-01", periods=60, freq="B")
        daily_vol = pd.Series(np.full(60, 0.01), index=idx)   # 1% diario
        # annualize=True (anualiza) y annualize=False (deja diaria) deben pintar.
        assert isinstance(plotter.plot_volatility(daily_vol, annualize=True), Figure)
        assert isinstance(plotter.plot_volatility(daily_vol, annualize=False), Figure)

    def test_style_not_applied_globally(self):
        before = plt.rcParams["axes.grid"]
        VaRPlotter(style="seaborn-v0_8-whitegrid")
        # Construir el plotter NO debe mutar el rcParams global de matplotlib.
        assert plt.rcParams["axes.grid"] == before
