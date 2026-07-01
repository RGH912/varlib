"""
Tests para ReportExporter (HTML).

Se escribe en directorios temporales (tmp_path). La sección de
backtesting en HTML usa pandas Styler (requiere jinja2): ese test se
omite automáticamente si jinja2 no está instalado.
"""

import pandas as pd
import pytest

from varlib.reporting.reports import ReportExporter
from varlib.models.parametric import ParametricVaR
from varlib.validation.backtesting import Backtester


@pytest.fixture
def exporter():
    return ReportExporter(author="tests", ticker="AAPL")


@pytest.fixture
def backtest_summary(train_returns, test_returns):
    return Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns).summary()


class TestHelpers:

    def test_fmt_value_float(self):
        assert ReportExporter.fmt_value(0.123456789) == "0.123457"

    def test_fmt_value_small_float_scientific(self):
        assert "e" in ReportExporter.fmt_value(1e-6)

    def test_fmt_value_list(self):
        assert ReportExporter.fmt_value([0.1, 0.2]) == "0.100000, 0.200000"

    def test_fmt_value_str(self):
        assert ReportExporter.fmt_value("hola") == "hola"

    def test_unsupported_item_raises(self, exporter, tmp_path):
        # Un tipo no soportado (un escalar suelto) como item -> TypeError.
        with pytest.raises(TypeError):
            exporter.export_html({"X": 42}, filepath=str(tmp_path / "x.html"))


class TestHTML:

    def test_export_html_basic(self, exporter, tmp_path):
        out = tmp_path / "report.html"
        results = {
            "VaR Summary":      pd.DataFrame([{"metodo": "Parametrico", "var": -0.02}]),
            "Parámetros GARCH": {"omega": 1e-5, "alpha": [0.1]},   # dict -> clave/valor
        }
        exporter.export_html(results, filepath=str(out))
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "VaR Summary" in text and "AAPL" in text

    def test_export_html_with_backtesting(self, exporter, backtest_summary, tmp_path):
        pytest.importorskip("jinja2")
        out = tmp_path / "bt.html"
        exporter.export_html({"Backtesting": backtest_summary}, filepath=str(out))
        assert out.exists()
        assert "Backtesting" in out.read_text(encoding="utf-8")


class TestViolationsAndFigures:

    def test_html_with_violations_section(self, exporter, train_returns, test_returns, tmp_path):
        bt = Backtester(ParametricVaR(0.95)).run_expanding(train_returns, test_returns)
        violaciones_df = pd.DataFrame({
            "Fecha":      bt.var_series_.index.strftime("%Y-%m-%d"),
            "VaR (%)":    (bt.var_series_.values * 100).round(4),
            "Violación":  bt.violations_,
        })
        out = tmp_path / "viol.html"
        exporter.export_html({"Violaciones": violaciones_df}, filepath=str(out))
        assert "Violaciones" in out.read_text(encoding="utf-8")

    def test_html_with_figures(self, exporter, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [1, 2, 3])
        out = tmp_path / "figs.html"
        exporter.export_html({"Gráficos": fig}, filepath=str(out))
        plt.close(fig)
        text = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," in text   # figura embebida


class TestComposite:
    """
    Contenido como item único o como lista de items bajo un mismo título.
    """

    @staticmethod
    def _fig():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [1, 2, 3])
        return fig, plt

    def test_figure_as_value_embeds_in_html(self, exporter, tmp_path):
        fig, plt = self._fig()
        out = tmp_path / "f.html"
        exporter.export_html({"Distribución": fig}, filepath=str(out))
        plt.close(fig)
        assert "data:image/png;base64," in out.read_text(encoding="utf-8")

    def test_list_of_dataframes_under_one_section(self, exporter, tmp_path):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"x": [5], "y": [6]})
        out = tmp_path / "two.html"
        exporter.export_html({"Tablas": [df1, df2]}, filepath=str(out))
        html = out.read_text(encoding="utf-8")
        # Una sola sección "Tablas" que contiene las dos tablas.
        seccion = html.split("Tablas")[1].split("</section>")[0]
        assert seccion.count("<table") == 2

    def test_user_styler_format_respected(self, exporter, tmp_path):
        pytest.importorskip("jinja2")
        # El usuario controla el formato pasando un Styler; el valor sigue
        # siendo numérico pero se muestra como porcentaje.
        df = pd.DataFrame([{"VaR": 0.0213, "ES": 0.0285}])
        out = tmp_path / "styler.html"
        exporter.export_html({"Riesgo": df.style.format("{:.2%}")}, filepath=str(out))
        html = out.read_text(encoding="utf-8")
        assert "2.13%" in html and "2.85%" in html

    def test_dict_renders_as_key_value_table(self, exporter, tmp_path):
        # Un dict (p.ej. un .summary()) se renderiza como tabla clave/valor.
        out = tmp_path / "kv.html"
        exporter.export_html({"Params": {"omega": 1e-5, "alpha": 0.08}},
                             filepath=str(out))
        html = out.read_text(encoding="utf-8")
        assert "Parámetro" in html and "Valor" in html and "omega" in html

    def test_dict_text_trick_format_respected(self, exporter, tmp_path):
        # Si el valor ya viene como string (truco del texto), sale tal cual.
        out = tmp_path / "kv2.html"
        exporter.export_html({"R": {"VaR": f"{0.0213:.2%}"}}, filepath=str(out))
        assert "2.13%" in out.read_text(encoding="utf-8")

    def test_interleave_figure_between_tables_preserves_order(self, exporter, tmp_path):
        import re
        fig, plt = self._fig()
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "order.html"
        exporter.export_html(
            {"T1": df, "Fig": fig, "T2": df}, filepath=str(out)
        )
        plt.close(fig)
        html = out.read_text(encoding="utf-8")
        assert [m for m in re.findall(r"<h2>(.*?)</h2>", html)] == ["T1", "Fig", "T2"]


class TestFullReport:

    def test_export_html_creates_file(self, exporter, backtest_summary, tmp_path):
        out = tmp_path / "full_report.html"
        exporter.export_html({"Backtesting": backtest_summary}, filepath=str(out))
        assert out.exists() and out.stat().st_size > 0
