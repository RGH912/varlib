"""
Tests para varlib.reporting.console (format_summary / print_summary).
"""

import pandas as pd

from varlib.reporting.console import format_summary, print_summary
from varlib.models.parametric import ParametricVaR


class TestFormatSummary:

    def test_dict_aligned(self):
        out = format_summary({"a": 1, "bbb": 2.5})
        assert "a" in out and "bbb" in out
        # Las claves se alinean: ambas líneas tienen el ':' en la misma columna.
        cols = [line.index(":") for line in out.splitlines() if ":" in line]
        assert len(set(cols)) == 1

    def test_title_underlined(self):
        out = format_summary({"x": 1}, title="Cabecera")
        lines = out.splitlines()
        assert lines[0] == "Cabecera"
        assert lines[1] == "-" * len("Cabecera")

    def test_object_with_summary(self, train_returns):
        model = ParametricVaR(0.95).fit(train_returns)
        out = format_summary(model, "Parametrico")
        # Debe contener claves del summary del modelo.
        assert "model" in out and "var" in out and "sigma" in out

    def test_dataframe(self):
        df = pd.DataFrame([{"modelo": "A", "p_uc": 0.5}])
        out = format_summary(df)
        assert "modelo" in out and "A" in out

    def test_float_formatting(self):
        # Float normal con 6 decimales; muy pequeño en notación científica.
        out = format_summary({"normal": 0.123456, "tiny": 1e-6})
        assert "0.123456" in out
        assert "e" in out  # 1e-6 -> notación científica

    def test_list_values(self):
        out = format_summary({"alpha": [0.1, 0.88]})
        assert "0.100000, 0.880000" in out

    def test_print_summary_outputs(self, capsys):
        print_summary({"k": 1}, title="T")
        captured = capsys.readouterr().out
        assert "T" in captured and "k" in captured
