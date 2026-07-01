"""
varlib.reporting.reports
=========================
Exportación de reportes en HTML autónomo y auto-contenido.

Clases
------
ReportExporter
    Recibe un diccionario {titulo: contenido} y genera una sección HTML por
    entrada. El contenido puede ser un item o una lista de items (que se apilan
    bajo el mismo título), renderizados por tipo:
    - dict              -> tabla clave/valor (p.ej. un .summary() de un modelo
      o del Backtester)
    - pd.DataFrame       -> tabla. El df de comparación (compare_models_*) se
      detecta y prepara automáticamente
    - pandas Styler      -> su HTML tal cual (formato a gusto del usuario)
    - matplotlib Figure  -> imagen embebida (base64)
    En la tabla clave/valor los escalares se formatean con fmt_value. Para un
    formato concreto, se pasa el valor ya como string (truco del texto).
    Las columnas de p-valor y booleanas se colorean por contenido.

Dependencias
------------
- matplotlib (para los gráficos embebidos vía base64)
"""

import base64
import io
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

try:
    import matplotlib  # noqa: F401  (solo para detectar disponibilidad)
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class ReportExporter:
    """
    Exporta resultados VaR a un reporte HTML autónomo.

    Parameters
    ----------
    author : str, optional
        Nombre del autor para los metadatos del reporte.
    ticker : str, optional
        Ticker del activo analizado.
    significance : float, optional
        Nivel de significación con el que se colorean las celdas de p-valor
        (rojo si p < significance, verde si no). Debe coincidir con el usado
        en el Backtester. Por defecto 0.05.

    Examples
    --------
    exporter = ReportExporter(author="PFG", ticker="AAPL")
    exporter.export_html(results, figures, filepath="output/aapl_report.html")
    """

    def __init__(
        self,
        author: str = "varlib",
        ticker: str = "N/A",
        significance: float = 0.05,
    ) -> None:
        if not 0.0 < significance < 1.0:
            raise ValueError(
                f"[significance] debe estar en (0, 1). Recibido: '{significance}'"
            )
        self.author = author
        self.ticker = ticker
        self.significance = significance

    # ── HTML ──────────────────────────────────────────────────────────────────

    def export_html(
        self,
        results: dict[str, Any],
        filepath: str = "report.html",
    ) -> None:
        """
        Exporta resultados a un archivo HTML autónomo y auto-contenido.

        Las figuras se embeben como imágenes base64 (sin dependencias externas)
        y van como un valor más en results, en el orden/sección que se quiera.

        Parameters
        ----------
        results : dict
            Diccionario {titulo: contenido}. El contenido puede ser un único
            item o una lista de items que se apilan bajo el mismo título de
            sección. Cada item se renderiza por tipo: dict -> tabla clave/valor
            (p.ej. un .summary()), DataFrame -> tabla (el df de comparación se
            detecta y prepara solo), pandas Styler -> su HTML tal cual (formato
            con df.style.format(...)) y matplotlib Figure -> imagen embebida.
        filepath : str, optional
            Ruta de salida. El directorio padre debe existir (si no, se
            lanza FileNotFoundError al escribir).
        """
        sections_html = []
        for title, content in results.items():
            # Una clave -> una sección. El valor puede ser un item o una lista
            # de items que se apilan (con separación) bajo el mismo título.
            items = [h for h in (self._item_to_html(it)
                                 for it in self._as_list(content)) if h]
            body = "\n".join(f'<div class="report-item">{h}</div>' for h in items)
            sections_html.append(self._html_section(str(title), body))

        html = self._build_html(
            title=f"VaR Report - {self.ticker}",
            sections=sections_html,
            author=self.author,
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

    # ── Utilidades internas ────────────────────────────────────────────────────

    @staticmethod
    def fmt_value(v: Any) -> str:
        """
        Formatea un escalar para una celda de tabla: notación científica para
        valores muy pequeños, 6 decimales para el resto, listas separadas por
        comas. Útil para construir la columna de valores de una tabla
        clave/valor a partir de un dict (mismo criterio que print_summary).
        """
        if isinstance(v, float):
            if abs(v) < 1e-4 and v != 0:
                return f"{v:.4e}"
            return f"{v:.6f}"
        if isinstance(v, list):
            return ", ".join(f"{x:.6f}" for x in v)
        return str(v)

    # ── Detección y preparación del df de comparación ──────────────────────────

    # Columnas (y orden) con que se presenta el df de compare_models_* en el
    # reporte: las que de verdad se miran en un backtest, no todo el summary.
    _COMPARISON_COLS = [
        "modelo", "confianza", "n_obs", "n_violations", "violation_rate",
        "expected_rate", "p_uc", "reject_uc", "p_ind", "reject_ind",
        "p_cc", "reject_cc",
    ]
    # Esquema completo de Backtester.summary(). Lo que esté fuera de él es una
    # columna añadida por el usuario (p.ej. "Ventana") y se conserva.
    _SUMMARY_SCHEMA = {
        "modelo", "confianza", "n_obs", "n_violations", "violation_rate",
        "expected_rate", "LR_uc", "p_uc", "reject_uc",
        "LR_ind", "p_ind", "reject_ind", "LR_cc", "p_cc", "reject_cc",
    }

    @staticmethod
    def _is_comparison_df(df: pd.DataFrame) -> bool:
        """
        True si df parece la salida de compare_models_* (se reconoce por sus
        columnas características).
        """
        signature = {"modelo", "n_violations", "violation_rate", "p_uc"}
        return signature.issubset(df.columns)

    @classmethod
    def _prepare_comparison_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deja el df de comparación listo para el reporte: selecciona las columnas
        relevantes y formatea las tasas como porcentaje. Los p-valores se
        muestran con 4 decimales (como string, el coloreado los relee con
        float()).
        """
        # Columnas extra del usuario (no del summary, p.ej. "Ventana") primero,
        # luego el subconjunto relevante en su orden.
        extra = [c for c in df.columns if c not in cls._SUMMARY_SCHEMA]
        cols = extra + [c for c in cls._COMPARISON_COLS if c in df.columns]
        out = df[cols].copy()
        for c in ("confianza", "violation_rate", "expected_rate"):
            if c in out.columns:
                out[c] = out[c].map(lambda x: f"{x:.2%}")
        for c in ("p_uc", "p_ind", "p_cc"):
            if c in out.columns:
                out[c] = out[c].map(lambda x: f"{x:.4f}")
        return out

    # ── Coloreado por contenido (p-valores y booleanos) ────────────────────────

    @staticmethod
    def _cell_color(col_name: str, value: Any, significance: float = 0.05) -> str | None:
        """
        Color de fondo (hex sin '#') de una celda según su contenido, o None.

        - Columnas p_* : rojo si p < significance (se rechaza H0), verde si no.
        - Booleanos    : rojo si True (rechazo/violación), verde si False.
        """
        if str(col_name).startswith("p_"):
            try:
                return "FFCDD2" if float(value) < significance else "C8E6C9"
            except (TypeError, ValueError):
                return None
        if isinstance(value, (bool, np.bool_)):
            return "FFCDD2" if bool(value) else "C8E6C9"
        return None

    @classmethod
    def _df_to_html(cls, df: pd.DataFrame, significance: float = 0.05) -> str:
        """
        Renderiza un DataFrame como tabla HTML. Si tiene columnas coloreables
        (p_* o booleanas) usa el Styler de pandas. Si no, to_html plano.
        """
        colorable = any(
            str(c).startswith("p_") or pd.api.types.is_bool_dtype(df[c])
            for c in df.columns
        )
        if not colorable:
            return df.to_html(index=False, classes="var-table", border=0)

        def _style_row(row):
            return [
                f"background-color: #{c}" if (c := cls._cell_color(col, row[col], significance)) else ""
                for col in row.index
            ]

        return cls._styler_to_html(df.style.apply(_style_row, axis=1))

    @staticmethod
    def _styler_to_html(styled: Styler) -> str:
        """
        Renderiza un Styler de pandas a HTML ocultando el índice. Sirve tanto
        para el coloreado interno como para un Styler que pase el usuario (con
        su propio .format()/.background_gradient()/etc.). Si falta jinja2, cae a
        la tabla plana del DataFrame subyacente.
        """
        try:
            try:
                styled = styled.hide(axis="index")          # pandas >= 1.4
            except (AttributeError, TypeError):
                styled = styled.hide_index()                # pandas < 1.4
            return styled.to_html(table_attributes='class="var-table"')
        except Exception:
            return styled.data.to_html(index=False, classes="var-table", border=0)

    # ── Despacho de contenido por tipo (item o lista de items) ─────────────────

    @staticmethod
    def _as_list(content: Any) -> list:
        """
        Normaliza el contenido de una sección a lista de items. Una lista/tupla
        se trata como varios items bajo el mismo título. Cualquier otra cosa
        (DataFrame, dict, Figure, Series...) es un único item.
        """
        if isinstance(content, (list, tuple)):
            return list(content)
        return [content]

    @staticmethod
    def _is_figure(obj: Any) -> bool:
        """
        True si obj es una figura de matplotlib.
        """
        if not _HAS_MPL:
            return False
        from matplotlib.figure import Figure
        return isinstance(obj, Figure)

    def _item_to_html(self, item: Any) -> str:
        """
        Renderiza un único item a HTML según su tipo: figura -> imagen, Styler
        de pandas -> su HTML tal cual (respeta el formato del usuario), dict ->
        tabla clave/valor (p.ej. un .summary()), DataFrame -> tabla (el df de
        comparación se detecta y prepara solo).

        En la tabla clave/valor los valores se formatean con fmt_value. Para un
        formato concreto, pasa el valor ya como string (p.ej. f"{x:.4%}").
        """
        if self._is_figure(item):
            try:
                b64 = self._fig_to_base64(item)
            except Exception:
                return ""
            return (
                f'<div class="fig-container">'
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%;height:auto;" /></div>'
            )
        if isinstance(item, Styler):
            return self._styler_to_html(item)
        if isinstance(item, dict):
            # dict -> tabla clave/valor (DataFrame de 2 columnas, valores
            # formateados con fmt_value salvo que ya vengan como string).
            item = pd.DataFrame(
                [(str(k), self.fmt_value(v)) for k, v in item.items()],
                columns=["Parámetro", "Valor"],
            )
        if not isinstance(item, pd.DataFrame):
            raise TypeError(
                f"[results] cada item debe ser un DataFrame, un dict, un Styler "
                f"o una figura. Recibido: '{type(item).__name__}'."
            )
        df = item
        if self._is_comparison_df(df):
            df = self._prepare_comparison_df(df)
        return self._df_to_html(df, self.significance)

    @staticmethod
    def _fig_to_base64(fig) -> str:
        """
        Convierte una figura matplotlib a PNG en base64.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @staticmethod
    def _html_section(title: str, content_html: str) -> str:
        return (
            f'<section class="report-section">'
            f'<h2>{title}</h2>'
            f'{content_html}'
            f'</section>'
        )

    @staticmethod
    def _build_html(title: str, sections: list[str], author: str = "varlib") -> str:
        css = """
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #F5F5F5;
               color: #212121; margin: 0; padding: 20px; }
        h1   { color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 8px; }
        h2   { color: #1976D2; margin-top: 30px; }
        .report-section { background: white; padding: 20px 25px;
                          border-radius: 6px; margin-bottom: 20px;
                          box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        .var-table { border-collapse: collapse; width: 100%; font-size: 13px; }
        .var-table th { background: #1565C0; color: white; padding: 8px 12px;
                        text-align: center; }
        .var-table td { padding: 6px 12px; border: 1px solid #BDBDBD;
                        text-align: center; }
        .var-table tr:nth-child(even) td { background: #E3F2FD; }
        .fig-container { margin: 12px 0; text-align: center; }
        .report-item + .report-item { margin-top: 22px; }
        .meta { color: #757575; font-size: 12px; margin-bottom: 15px; }
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        header = (
            f'<h1>{title}</h1>'
            f'<p class="meta">Autor: {author} | Generado: {now} | varlib</p>'
        )
        body = header + "\n".join(sections)
        return (
            f'<!DOCTYPE html>\n<html lang="es">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>{title}</title>\n'
            f'<style>{css}</style>\n'
            f'</head>\n<body>\n{body}\n</body>\n</html>\n'
        )
