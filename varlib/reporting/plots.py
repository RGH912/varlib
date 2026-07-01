"""
varlib.reporting.plots
=======================
Visualizaciones para modelos VaR: series temporales, comparativas,
violaciones de backtesting y volatilidad condicional GARCH.

Gráficos disponibles
--------------------
- plot_var: VaR sobre los retornos, con violaciones. Acepta el VaR como
  escalar (estático, ventana fija) o como serie (dinámico, p.ej. salida
  de dynamic_var o del Backtester). Sirve igual para el test o la serie
  completa según el tramo de retornos que se pase.
- plot_simulation_histogram: distribución simulada (Monte Carlo) con
  VaR/ES.
- plot_volatility: volatilidad condicional GARCH + pronóstico.

Uso típico
----------
from varlib.reporting.plots import VaRPlotter
plotter = VaRPlotter()
fig = plotter.plot_var(returns, model.compute_var())   # escalar o serie
plotter.save(fig, "var.png")
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from varlib.models.base import annualize_volatility


class VaRPlotter:
    """
    Clase de visualizacion para modelos VaR.

    Todos los métodos devuelven una matplotlib.figure.Figure que
    puede mostrarse con .show(), guardarse con VaRPlotter.save()
    o embeberse en HTML con varlib.reporting.reports.ReportExporter.

    El estilo de matplotlib se aplica de forma **local** a cada figura
    (mediante plt.style.context), sin modificar el estado global del
    proceso.

    Parameters
    ----------
    style : str, optional
        Estilo de matplotlib. Por defecto 'seaborn-v0_8-whitegrid'. Si no
        está disponible se usa 'default'.
    figsize : tuple, optional
        Tamaño por defecto de las figuras en pulgadas. Por defecto (12, 5).
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: tuple = (12, 5),
    ) -> None:
        self.figsize = figsize
        # Resolver un estilo válido una sola vez. Se aplica por figura
        # con plt.style.context (no se muta el estado global).
        self.style = style if style in plt.style.available else "default"

    # ── VaR sobre los retornos (estático o dinámico) ───────────────────────────

    def plot_var(
        self,
        returns: pd.Series,
        var: float | pd.Series,
        es: float | pd.Series | None = None,
        title: str = "Retornos y VaR",
        xlabel: str = "Fecha",
        ylabel: str = "Retorno (%)",
        percent: bool = True,
    ) -> plt.Figure:
        """
        Superpone el VaR (y opcionalmente el ES) sobre los retornos y
        marca las excedencias en rojo.

        El argumento var (y es) puede ser:

        - un escalar (float): VaR ESTÁTICO, estimado con una ventana fija,
          se dibuja como una línea horizontal constante.
        - una pd.Series: VaR DINÁMICO, uno por fecha (p.ej. la salida de
          dynamic_var o el var_series_ del Backtester). El tramo sin VaR
          (calentamiento) aparece como un hueco en la línea.

        El VaR es una pérdida positiva, se dibuja en -VaR (umbral en el
        espacio de retornos) y hay violación si retorno < -VaR. Las series
        se alinean por índice (no por posición), así que da igual el orden
        o que sobren/falten fechas.

        Parameters
        ----------
        returns : pd.Series
            Retornos a representar (tramo de test o serie completa).
        var : float or pd.Series
            VaR (pérdida positiva): escalar (estático) o serie (dinámico).
        es : float or pd.Series, optional
            ES para una segunda línea, escalar o serie.
        title : str, optional
            Título.
        xlabel : str, optional
            Etiqueta del eje X. Por defecto "Fecha".
        ylabel : str, optional
            Etiqueta del eje Y. Por defecto "Retorno (%)". Útil cambiarla si la
            serie no son retornos puros, p.ej. P&L al ponderar por una posición.
        percent : bool, optional
            Si True (defecto) los valores se escalan ×100 (retornos en %). Pon
            False para dibujarlos en su escala original (p.ej. P&L ponderado por
            una posición). En ese caso ajusta también ylabel.

        Returns
        -------
        matplotlib.figure.Figure
        """
        factor = 100 if percent else 1

        def _lab(name, x):
            return f"{name} = " + (f"{x:.2%}" if percent else f"{x:.4g}")

        # Normalizar var/es a serie alineada + etiqueta de leyenda.
        if isinstance(var, pd.Series):
            var_aligned = var.reindex(returns.index)
            var_label = "VaR dinámico"
        else:
            var_aligned = pd.Series(var, index=returns.index)
            var_label = _lab("VaR", var)

        es_aligned = None
        es_label = "ES"
        if es is not None:
            if isinstance(es, pd.Series):
                es_aligned = es.reindex(returns.index)
                es_label = "ES dinámico"
            else:
                es_aligned = pd.Series(es, index=returns.index)
                es_label = _lab("ES", es)

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            ax.plot(returns.index, returns.values * factor,
                    color="#455A64", lw=0.9, alpha=0.85, label="Retorno")
            # VaR es pérdida positiva: se dibuja en -VaR sobre los retornos.
            ax.plot(var_aligned.index, -var_aligned.values * factor,
                    color="#F44336", lw=1.8, linestyle="--", label=var_label)

            if es_aligned is not None:
                ax.plot(es_aligned.index, -es_aligned.values * factor,
                        color="#B71C1C", lw=1.2, linestyle=":", label=es_label)

            # Violación: retorno < -VaR. Donde el VaR es NaN, la comparación
            # devuelve False (no cuenta como violación).
            violations = returns.values < -var_aligned.values
            if violations.any():
                viol_idx = returns.index[violations]
                viol_val = returns.values[violations] * factor
                ax.scatter(viol_idx, viol_val, color="#B71C1C", s=22, zorder=5,
                           label=f"Violaciones ({int(violations.sum())})")

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            fig.tight_layout()
        return fig

    # ── Volatilidad condicional GARCH ─────────────────────────────────────────

    def plot_volatility(
        self,
        conditional_vol: pd.Series,
        forecast: np.ndarray | None = None,
        title: str = "Volatilidad condicional GARCH",
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> plt.Figure:
        """
        Gráfico de la volatilidad condicional histórica más pronóstico.

        La entrada (conditional_vol y forecast) se espera en frecuencia diaria
        y escala decimal, como devuelven GARCHParametricVaR.conditional_vol_series_
        y forecast_volatility. Por defecto se anualiza para mostrarla
        (annualize=True, *sqrt(periods_per_year)). Con annualize=False se grafica
        en su escala diaria. La etiqueta del eje se ajusta al modo elegido.

        Parameters
        ----------
        conditional_vol : pd.Series
            Volatilidad condicional diaria (serie histórica), en escala decimal.
        forecast : np.ndarray, optional
            Pronóstico de volatilidad diaria para los próximos pasos, en la
            misma escala que conditional_vol.
        title : str, optional
            Título.
        annualize : bool, optional
            True (defecto) anualiza los valores diarios para mostrarlos
            (*sqrt(periods_per_year)). False los grafica en escala diaria.
        periods_per_year : int, optional
            Periodos por año para anualizar. Por defecto 252.

        Returns
        -------
        matplotlib.figure.Figure
        """
        vol = annualize_volatility(conditional_vol, periods_per_year) if annualize else conditional_vol
        vol_pct = vol * 100

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            ax.plot(vol_pct.index, vol_pct.values,
                    color="#455A64", lw=0.9, label="Vol. condicional")

            if forecast is not None and len(forecast) > 0:
                fc = np.asarray(forecast)
                fc = annualize_volatility(fc, periods_per_year) if annualize else fc
                fc_pct = fc * 100
                # Extender el eje temporal con fechas futuras
                last_date = conditional_vol.index[-1]
                try:
                    freq = pd.infer_freq(conditional_vol.index)
                    future_dates = pd.date_range(start=last_date, periods=len(fc_pct) + 1,
                                                 freq=freq or "B")[1:]
                except Exception:
                    future_dates = range(len(fc_pct))

                ax.plot(future_dates, fc_pct,
                        color="#FF5722", lw=1.8, linestyle="--", label="Pronóstico")

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Volatilidad anualizada (%)" if annualize
                          else "Volatilidad diaria (%)")
            ax.legend(fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            fig.tight_layout()
        return fig

    # ── VaR estático: histograma de retornos simulados ────────────────────────

    def plot_simulation_histogram(
        self,
        simulated_returns: np.ndarray,
        var_value: float,
        es_value: float | None = None,
        title: str = "Distribucion de retornos simulados",
    ) -> plt.Figure:
        """
        VaR ESTÁTICO: histograma de la distribución simulada con líneas
        VaR y ES (umbrales puntuales).

        Parameters
        ----------
        simulated_returns : np.ndarray
            Array de retornos simulados.
        var_value : float
            VaR estimado.
        es_value : float, optional
            ES estimado.
        title : str, optional
            Título.

        Returns
        -------
        matplotlib.figure.Figure
        """
        r_pct = simulated_returns * 100

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            # Un único histograma (mismos bins y normalización para todo): los
            # bins de la cola de pérdidas (r <= -VaR) se recolorean en rojo, así
            # que conservan la altura real de la distribución en vez de
            # re-normalizarse como una subdistribución independiente.
            counts, edges = np.histogram(r_pct, bins=100, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            widths = np.diff(edges)
            tail = centers <= -var_value * 100
            dist_color, tail_color = "#607D8B", "#F44336"
            colors = np.where(tail, tail_color, dist_color)
            ax.bar(centers, counts, width=widths, color=colors, alpha=0.7,
                   align="center", edgecolor="none")

            # Entradas de leyenda mediante proxy artists (la cola no se dibuja
            # con label propio, y bar([], []) no transmite el color al handle).
            handles = [
                Patch(facecolor=dist_color, alpha=0.7, label="Distribución simulada"),
                Patch(facecolor=tail_color, alpha=0.7, label="Cola de pérdidas (≤ -VaR)"),
            ]

            # VaR es pérdida positiva: la línea de retorno va en -VaR.
            handles.append(ax.axvline(-var_value * 100, color="#F44336", lw=2,
                                      linestyle="--", label=f"VaR = {var_value:.2%}"))

            if es_value is not None:
                handles.append(ax.axvline(-es_value * 100, color="#B71C1C", lw=1.5,
                                          linestyle=":", label=f"ES = {es_value:.2%}"))

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Retorno (%)")
            ax.set_ylabel("Densidad")
            ax.legend(handles=handles, fontsize=10)
            fig.tight_layout()
        return fig

    # ── Guardar figura ────────────────────────────────────────────────────────

    @staticmethod
    def save(fig: plt.Figure, filepath: str, dpi: int = 150) -> None:
        """
        Guarda una figura en disco.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        filepath : str
            Ruta de destino (extensión determina el formato: .png, .pdf, .svg).
            La carpeta destino debe existir.
        dpi : int, optional
            Resolución para formatos rasterizados. Por defecto 150.
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
