"""
examples/demo_offline.py
========================
Demo end-to-end de varlib SIN conexión a internet.

Genera una serie de retornos sintética con colas pesadas (t de Student) y
semilla fija: 100 % reproducible y sin red. A partir de ahí, el flujo es
idéntico al de demo.py (que usa datos reales de AAPL); lo único que cambia es
el principio, la obtención de los datos.

Ejecución:
    python examples/demo_offline.py
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from varlib import (
    ParametricVaR,
    HistoricalVaR,
    MonteCarloVaR,
    GARCHParametricVaR,
    GARCHMonteCarloVaR,
    Backtester,
    dynamic_var,
    compare_models_rolling,
    compare_models_expanding,
    VaRPlotter,
    ReportExporter,
    print_summary,
)

CONFIDENCE, HORIZON, N_SIMS, SEED, WINDOW = 0.95, 1, 50_000, 42, 250
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Datos (LO ÚNICO que difiere de demo.py) ─────────────────────────────────────
print("=" * 60)
print("DEMO OFFLINE varlib - Value at Risk (datos sintéticos, sin red)")
print("=" * 60)

LABEL, PREFIX = "SYN", "syn_offline"
MU, SIGMA, NU = 0.0005, 0.012, 5     # media, vol diarias y g.l. de la t (colas pesadas)
fechas  = pd.date_range("2019-01-01", periods=1_250, freq="B")
returns = pd.Series(MU + SIGMA * np.random.default_rng(SEED).standard_t(NU, 1_250),
                    index=fechas, name=LABEL)
train, test = returns.iloc[:WINDOW], returns.iloc[WINDOW:]


# ── VaR/ES con los 5 métodos (ajustados sobre el warm-up) ───────────────────────
print(f"\n[1] Datos: {len(returns)} retornos  |  warm-up={len(train)}  test={len(test)}")
print(f"    media diaria={returns.mean():.4%}  vol diaria={returns.std():.4%}  "
      f"curtosis={returns.kurt():.4f}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    models = {
        "Parametrico":  ParametricVaR(CONFIDENCE, HORIZON).fit(train),
        "Historico":    HistoricalVaR(CONFIDENCE, HORIZON).fit(train),
        "Monte Carlo":  MonteCarloVaR(CONFIDENCE, HORIZON, N_SIMS, SEED).fit(train),
        "GARCH Param.": GARCHParametricVaR(CONFIDENCE, HORIZON).fit(train),
        "GARCH MC":     GARCHMonteCarloVaR(CONFIDENCE, HORIZON, N_SIMS, SEED).fit(train),
    }

print(f"\n[2] VaR/ES (confianza={CONFIDENCE:.0%}, h={HORIZON}d):")
print(f"    {'Metodo':<14}  {'VaR':>9}  {'ES':>9}")
print(f"    {'─'*14}  {'─'*9}  {'─'*9}")
filas = []
for nombre, m in models.items():
    v, e = m.compute_var(), m.compute_es()
    print(f"    {nombre:<14}  {v:>9.4%}  {e:>9.4%}")
    filas.append({"Método": nombre, "VaR": v, "ES": e})
var_summary = pd.DataFrame(filas)


# ── Comparación por backtesting: deslizante y expansiva (compare_models_*) ──────
# Un Backtester por modelo. El mismo dict sirve para las dos ventanas (se copia
# dentro), por lo que no se muta al reutilizarlo.
eval_start = test.index[0]
bts = {
    "Historico":    Backtester(HistoricalVaR(CONFIDENCE)),
    "Param-Normal": Backtester(ParametricVaR(CONFIDENCE, dist="normal")),
    "Param-t":      Backtester(ParametricVaR(CONFIDENCE, dist="t")),
    "Param-skewt":  Backtester(ParametricVaR(CONFIDENCE, dist="skewt")),
    "GARCH-Param":  Backtester(GARCHParametricVaR(CONFIDENCE, dist="normal")),
    "GARCH-MC":     Backtester(GARCHMonteCarloVaR(CONFIDENCE, HORIZON, N_SIMS, SEED)),
}
print(f"\n[3] Comparación por backtesting (Kupiec/Christoffersen) sobre {len(test)} días...")
cmp_rolling   = compare_models_rolling(bts, returns, WINDOW, eval_start=eval_start)
cmp_expanding = compare_models_expanding(bts, train, test)

cmp_cols = ["modelo", "n_violations", "violation_rate", "p_uc", "reject_cc"]
cmp_fmt  = {"violation_rate": lambda x: f"{x:.2%}", "p_uc": lambda x: f"{x:.4f}"}
print("\n  Ventana deslizante:")
print(cmp_rolling[cmp_cols].to_string(index=False, formatters=cmp_fmt))
print("\n  Ventana expansiva:")
print(cmp_expanding[cmp_cols].to_string(index=False, formatters=cmp_fmt))


# ── Capa dynamic_var (serie temporal de VaR con ventana deslizante) ─────────────
var_serie = dynamic_var(ParametricVaR(CONFIDENCE), returns, window=WINDOW)
print(f"\n[4] dynamic_var (Paramétrico, {WINDOW}d): {var_serie.notna().sum()} VaR estimados"
      f"  (último: {var_serie.dropna().iloc[-1]:.4%})")


# ── Parámetros del GARCH (valores del summary del modelo) ───────────────────────
garch_m = models["GARCH Param."]
s_gp = garch_m.summary()
garch_params = {
    "omega":              s_gp["omega"],
    "alpha[1]":           s_gp["alpha"][0],
    "beta[1]":            s_gp["beta"][0],
    "persistencia":       s_gp["persistence"],
    "vol. incondicional": f"{s_gp['unconditional_vol']:.4%}",
    "vol. condicional T": f"{s_gp['conditional_vol_T']:.4%}",
    "AIC":                s_gp["aic"],
    "BIC":                s_gp["bic"],
}
print(f"\n[5] Parámetros GARCH(1,1):")
print_summary(garch_params)
# VaR/ES puntuales del GARCH como Styler (valores numéricos, formato %).
garch_var = pd.DataFrame([{
    "VaR 95% / 1d": garch_m.compute_var(),
    "ES 95% / 1d":  garch_m.compute_es(),
}]).style.format("{:.4%}")


# ── Gráficos ────────────────────────────────────────────────────────────────────
print(f"\n[6] Generando gráficos en {OUTPUT_DIR}/ ...")
plotter = VaRPlotter()

fig_var_retornos = plotter.plot_var(
    train, models["Parametrico"].compute_var(), models["Parametrico"].compute_es(),
    title=f"{LABEL} - Retornos con VaR Paramétrico 95%")

mc = models["GARCH MC"]
if mc.simulated_returns_ is None:
    mc.simulate_paths()
fig_hist_mc = plotter.plot_simulation_histogram(
    mc.simulated_returns_, var_value=mc.compute_var(), es_value=mc.compute_es(),
    title=f"{LABEL} - Distribución simulada GARCH-MC")

# Vol. condicional y pronóstico, ambos diarios; plot_volatility los anualiza.
fig_volatilidad = plotter.plot_volatility(
    garch_m.conditional_vol_series_, forecast=garch_m.forecast_volatility(steps=30),
    title=f"{LABEL} - Volatilidad condicional GARCH(1,1)")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bt_ref = Backtester(ParametricVaR(CONFIDENCE)).run_expanding(train, test)
fig_violaciones = plotter.plot_var(
    test, bt_ref.var_series_, title=f"{LABEL} - Backtesting VaR (ventana expansiva)")

fig_var_deslizante = plotter.plot_var(
    returns, var_serie, title=f"{LABEL} - VaR deslizante {WINDOW}d (serie completa)")

for fig, nombre in [
    (fig_var_retornos,   "retornos_var"),
    (fig_hist_mc,        "sim_hist"),
    (fig_volatilidad,    "volatilidad"),
    (fig_violaciones,    "violations"),
    (fig_var_deslizante, "dynamic_var"),
]:
    plotter.save(fig, OUTPUT_DIR / f"{PREFIX}_fig_{nombre}.png")


# ── Reporte HTML (secciones por tema; listas que agrupan tablas y figuras) ──────
results = {
    "VaR / ES por método": var_summary.style.format({"VaR": "{:.4%}", "ES": "{:.4%}"}),
    "Modelo GARCH(1,1)": [garch_params, garch_var, fig_volatilidad],
    "VaR estático sobre los retornos": fig_var_retornos,
    "Distribución simulada (Monte Carlo)": fig_hist_mc,
    "Comparación backtesting (deslizante)": [cmp_rolling, fig_var_deslizante],
    "Comparación backtesting (expansiva)": [cmp_expanding, fig_violaciones]
}
html_path = OUTPUT_DIR / f"{PREFIX}_report.html"
ReportExporter(author="varlib-demo", ticker=LABEL).export_html(results, filepath=str(html_path))
print(f"\n[7] Reporte HTML: {html_path}")

print(f"\n{'=' * 60}")
print(f"Demo completada. Archivos en: {OUTPUT_DIR}")
print(f"{'=' * 60}\n")
