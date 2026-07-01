"""
examples/estudio_empirico.py -- Análisis empírico del PFG sobre varlib
======================================================================
Estudio definitivo del PFG: compara los 9 modelos de VaR de la librería
sobre cuatro casos (S&P 500 y EUR/USD, en régimen de estrés y de calma)
mediante backtesting con ventana deslizante, y añade -solo sobre el caso
de estrés del S&P 500- un análisis de robustez con ventana expansiva y
las figuras ilustrativas.

Este script SOLO orquesta la librería: toda la lógica reutilizable
(modelos, dynamic_var, compare_models_*, Backtester, VaRPlotter,
ReportExporter) vive en el paquete varlib. Aquí se fija el protocolo del
estudio, se aplica y se vuelca el resultado en un reporte HTML.

Protocolo (idéntico en los cuatro casos)
----------------------------------------
- Niveles de confianza : 95 % y 99 %.
- 9 modelos            : Histórico; Paramétrico Normal/t/skew-t;
                         Monte Carlo normal; GARCH-Paramétrico
                         Normal/t/skew-t; GARCH-Monte Carlo normal.
- Horizonte 1 día, ventana deslizante 250 días, GARCH(1,1) reajustado en
  cada paso, Monte Carlo con N=50 000 y semilla fija.
- Métricas: tasa de violaciones, Kupiec (LR_uc) y Christoffersen
  (LR_ind, LR_cc), con LR_cc como veredicto.

Salida
------
Un reporte HTML autónomo (tablas de backtesting + figuras embebidas) y las
figuras como PNG en OUTPUT_DIR. Las semillas Monte Carlo son fijas, de modo
que el estudio es reproducible.
"""

from pathlib import Path

# Raíz del proyecto (para las rutas de salida).
ROOT = Path(__file__).resolve().parent.parent

from varlib import (
    DataLoader,
    HistoricalVaR,
    ParametricVaR,
    MonteCarloVaR,
    GARCHParametricVaR,
    GARCHMonteCarloVaR,
    Backtester,
    dynamic_var,
    compare_models_rolling,
    compare_models_expanding,
    VaRPlotter,
    ReportExporter,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Constantes del protocolo (fijas para todo el estudio)
# ──────────────────────────────────────────────────────────────────────────────

CONF_LEVELS = (0.95, 0.99)    # niveles de confianza evaluados
WINDOW      = 250             # ventana deslizante del análisis principal
HORIZON     = 1               # horizonte en días
N_SIM       = 50_000          # trayectorias Monte Carlo
SEED        = 42              # semilla de los modelos Monte Carlo

# Cuatro casos: (etiqueta, ticker, inicio_descarga, fin_descarga).
# Cada descarga incluye ~1 año previo de warm-up que la ventana de 250
# días consume antes de emitir el primer VaR, de modo que la evaluación
# arranca por defecto en torno al periodo de interés.
# El fin es el primer día del mes SIGUIENTE al último mes de interés: como
# yfinance trata 'end' como exclusivo, así se incluye ese último mes completo
# (p. ej. fin=2021-07-01 abarca todo junio de 2021).
CASES = [
    ("S&P 500 - estres",  "^GSPC",    "2019-01-01", "2022-07-01"), # 2021-07-01
    ("S&P 500 - calma",   "^GSPC",    "2016-01-01", "2020-01-01"),
    ("EUR/USD - estres",  "EURUSD=X", "2021-06-01", "2023-07-01"),
    ("EUR/USD - calma",   "EURUSD=X", "2017-01-01", "2020-02-01"),
]

# Robustez expansiva y figuras: solo sobre el caso de estrés del S&P 500
# (el primero de la lista). La partición train/test es el split por warm-up
# (train = primeros WINDOW días, test = resto); como ventana = warm-up = 250,
# deslizante y expansiva evalúan el mismo tramo (test).
STRESS_CASE_IDX = 0

OUTPUT_DIR = ROOT / "output" / "estudio"

# Los 9 modelos del protocolo por nivel de confianza: única fuente de verdad de
# la configuración, reutilizada por el análisis de cada caso y por la robustez
# expansiva. compare_models_* clona (deepcopy) cada Backtester en cada
# ejecución, así que reutilizar estos objetos no acarrea efectos colaterales.
BACKTESTERS = {
    conf: {
        "Historico": Backtester(HistoricalVaR(conf, HORIZON)),
        "Param-Normal": Backtester(ParametricVaR(conf, HORIZON, dist="normal")),
        "Param-t": Backtester(ParametricVaR(conf, HORIZON, dist="t")),
        "Param-skewt": Backtester(ParametricVaR(conf, HORIZON, dist="skewt")),
        "MonteCarlo(normal)": Backtester(
            MonteCarloVaR(conf, HORIZON, n_simulations=N_SIM, random_state=SEED)),
        "GARCH-Param-Normal": Backtester(
            GARCHParametricVaR(conf, HORIZON, dist="normal")),
        "GARCH-Param-t": Backtester(
            GARCHParametricVaR(conf, HORIZON, dist="t")),
        "GARCH-Param-skewt": Backtester(
            GARCHParametricVaR(conf, HORIZON, dist="skewt")),
        "GARCH-MC(normal)": Backtester(
            GARCHMonteCarloVaR(conf, HORIZON, n_simulations=N_SIM, random_state=SEED)),
    }
    for conf in CONF_LEVELS
}


# ──────────────────────────────────────────────────────────────────────────────
#  Funciones
# ──────────────────────────────────────────────────────────────────────────────

def analyze_case(label: str, ticker: str, start: str, end: str) -> dict:
    """
    Ejecuta el protocolo completo de un caso y DEVUELVE sus resultados:
    descarga los datos, calcula los log-retornos y corre el
    backtesting con ventana DESLIZANTE de 250 días sobre los 9 modelos,
    con una tabla por nivel de confianza. La deslizante arranca por defecto,
    el warm-up lo consume la propia ventana.

    Separa la preparación de datos del análisis: la serie de retornos se
    devuelve para reutilizarla después (robustez y figuras del caso de estrés).

    Returns
    -------
    dict
        Claves: 'label', 'ticker', 'returns' (log-retornos),
        'tables' ({conf: tabla de backtesting deslizante}, una por nivel).
    """
    loader = DataLoader(ticker, start=start, end=end).download()
    returns = loader.get_log_returns()

    # Chequeo rápido de la descarga: rango, filas, NAs eliminados y % válido.
    info = loader.summary()
    print(f"[{label}] {info['ticker']} {info['start']} -> {info['end']} | "
          f"filas={info['n_raw']} NAs={info['n_nan_dropped']} "
          f"({info['pct_complete']:.1%} valido) | retornos={info['n_returns']}")

    tables = {
        conf: compare_models_rolling(BACKTESTERS[conf], returns, WINDOW)
        for conf in CONF_LEVELS
    }
    return {"label": label, "ticker": ticker, "returns": returns, "tables": tables}


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Los cuatro casos: backtesting deslizante. Una sección por caso y nivel
#    de confianza (una tabla para el 95 % y otra para el 99 %).
results = []
for label, ticker, start, end in CASES:
    results.append(analyze_case(label, ticker, start, end))
report = {
    f"Backtesting deslizante 250d - {r['label']} ({conf:.0%})": r["tables"][conf]
    for r in results
    for conf in CONF_LEVELS
}

# 2) Robustez y figuras: solo sobre el caso de estrés del S&P 500,
#    reutilizando sus retornos. Split por warm-up = WINDOW.
stress_case_sp500 = results[STRESS_CASE_IDX]
returns = stress_case_sp500["returns"]
train   = returns.iloc[:WINDOW]
test    = returns.iloc[WINDOW:]

# Misma comparación con ventana EXPANSIVA (warm-up = train, evalúa test),
# también una tabla por nivel. Frente a la deslizante del mismo caso
# evidencia la mayor inercia de la expansiva ante el cambio de régimen.
for conf in CONF_LEVELS:
    expanding_table = compare_models_expanding(BACKTESTERS[conf], train, test)
    report[f"Robustez S&P 500 estres ({conf:.0%}) - deslizante vs expansiva"] = [
        stress_case_sp500["tables"][conf], expanding_table,
    ]

# Figuras (Paramétrico Normal 99 %: subestima la cola en la crisis).
plotter = VaRPlotter()
failing_model = ParametricVaR(0.99, HORIZON, dist="normal")
var_rolling   = dynamic_var(failing_model, returns, window=WINDOW)
var_expanding = dynamic_var(failing_model, returns, window=WINDOW, expanding=True)

fig_failing = plotter.plot_var(
    returns, var_rolling,
    title=f"{stress_case_sp500['label']}: VaR Paramétrico Normal 99% (deslizante 250d)",
)

# VaR GARCH skew-t 99% (deslizante 250d): reacciona a la volatilidad y captura
# las colas, frente al Paramétrico Normal estático que las subestima.
var_garch_skewt = dynamic_var(
    GARCHParametricVaR(0.99, HORIZON, dist="skewt"), returns, window=WINDOW)
fig_garch_skewt = plotter.plot_var(
    returns, var_garch_skewt,
    title=f"{stress_case_sp500['label']}: VaR GARCH Paramétrico skew-t 99% (deslizante 250d)",
)

fig_rolling_vs_expanding = plotter.plot_var(
    returns, var_rolling,
    title=f"{stress_case_sp500['label']}: VaR Normal 99% deslizante vs expansivo",
)
# Añadimos la línea expansiva sobre los mismos ejes (VaR = pérdida positiva,
# se dibuja en -VaR y en %, igual que hace plot_var con percent=True).
ax = fig_rolling_vs_expanding.axes[0]
ax.plot(var_expanding.index, -var_expanding.values * 100,
        color="#1565C0", lw=1.8, linestyle="-.", label="VaR expansivo")
ax.axvline(test.index[0], color="#000000", lw=1.0, linestyle=":", alpha=0.6,
            label="Inicio evaluación")
ax.legend(fontsize=9)

# Histograma de la distribución simulada del GARCH-MC (ajustado a la
# ventana previa a la crisis, = train), con VaR y ES marcados.
mc = GARCHMonteCarloVaR(0.99, HORIZON, n_simulations=N_SIM, random_state=SEED)
mc.fit(train)
mc.simulate_paths()
fig_histogram = plotter.plot_simulation_histogram(
    mc.simulated_returns_,
    var_value=mc.compute_var(),
    es_value=mc.compute_es(),
    title=f"S&P 500 - Distribución normal simulada GARCH-MC (99%)",
)

# Volatilidad condicional GARCH(1,1) con pronóstico, sobre toda la serie del
# caso de estrés: muestra el repunte de volatilidad durante la crisis.
garch_vol = GARCHParametricVaR(0.99, HORIZON, dist="normal").fit(returns)
fig_volatility = plotter.plot_volatility(
    garch_vol.conditional_vol_series_,
    forecast=garch_vol.forecast_volatility(steps=30),
    title=f"S&P 500 - Volatilidad condicional GARCH(1,1) y pronóstico",
)

report["Figuras - S&P 500 estres"] = [
    fig_failing, fig_garch_skewt, fig_rolling_vs_expanding, fig_histogram, fig_volatility,
]

# 3) Reporte HTML autónomo (tablas + figuras embebidas).
ReportExporter(author="PFG", ticker=stress_case_sp500["ticker"]).export_html(
    report, filepath=str(OUTPUT_DIR / "reporte_estudio.html"))

# Figuras también como PNG sueltos (save() cierra cada figura: tras exportar).
for name, fig in (("fig_modelo_fallo", fig_failing),
                    ("fig_garch_skewt", fig_garch_skewt),
                    ("fig_deslizante_vs_expansiva", fig_rolling_vs_expanding),
                    ("fig_histograma_garch_mc", fig_histogram),
                    ("fig_volatilidad_garch", fig_volatility)):
    plotter.save(fig, OUTPUT_DIR / f"{name}.png")


