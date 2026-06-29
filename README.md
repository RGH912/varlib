# varlib

Librería de estimación del **Value at Risk (VaR)** y el **Expected Shortfall (ES)**
con backtesting estadístico, desarrollada como Proyecto de Fin de Grado.

Implementa varios modelos de VaR con una API orientada a objetos, validación
mediante contrastes (Kupiec, Christoffersen) y exportación de resultados a
gráficos y reportes HTML autónomos.

## Características

- **Modelos de VaR/ES**: histórico, paramétrico (Normal, t de Student, skew-t),
  Monte Carlo y sus variantes con volatilidad condicional **GARCH(p, q)**.
- **Estimación por máxima verosimilitud** de los parámetros (vía `arch`).
- **Backtesting**: tests de Kupiec (cobertura incondicional) y Christoffersen
  (independencia y cobertura condicional), con ventana deslizante o expansiva.
- **VaR dinámico** sobre ventana móvil o expansiva.
- **Reporting**: resúmenes por consola, gráficos (`matplotlib`) y reportes HTML
  auto-contenidos (figuras embebidas en base64).

## Requisitos

- Python ≥ 3.10
- Dependencias (se instalan solas): `numpy`, `pandas`, `scipy`, `matplotlib`,
  `arch`, `yfinance`.

## Instalación

```bash
# (opcional) crear y activar un entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# instalar la librería en modo editable
pip install -e .

# con las dependencias de desarrollo (tests)
pip install -e ".[dev]"
```

## Ejemplo sencillo de uso

```python
from varlib import (
    DataLoader, HistoricalVaR, ParametricVaR, GARCHParametricVaR,
    Backtester, compare_models_rolling, dynamic_var,
    VaRPlotter, ReportExporter, print_summary,
)

# 1. Descarga de datos, rendimientos logarítmicos y resumen del activo
loader = DataLoader("^GSPC", start="2020-01-01", end="2024-12-31")
loader.download()
returns = loader.get_log_returns()
print_summary(loader)

# 2. Estimación del VaR y el ES con un modelo GARCH(2,1) (confianza del 99 %)
modelo = GARCHParametricVaR(confidence=0.99, p=2, q=1, dist="skewt").fit(returns)
var = modelo.compute_var()
es = modelo.compute_es()
print(f"VaR 99 %: {var:.2%}")
print(f"ES 99 %: {es:.2%}")

# 3. Comparación de varios modelos por backtesting con ventana deslizante
modelos = {
    "Historico":   Backtester(HistoricalVaR(0.95)),
    "Param-skewt": Backtester(ParametricVaR(0.95, dist="skewt")),
    "GARCH-skewt": Backtester(GARCHParametricVaR(0.95, dist="skewt")),
}
comparacion = compare_models_rolling(modelos, returns, window=250)
print(comparacion)

# 4. Visualización del VaR dinámico y guardado de la figura en disco
var_series = dynamic_var(ParametricVaR(0.95), returns, window=250)
plotter = VaRPlotter()
fig = plotter.plot_var(returns, var_series)
plotter.save(fig, "var_sp500.png")

# 5. Montaje de un reporte HTML con la comparación y el gráfico
reporte = ReportExporter(author="varlib", ticker="S&P 500")
reporte.export_html(
    {
        "Comparación de modelos (ventana deslizante)": comparacion,
        "VaR dinámico sobre los retornos": fig,
    },
    filepath="reporte_var_sp500.html",
)
```

> Convención de `arch`: en GARCH(p, q), **p** es el orden ARCH (α) y **q** el
> orden GARCH (β), a la inversa de Bollerslev (1986). En GARCH(1,1) la
> distinción es irrelevante.

## Estructura del proyecto

```
libreria_var_garch/
├── varlib/                 # Paquete principal
│   ├── data/               # DataLoader (descarga y preprocesamiento)
│   ├── models/             # Modelos VaR (histórico, paramétrico, MC, GARCH)
│   ├── validation/         # Backtesting, VaR dinámico y comparación
│   └── reporting/          # Consola, gráficos y reportes HTML
├── tests/                  # Suite de tests (pytest), un archivo por módulo
├── examples/               # Scripts de demostración y estudio empírico
├── main.py                 # Demo principal
└── pyproject.toml          # Configuración del proyecto y dependencias
```

## Tests

```bash
pytest                  # toda la suite
pytest -m "not slow"    # omite las pruebas lentas (GARCH por MLE, Monte Carlo)
pytest --cov            # con informe de cobertura
```

## Autor

Ricardo García — Proyecto de Fin de Grado.
