"""
varlib -- Libreria de estimacion del Value at Risk (VaR)
=========================================================
Implementa multiples metodos de VaR con API orientada a objetos,
backtesting estadistico y exportacion de reportes.

Modulos disponibles
-------------------
- varlib.data        : adquisicion y preprocesamiento de datos
- varlib.models      : modelos VaR (parametrico, historico, Monte Carlo, GARCH)
- varlib.validation  : backtesting (Kupiec, Christoffersen)
- varlib.reporting   : graficos y exportacion de reportes
"""

from varlib.data.loader import DataLoader

from varlib.models.parametric       import ParametricVaR
from varlib.models.historical       import HistoricalVaR
from varlib.models.montecarlo       import MonteCarloVaR
from varlib.models.garch_parametric import GARCHParametricVaR
from varlib.models.garch_montecarlo import GARCHMonteCarloVaR

from varlib.validation.backtesting  import Backtester
from varlib.validation.dynamic       import dynamic_var
from varlib.validation.comparison    import (
    compare_models_expanding,
    compare_models_rolling,
)

from varlib.reporting.plots   import VaRPlotter
from varlib.reporting.reports import ReportExporter
from varlib.reporting.console import format_summary, print_summary

__all__ = [
    # datos
    "DataLoader",
    # modelos
    "ParametricVaR",
    "HistoricalVaR",
    "MonteCarloVaR",
    "GARCHParametricVaR",
    "GARCHMonteCarloVaR",
    # validacion
    "Backtester",
    "dynamic_var",
    "compare_models_expanding",
    "compare_models_rolling",
    # reporting
    "VaRPlotter",
    "ReportExporter",
    "format_summary",
    "print_summary",
]

__version__ = "0.1.0"
