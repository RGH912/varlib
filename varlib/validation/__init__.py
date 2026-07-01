"""
Subpaquete de validacion y backtesting.
"""

from varlib.validation.backtesting import Backtester
from varlib.validation.dynamic import dynamic_var
from varlib.validation.comparison import (
    compare_models_expanding,
    compare_models_rolling,
)

__all__ = [
    "Backtester",
    "dynamic_var",
    "compare_models_expanding",
    "compare_models_rolling",
]
