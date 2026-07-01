"""
varlib.reporting.console
========================
Formateo legible de resúmenes (summary) para mostrar por consola.

Mantiene la presentación separada de los modelos: estos solo devuelven
datos estructurados (summary() -> dict, o DataFrame en el Backtester) y
aquí se formatean de forma uniforme y alineada. Así se evita repetir
bucles de print en los scripts y demos.

Uso típico
----------
from varlib.reporting.console import print_summary
print_summary(model, "ParametricVaR 95%")     # objeto con .summary()
print_summary(loader.summary(), "Activo")      # dict directo
"""

from typing import Any

import pandas as pd


# Claves cuyos valores son tasas/retornos o volatilidades: se muestran como
# porcentaje en lugar de decimal. El resto (coeficientes y formas
# adimensionales: nu, lambda, alpha, beta, persistence, skew, kurt...) se
# dejan en decimal. 'confidence' se deja fuera a propósito.
_PERCENT_KEYS = frozenset({
    "var", "es", "mu", "sigma", "annualized_vol",
    "mean_return", "std_return", "min_return", "max_return",
    "pct_complete",
    "unconditional_vol", "conditional_vol_T",
    "annualized_cond_vol", "annualized_uncond_vol",
    "sim_mean", "sim_std",
})

# Claves-porcentaje que se muestran con 2 decimales en vez de 4 (valores
# cercanos a 100%, donde 4 decimales sobran).
_PERCENT_2DP_KEYS = frozenset({"pct_complete"})


def _fmt(v: Any) -> str:
    """
    Formatea un valor escalar (o lista) para una celda de resumen.
    """
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != 0.0 and abs(v) < 1e-4:
            return f"{v:.4e}"
        return f"{v:.6f}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def format_summary(data: Any, title: str | None = None) -> str:
    """
    Devuelve una representación alineada de un resumen.

    Parameters
    ----------
    data : objeto con .summary(), dict, pd.Series o pd.DataFrame
        Si el objeto expone un método summary() (modelos, DataLoader,
        Backtester) se llama automáticamente. Un dict o Series se muestra
        como 'clave : valor' alineado, un DataFrame con su to_string().
    title : str, optional
        Título opcional, subrayado con guiones.

    Returns
    -------
    str
        Texto listo para imprimir (no incluye salto final).
    """
    if hasattr(data, "summary") and callable(data.summary):
        data = data.summary()

    parts: list[str] = []
    if title:
        parts.append(title)
        parts.append("-" * len(title))

    if isinstance(data, pd.DataFrame):
        if len(data) == 1:
            # Una sola fila (p.ej. Backtester.summary()): se muestra como
            # pares clave:valor, igual que un dict o Series.
            data = data.iloc[0]
        else:
            parts.append(data.to_string(index=False))
            return "\n".join(parts)

    if isinstance(data, pd.Series):
        pairs = list(data.items())
    elif isinstance(data, dict):
        pairs = list(data.items())
    else:
        parts.append(str(data))
        return "\n".join(parts)

    width = max((len(str(k)) for k, _ in pairs), default=0)
    for k, v in pairs:
        if k in _PERCENT_KEYS and isinstance(v, float):
            decimals = 2 if k in _PERCENT_2DP_KEYS else 4
            cell = f"{v:.{decimals}%}"
        else:
            cell = _fmt(v)
        parts.append(f"  {str(k):<{width}} : {cell}")
    return "\n".join(parts)


def print_summary(data: Any, title: str | None = None) -> None:
    """
    Imprime el resumen formateado por format_summary.

    Parameters
    ----------
    data : objeto con .summary(), dict, pd.Series o pd.DataFrame
    title : str, optional
        Título opcional.
    """
    print(format_summary(data, title))
