"""
demo_completo.py -- Recorrido completo de varlib, módulo a módulo
=================================================================
Ejecuta un ejemplo de cada componente de la librería (datos, modelos,
backtesting, comparación y reporting) sobre datos reales de AAPL.
Cada bloque puede ejecutarse de forma independiente.
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
#  BLOQUE 1 - DataLoader
#  varlib/data/loader.py
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  BLOQUE 1 - DataLoader")
print("=" * 60)

from varlib.data.loader import DataLoader
from varlib.reporting.console import print_summary

loader = DataLoader("AAPL", start="2020-01-01", end="2024-12-31")
print(loader)           # repr sin descargar

loader.download()
print(loader)           # repr con datos

returns = loader.get_log_returns()

# Warm-up fijo de 250d (≈1 año) para ajustar.
WARMUP = 250
train, test = loader.split(WARMUP)

info = loader.summary()
print_summary(info, "Resumen del activo")
print(f"\nWarm-up {WARMUP}d / evaluación  : train={len(train)} días  |  test={len(test)} días")

# ──────────────────────────────────────────────────────────────────────────────
#  BLOQUE 2 - BaseVaR + ParametricVaR
#  varlib/models/base.py + parametric.py
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  BLOQUE 2 - ParametricVaR")
print("=" * 60)

from varlib.models.parametric import ParametricVaR

# ── 2a. Uso básico ────────────────────────────────────────────────────────────
model_95 = ParametricVaR(confidence=0.95, horizon=1)
print(f"\nAntes de fit: {model_95}")

model_95.fit(returns)
print(f"Después: {model_95}")

var_95  = model_95.compute_var()
es_95 = model_95.compute_es()
print(f"\nParámetros estimados:")
print(f"mu diario      = {model_95.mu_:.6f}  ({model_95.mu_:.4%})")
print(f"sigma diario      = {model_95.sigma_:.6f}  ({model_95.sigma_:.4%})")
print(f"z (alpha=5%)      = {model_95.z_:.4f}")
print(f"Vol anual   = {model_95.summary()['annualized_vol']:.4%}")

print(f"\nResultados (horizonte 1 día):")
print(f"VaR  95%  = {var_95:.4%}")
print(f"ES 95%  = {es_95:.4%}")

# ── 2b. summary() completo ────────────────────────────────────────────────────
print()
print_summary(model_95, "summary() del modelo 95% / 1 dia")

# ── 2c. Distintos niveles de confianza ────────────────────────────────────────
print(f"\nComparativa de niveles de confianza (horizonte 1 día):")
print(f"  {'Confianza':>10}  {'VaR':>10}  {'ES':>10}")
print(f"  {'─'*10}  {'─'*10}  {'─'*10}")
for conf in [0.90, 0.95, 0.99]:
    m = ParametricVaR(confidence=conf).fit(returns)
    print(f"  {conf:>10.0%}  {m.compute_var():>10.4%}  {m.compute_es():>10.4%}")

# ── 2d. Distintos horizontes ──────────────────────────────────────────────────
print(f"\nVaR Paramétrico a distintos horizontes (dist=normal, confianza 95%):")
print(f"  {'Horizonte':>10}  {'VaR':>10}  {'ES':>10}")
print(f"  {'─'*10}  {'─'*10}  {'─'*10}")
for h in [1, 5, 10, 20]:
    m = ParametricVaR(confidence=0.95, horizon=h).fit(returns)
    print(f"  {h:>9}d  {m.compute_var():>10.4%}  {m.compute_es():>10.4%}")

# ── 2e. Distribuciones de los retornos (Normal vs t vs skew-t) ────────────────
print(f"\nComparativa de distribuciones del paramétrico (95%, 1d):")
print(f"  {'Distribución':<12}  {'nu':>6}  {'lambda':>7}  {'VaR':>10}  {'ES':>10}")
print(f"  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*10}  {'─'*10}")
for d in ["normal", "t", "skewt"]:
    m = ParametricVaR(confidence=0.95, dist=d).fit(returns)
    nu_str  = f"{m.nu_:.2f}"     if m.nu_     is not None else "  -"
    lam_str = f"{m.lambda_:.3f}" if m.lambda_ is not None else "   -"
    print(f"  {d:<12}  {nu_str:>6}  {lam_str:>7}  "
          f"{m.compute_var():>10.4%}  {m.compute_es():>10.4%}")

# ── 2f. Guards de error ───────────────────────────────────────────────────────
print(f"\nTest de guards de error:")
try:
    ParametricVaR().compute_var()   # sin fit
except RuntimeError as e:
    print(f"RuntimeError (esperado) OK  ->  {e}")

try:
    ParametricVaR().fit(pd.Series([0.01, 0.02]))   # muy pocos datos
except ValueError as e:
    print(f"ValueError   (esperado) OK  ->  {e}")

try:
    ParametricVaR(dist="xyz")        # distribución no reconocida
except ValueError as e:
    print(f"ValueError   (esperado) OK  ->  {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  BLOQUE 3 - HistoricalVaR
#  varlib/models/historical.py
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  BLOQUE 3 - HistoricalVaR")
print("=" * 60)

from varlib.models.historical import HistoricalVaR

# ── 3a. Uso básico ────────────────────────────────────────────────────────────
model_h = HistoricalVaR(confidence=0.95).fit(returns)
print(f"\nHistórico clásico (percentil empírico, 95%, 1d):")
print(f"VaR = {model_h.compute_var():.4%}")
print(f"ES = {model_h.compute_es():.4%}")

# ── 3b. Comparativa de niveles de confianza ───────────────────────────────────
print(f"\n  {'Confianza':>10}  {'VaR 1d':>10}  {'ES 1d':>10}  "
      f"{'VaR 10d':>10}  {'ES 10d':>10}")
print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
for conf in [0.90, 0.95, 0.99]:
    m1  = HistoricalVaR(confidence=conf, horizon=1).fit(returns)
    m10 = HistoricalVaR(confidence=conf, horizon=10).fit(returns)
    print(f"  {conf:>10.0%}  {m1.compute_var():>10.4%}  {m1.compute_es():>10.4%}"
          f"  {m10.compute_var():>10.4%}  {m10.compute_es():>10.4%}")

# ── 3c. summary() ─────────────────────────────────────────────────────────────
print()
print_summary(model_h, "summary() Historico 95% / 1d")

# ── 3d. Guard de error ────────────────────────────────────────────────────────
print(f"\nTest de guards:")
try:
    HistoricalVaR().compute_var()
except RuntimeError as e:
    print(f"RuntimeError (esperado) OK  ->  {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  BLOQUE 4 - MonteCarloVaR
#  Paso 6 del plan: varlib/models/montecarlo.py
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  BLOQUE 4 - MonteCarloVaR")
print("=" * 60)

from varlib.models.montecarlo import MonteCarloVaR

# ── 4a. Uso básico ────────────────────────────────────────────────────────────
model_mc = MonteCarloVaR(
    confidence=0.95, horizon=1, n_simulations=50_000, random_state=42
).fit(returns)

print(f"\n{model_mc}")
print(f"mu={model_mc.mu_:.4%}  sigma={model_mc.sigma_:.4%}")

var_mc  = model_mc.compute_var()   # llama a simulate() automáticamente
es_mc = model_mc.compute_es()
print(f"\nResultados (50 000 sims, horizonte 1 día, conf. 95%):")
print(f"VaR  = {var_mc:.4%}")
print(f"ES = {es_mc:.4%}")

# ── 4b. Convergencia según N de simulaciones ──────────────────────────────────
print(f"\nConvergencia según N de simulaciones (conf. 95%, h=1):")
print(f"  {'N sims':>10}  {'VaR':>10}  {'ES':>10}")
print(f"  {'─'*10}  {'─'*10}  {'─'*10}")
for n in [1_000, 5_000, 10_000, 50_000, 100_000]:
    m = MonteCarloVaR(0.95, 1, n_simulations=n, random_state=42).fit(returns)
    print(f"  {n:>10,}  {m.compute_var():>10.4%}  {m.compute_es():>10.4%}")

# ── 4c. Comparativa de niveles de confianza y horizontes ──────────────────────
print(f"\n  {'Confianza':>10}  {'VaR 1d':>10}  {'ES 1d':>10}  "
      f"{'VaR 10d':>10}  {'ES 10d':>10}")
print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
for conf in [0.90, 0.95, 0.99]:
    m1  = MonteCarloVaR(conf, 1,  50_000, 42).fit(returns)
    m10 = MonteCarloVaR(conf, 10, 50_000, 42).fit(returns)
    print(f"  {conf:>10.0%}  {m1.compute_var():>10.4%}  {m1.compute_es():>10.4%}"
          f"  {m10.compute_var():>10.4%}  {m10.compute_es():>10.4%}")

# ── 4d. Guards de error ───────────────────────────────────────────────────────
print(f"\nTest de guards:")
try:
    MonteCarloVaR(n_simulations=0)
except ValueError as e:
    print(f"ValueError   (esperado) OK  ->  {e}")
try:
    MonteCarloVaR().compute_var()
except RuntimeError as e:
    print(f"RuntimeError (esperado) OK  ->  {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  BLOQUE 5 - GARCHParametricVaR
#  varlib/models/garch_parametric.py
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  BLOQUE 5 - GARCHParametricVaR")
print("=" * 60)

from varlib.models.garch_parametric import GARCHParametricVaR
from varlib.models.base import annualize_volatility

# ── 5a. Ajuste GARCH(1,1) Normal ─────────────────────────────────────────────
model_gp = GARCHParametricVaR(
    confidence=0.95, horizon=1, p=1, q=1, dist="normal"
).fit(returns)

print(f"\n{model_gp}\n")
print_summary(model_gp, "Parámetros GARCH(1,1)")
print(f"Persistencia {model_gp.persistence_:.4f}: "
      f"{'[!] no estacionario' if model_gp.persistence_ >= 1 else 'OK estacionario'}")

var_gp  = model_gp.compute_var()
es_gp = model_gp.compute_es()
print(f"\nVaR  95% / 1d = {var_gp:.4%}")
print(f"ES 95% / 1d = {es_gp:.4%}")

# ── 5b. Pronóstico de volatilidad ─────────────────────────────────────────────
STEPS = 30
vol_forecast = model_gp.forecast_volatility(steps=STEPS)   # diaria (decimal)
print(f"\nPronóstico de volatilidad anualizada ({STEPS} días):")
print(f"  {'Día':>5}  {'Vol. anual.':>12}")
print(f"  {'─'*5}  {'─'*12}")
for dia in [1, 2, 3, 5, 10, 20, 30]:
    v = annualize_volatility(vol_forecast[dia - 1])   # anualizar para mostrar
    print(f"  {dia:>5}  {v:>12.4%}")
print(f"Vol. incondicional ~ {model_gp.summary()['annualized_uncond_vol']:.4%}")

# ── 5c. Distribución t-Student ────────────────────────────────────────────────
model_gp_t = GARCHParametricVaR(confidence=0.95, dist="t").fit(returns)

print(f"\n{model_gp_t}\n")
print_summary(model_gp_t, "Parámetros GARCH(1,1) con distribución t-Student")
print(f"\nnu (grados de libertad) = {model_gp_t.nu_:.4f}")
print(f"VaR  95% / 1d (t) = {model_gp_t.compute_var():.4%}")
print(f"ES 95% / 1d (t) = {model_gp_t.compute_es():.4%}")
print(f"AIC = {model_gp_t.garch_result_.aic:.2f}")
print(f"BIC = {model_gp_t.garch_result_.bic:.2f}")

# ── 5d. Comparativa de distribuciones ────────────────────────────────────────
print(f"\nComparativa de distribuciones (GARCH(1,1)):")
print(f"  {'Distribución':<12}  {'alpha+beta':>10}  {'VaR 95%':>10}  "
      f"{'ES 95%':>10}  {'AIC':>10}")
print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
for d in ["normal", "t", "skewt"]:
    m = GARCHParametricVaR(confidence=0.95, dist=d).fit(returns)
    print(f"  {d:<12}  {m.persistence_:>8.6f}  {m.compute_var():>10.4%}  "
          f"{m.compute_es():>10.4%}  {m.garch_result_.aic:>10.2f}")

# ── 5e. Comparativa de horizontes ─────────────────────────────────────────────
print(f"\nVaR GARCH a distintos horizontes (dist=normal, confidence=95%):")
print(f"  {'Horizonte':>10}  {'VaR':>10}  {'ES':>10}")
print(f"  {'─'*10}  {'─'*10}  {'─'*10}")
for h in [1, 5, 10, 20]:
    m = GARCHParametricVaR(confidence=0.95, horizon=h).fit(returns)
    print(f"  {h:>9}d  {m.compute_var():>10.4%}  {m.compute_es():>10.4%}")

# ── 5f. Comparativa global de los CUATRO métodos ──────────────────────────────###
print(f"\n  {'─'*62}")
print(f"COMPARATIVA GLOBAL  (conf. 95%, horizonte 1 día, serie completa AAPL)")
print(f"  {'─'*62}")
print(f"  {'Método':<30}  {'VaR':>10}  {'ES':>10}")
print(f"  {'─'*30}  {'─'*10}  {'─'*10}")
for nombre, m in [
    ("Paramétrico (Normal)", ParametricVaR(0.95).fit(returns)),
    ("Histórico (uniforme)", HistoricalVaR(0.95).fit(returns)),
    ("Monte Carlo (50k, Normal)", MonteCarloVaR(0.95, 1, 50_000, 42).fit(returns)),
    ("GARCH(1,1) Normal", GARCHParametricVaR(0.95).fit(returns)),
    ("GARCH(1,1) t-Student", GARCHParametricVaR(0.95, dist="t").fit(returns)),
    ("GARCH(1,1) Skew-t", GARCHParametricVaR(0.95, dist="skewt").fit(returns))
]:
    print(f"  {nombre:<30}  {m.compute_var():>10.4%}  {m.compute_es():>10.4%}")

# ── 5g. Guards de error ───────────────────────────────────────────────────────
print(f"\nTest de guards:")
try:
    GARCHParametricVaR(p=0)
except ValueError as e:
    print(f"ValueError (esperado) OK  ->  {e}")
try:
    GARCHParametricVaR(dist="xyz")
except ValueError as e:
    print(f"ValueError (esperado) OK  ->  {e}")
try:
    GARCHParametricVaR().compute_var()
except RuntimeError as e:
    print(f"RuntimeError (esperado) OK  ->  {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  BLOQUE 6 - GARCHMonteCarloVaR
#  varlib/models/garch_montecarlo.py
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  BLOQUE 6 - GARCHMonteCarloVaR")
print("=" * 60)

from varlib.models.garch_montecarlo import GARCHMonteCarloVaR

# ── 6a. Uso básico ────────────────────────────────────────────────────────────
model_gmc = GARCHMonteCarloVaR(
    confidence=0.95, horizon=1, n_simulations=50_000, random_state=42
).fit(returns)

print(f"\n{model_gmc}\n")

print_summary(model_gmc, "Parámetros GARCH MonteCarlo")

# compute_var llama a simulate_paths() automáticamente
var_gmc  = model_gmc.compute_var()
es_gmc = model_gmc.compute_es()
print(f"\nResultados (50k sims, h=1, conf. 95%):")
print(f"VaR  = {var_gmc:.4%}")
print(f"ES = {es_gmc:.4%}")

# ── 6b. Estadísticos de la distribución simulada ──────────────────────────────
s = model_gmc.simulated_returns_
print(f"\nDistribución de los 50k retornos simulados (h=1):")
print(f"Media    = {s.mean():.4%}")
print(f"Desv.típ = {s.std():.4%}")
print(f"Mínimo   = {s.min():.4%}   Máximo = {s.max():.4%}")
print(f"Percentil  1% = {np.percentile(s,  1):.4%}")
print(f"Percentil  5% = {np.percentile(s,  5):.4%}")
print(f"Percentil 10% = {np.percentile(s, 10):.4%}")

# ── 6c. GARCH-MC vs GARCH-Paramétrico: deben ser muy similares ───────────────
print(f"\nGARCHMonteCarlo vs GARCHParametric (h=1, conf=95%):")
print(f"  {'Modelo':<30}  {'VaR':>10}  {'ES':>10}")
print(f"  {'─'*30}  {'─'*10}  {'─'*10}")
for nombre, m in [
    ("GARCH Paramétrico",      GARCHParametricVaR(0.95).fit(returns)),
    ("GARCH-MC 10k sims",      GARCHMonteCarloVaR(0.95, 1, 10_000, 42).fit(returns)),
    ("GARCH-MC 50k sims",      GARCHMonteCarloVaR(0.95, 1, 50_000, 42).fit(returns)),
    ("GARCH-MC 100k sims",     GARCHMonteCarloVaR(0.95, 1, 100_000, 42).fit(returns)),
]:
    print(f"  {nombre:<30}  {m.compute_var():>10.4%}  {m.compute_es():>10.4%}")

# ── 6d. Ventaja en horizontes largos: GARCH vs raíz del tiempo ───────────────
print(f"\nEscalado temporal: GARCH-MC vs regla sqrt(t) (conf. 95%):")
print(f"  {'Horizonte':>10}  {'GARCH-MC':>10}  {'GARCH-Param':>12}  "
      f"{'Param. sqrt(t)':>12}  {'Hist. sqrt(t)':>10}")
print(f"  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*10}")
for h in [1, 5, 10, 20]:
    m_gmc = GARCHMonteCarloVaR(0.95, h, 50_000, 42).fit(returns)
    m_gp  = GARCHParametricVaR(0.95, h).fit(returns)
    m_par = ParametricVaR(0.95, h).fit(returns)
    m_his = HistoricalVaR(0.95, h).fit(returns)
    print(f"  {h:>9}d  {m_gmc.compute_var():>10.4%}  "
          f"{m_gp.compute_var():>12.4%}  "
          f"{m_par.compute_var():>12.4%}  "
          f"{m_his.compute_var():>10.4%}")

# ── 6f. Guards de error ───────────────────────────────────────────────────────
print(f"\nTest de guards:")
try:
    GARCHMonteCarloVaR(n_simulations=-1)
except ValueError as e:
    print(f"ValueError   (esperado) OK  ->  {e}")
try:
    GARCHMonteCarloVaR().simulate_paths()
except RuntimeError as e:
    print(f"RuntimeError (esperado) OK  ->  {e}")

# ── 6g. Comparativa FINAL: métodos y distribuciones ──────────────────────────
print(f"\n  {'='*60}")
print(f"COMPARATIVA FINAL - MÉTODOS Y DISTRIBUCIONES  (conf. 95%, h=1, AAPL)")
print(f"  {'='*60}")
print(f"  {'Método':<34}  {'VaR':>10}  {'ES':>10}")
print(f"  {'─'*34}  {'─'*10}  {'─'*10}")
# Solo Paramétrico y GARCH-Paramétrico admiten dist (normal/t/skewt);
# Histórico es empírico y los Monte Carlo son normales por diseño.
for nombre, m in [
    ("Paramétrico Normal",  ParametricVaR(0.95, dist="normal").fit(returns)),
    ("Paramétrico t",                    ParametricVaR(0.95, dist="t").fit(returns)),
    ("Paramétrico skew-t",               ParametricVaR(0.95, dist="skewt").fit(returns)),
    ("Histórico (empírico)",             HistoricalVaR(0.95).fit(returns)),
    ("Monte Carlo Normal (50k)",         MonteCarloVaR(0.95, 1, 50_000, 42).fit(returns)),
    ("GARCH(1,1) Paramétrico Normal",    GARCHParametricVaR(0.95, dist="normal").fit(returns)),
    ("GARCH(1,1) Paramétrico t",         GARCHParametricVaR(0.95, dist="t").fit(returns)),
    ("GARCH(1,1) Paramétrico skew-t",    GARCHParametricVaR(0.95, dist="skewt").fit(returns)),
    ("GARCH(1,1) Monte Carlo (50k)",     GARCHMonteCarloVaR(0.95, 1, 50_000, 42).fit(returns)),
]:
    print(f"  {nombre:<34}  {m.compute_var():>10.4%}  {m.compute_es():>10.4%}")

# ──────────────────────────────────────────────────────────────────────────────
#  BLOQUE 7 - Backtesting (Kupiec + Christoffersen)
#  varlib/validation/backtesting.py
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  BLOQUE 7 - Backtesting")
print("=" * 60)

from varlib.validation.backtesting import Backtester

# ── 7a. Backtesting con ParametricVaR ─────────────────────────────────────────
print("\nBacktesting ParametricVaR (ventana expansiva)")
bt_par = Backtester(ParametricVaR(0.95)).run_expanding(train, test)

kup = bt_par.kupiec_test()
print(f"\n{bt_par}")
print(f"\nTest de Kupiec (LR_uc):")
print(f"Observaciones test : {kup['n_obs']}")
print(f"Violaciones        : {kup['n_violations']}  ({kup['violation_rate']:.2%} obs.)")
print(f"Tasa esperada      : {kup['expected_rate']:.2%}  (= 1 - 95%)")
print(f"LR_uc              : {kup['LR_uc']:.4f}")
print(f"p-valor            : {kup['p_value']:.4f}")
print(f"Rechaza H0 (5%)?   : {'Sí X  modelo rechazado' if kup['reject_H0'] else 'No OK  modelo válido'}")

chri = bt_par.christoffersen_test()
print(f"\nTest de Christoffersen (LR_ind / LR_cc):")
print(f"Matriz transición:   00={chri['n00']}  01={chri['n01']}  10={chri['n10']}  11={chri['n11']}")
print(f"pi[0][1] (tras no-viol.)  : {chri['pi01']:.4f}")
print(f"pi[1][1] (tras violación) : {chri['pi11']:.4f}")
print(f"LR_ind  : {chri['LR_ind']:.4f}    p={chri['p_value_ind']:.4f}"
      f"    {'X dependencia' if chri['reject_ind'] else 'OK independiente'}")
print(f"LR_cc   : {chri['LR_cc']:.4f}    p={chri['p_value_cc']:.4f}"
      f"    {'X rechazado' if chri['reject_cc'] else 'OK válido'}")

# ── 7b. summary() ─────────────────────────────────────────────────────────────
print()
print_summary(bt_par, "summary() backtesting ParametricVaR")

# ── 7c. Ventana deslizante vs. expansiva ─────────────────────────────────────────
print(f"\nVentana deslizante (250 días) vs. expansiva - ParametricVaR 95%:")
print(f"  {'Ventana':<18}  {'Viol.':>6}  {'Tasa':>7}  {'p_Kupiec':>9}")
print(f"  {'─'*18}  {'─'*6}  {'─'*7}  {'─'*9}")
for win, label in [(None, "Expansiva"), (250, "Deslizante 250d"), (125, "Deslizante 125d")]:
    bt_w = Backtester(ParametricVaR(0.95))
    if win is None:
        bt_w.run_expanding(train, test)
    else:
        bt_w.run_rolling(returns, win, eval_start=test.index[0])
    k_w = bt_w.kupiec_test()
    print(f"  {label:<18}  {k_w['n_violations']:>6}  "
          f"{k_w['violation_rate']:>7.2%}  {k_w['p_value']:>9.4f}")

# ── 7d. Capa dynamic_var (serie temporal de VaR con ventana deslizante) ───────
#  Estimador puntual (cada modelo) + aplicación rolling (capa aparte).
#  El propio Backtester reutiliza esta capa por debajo.
from varlib.validation.dynamic import dynamic_var

print(f"\ndynamic_var - serie temporal de VaR con ventana deslizante:")

# VaR deslizante diario con el paramétrico (refit barato cada día)
var_serie_par = dynamic_var(ParametricVaR(0.95), returns, window=250)
print(f"Paramétrico con ventana deslizante 250d -> {var_serie_par.notna().sum()} VaR estimados")
print(f"último VaR ({var_serie_par.dropna().index[-1].date()}): "
      f"{var_serie_par.dropna().iloc[-1]:.4%}")

# Verificación: run_rolling con eval_start reproduce exactamente la serie de
# dynamic_var sobre el test, lo que confirma que el recorte de buffer/ventana
# que hace eval_start está bien alineado.
bt_250 = Backtester(ParametricVaR(0.95)).run_rolling(returns, 250, eval_start=test.index[0])
coincide = np.allclose(
    var_serie_par.reindex(test.index).values, bt_250.var_series_.values
)
print(f"¿run_rolling(250, eval_start) reproduce dynamic_var en el test?  "
      f"{'OK series idénticas' if coincide else 'X difieren'}")

# GARCH con ventana deslizante (reajuste en cada paso)
print(f"\ndynamic_var GARCH con ventana deslizante")
var_serie_garch = dynamic_var(
    GARCHParametricVaR(0.95), returns, window=500
)
n_var = var_serie_garch.notna().sum()
print(f"GARCH(1,1) con ventana 500d -> {n_var} VaR estimados")
print(f"último VaR ({var_serie_garch.dropna().index[-1].date()}): "
      f"{var_serie_garch.dropna().iloc[-1]:.4%}")

# ── 7e. Comparación sistemática de modelos (deslizante y expansiva) ───────────
#  Recibe un dict {etiqueta: Backtester} y devuelve una línea resumen por modelo.
from varlib import compare_models_rolling, compare_models_expanding

WIN_CMP  = WARMUP          # ventana deslizante = warm-up expansivo: ambas alineadas
EVAL_CMP = test.index[0]   # las dos evalúan desde el mismo día
CONF     = 0.95            # confianza común a toda la comparación

# Un Backtester por cada combinación de metodología y distribución.
modelos_cmp = {
    "Historico":           Backtester(HistoricalVaR(CONF)),
    "Param-Normal":        Backtester(ParametricVaR(CONF, dist="normal")),
    "Param-t":             Backtester(ParametricVaR(CONF, dist="t")),
    "Param-skewt":         Backtester(ParametricVaR(CONF, dist="skewt")),
    "MonteCarlo":          Backtester(MonteCarloVaR(CONF, n_simulations=50_000, random_state=42)),
    "GARCH-Param-Normal":  Backtester(GARCHParametricVaR(CONF, dist="normal")),
    "GARCH-Param-t":       Backtester(GARCHParametricVaR(CONF, dist="t")),
    "GARCH-Param-skewt":   Backtester(GARCHParametricVaR(CONF, dist="skewt")),
    "GARCH-MC":            Backtester(GARCHMonteCarloVaR(CONF, n_simulations=50_000, random_state=42)),
}

# Vista compacta para consola: el summary() tiene 16 columnas (demasiado ancho),
# así que se imprime solo el subconjunto relevante del backtest.
_COLS_CMP = ["modelo", "n_violations", "violation_rate",
             "expected_rate", "p_uc", "p_cc", "reject_cc"]

def print_cmp(df):
    print(df[_COLS_CMP].to_string(index=False, formatters={
        "violation_rate": lambda x: f"{x:.2%}",
        "expected_rate": lambda x: f"{x:.2%}",
        "p_uc": lambda x: f"{x:.4f}",
        "p_cc": lambda x: f"{x:.4f}",
    }))

# Tabla del reporte: ventana DESLIZANTE de WIN_CMP días sobre todas las combinaciones.
cmp_rolling = compare_models_rolling(modelos_cmp, returns, WIN_CMP, eval_start=EVAL_CMP)
print(f"\nComparación sistemática - ventana DESLIZANTE:")
print_cmp(cmp_rolling)

# La misma comparación con ventana EXPANSIVA (warm-up = train, evalúa sobre test).
cmp_expanding = compare_models_expanding(modelos_cmp, train, test)
print(f"\nComparación sistemática - ventana EXPANSIVA")
print_cmp(cmp_expanding)

# ── 7f. Guards ────────────────────────────────────────────────────────────────
print(f"\nTest de guards:")
try:
    Backtester("")
except TypeError as e:
    print(f"TypeError    (esperado) OK  ->  {e}")
try:
    Backtester(ParametricVaR()).kupiec_test()
except RuntimeError as e:
    print(f"RuntimeError (esperado) OK  ->  {e}")
try:
    dynamic_var(ParametricVaR(), returns, window=10, min_obs=30)   # ventana < min_obs
except ValueError as e:
    print(f"ValueError   (esperado) OK  ->  {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  BLOQUE 8 - Reporting (VaRPlotter + ReportExporter)
#  Pasos 10-11 del plan: varlib/reporting/plots.py + reports.py
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  BLOQUE 8 - Reporting")
print("=" * 60)

from pathlib import Path
from varlib.reporting.plots   import VaRPlotter
from varlib.reporting.reports import ReportExporter

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

plotter  = VaRPlotter()

# ── 8a. Plot retornos + VaR ───────────────────────────────────────────────────
print("\nGráficos")

# Modelo base para los gráficos
model_par = ParametricVaR(0.95, 1).fit(returns)

fig_var_retornos = plotter.plot_var(
    returns,
    model_par.compute_var(),
    model_par.compute_es(),
    title="AAPL - Retornos diarios con VaR Paramétrico 95% estático",
)
plotter.save(fig_var_retornos, OUTPUT_DIR / "fig_retornos_var.png")
print(f"fig_retornos_var.png guardada")

# ── 8c. Histograma Monte Carlo ────────────────────────────────────────────────
mc_model = GARCHMonteCarloVaR(0.95, 1, 50_000, 42).fit(returns)
mc_model.simulate_paths()
fig_hist_montecarlo = plotter.plot_simulation_histogram(
    mc_model.simulated_returns_,
    var_value=mc_model.compute_var(),
    es_value=mc_model.compute_es(),
    title="AAPL - Distribución normal simulada GARCH-MC (50k, h=1)",
)
plotter.save(fig_hist_montecarlo, OUTPUT_DIR / "fig_sim_hist.png")
print(f"fig_sim_hist.png guardada")

# ── 8d. Volatilidad condicional GARCH ─────────────────────────────────────────
model_gp  = GARCHParametricVaR(0.95, 1).fit(returns)

# Vol. condicional y pronóstico, ambos diarios, plot_volatility los anualiza.
fig_volatilidad = plotter.plot_volatility(
    model_gp.conditional_vol_series_, forecast=model_gp.forecast_volatility(steps=30),
    title="AAPL - Volatilidad condicional GARCH(1,1)")
plotter.save(fig_volatilidad, OUTPUT_DIR / "fig_volatilidad.png")
print(f"fig_volatilidad.png guardada")

# ── 8e. Violaciones de backtesting (VaR dinámico, ventana expansiva) ──────────
fig_violaciones = plotter.plot_var(
    test, bt_par.var_series_,
    title="AAPL - Backtesting VaR Paramétrico 95% (solo test, ventana expansiva)",
)
plotter.save(fig_violaciones, OUTPUT_DIR / "fig_violations.png")
print(f"fig_violations.png guardada")

# ── 8e-bis. VaR rolling sobre la serie COMPLETA (capa dynamic_var) ────────────
fig_var_deslizante = plotter.plot_var(
    returns, var_serie_par,
    title="AAPL - Backtesting VaR Paramétrico 95% (serie completa, ventana deslizante)",
)
plotter.save(fig_var_deslizante, OUTPUT_DIR / "fig_dynamic_var.png")
print(f"fig_dynamic_var.png guardada")

# ── 8f. Exportar reporte HTML ─────────────────────────────────────────────────
print(f"\nExportando HTML", flush=True)
exporter = ReportExporter(author="varlib-PFG", ticker="AAPL")

# Params del GARCH (dict -> tabla clave/valor): valores del summary().
s_gp = model_gp.summary()
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
# VaR/ES puntuales del GARCH; Styler solo para mostrarlos como %.
garch_var = pd.DataFrame([{
    "VaR 95% / 1d": model_gp.compute_var(),
    "ES 95% / 1d":  model_gp.compute_es(),
}]).style.format("{:.4%}")

# {titulo: contenido}; una lista agrupa varios items (tablas/figuras) por sección.
results_report = {
    # El GARCH: sus parámetros, su VaR/ES puntual y la volatilidad condicional.
    "Modelo GARCH(1,1)": [garch_params, garch_var, fig_volatilidad],
    # VaR estático sobre los retornos.
    "VaR puntual sobre los retornos": fig_var_retornos,
    # Distribución simulada por Monte Carlo (con VaR/ES).
    "Distribución simulada (Monte Carlo)": fig_hist_montecarlo,
    # Cada comparación de backtesting junto a la gráfica de su misma ventana.
    "Comparación backtesting con ventana deslizante": [cmp_rolling, fig_var_deslizante],
    "Comparación backtesting con ventana expansiva": [cmp_expanding, fig_violaciones],
}

html_path = OUTPUT_DIR / "aapl_report.html"
exporter.export_html(results_report, filepath=html_path)
print(f"Reporte HTML creado y guardado en: {html_path}")

print("=" * 60)
print("  TODOS LOS BLOQUES COMPLETADOS CORRECTAMENTE")
print("=" * 60)
