"""Spearman-Brown gives a RELIABILITY (variance share). A correlation ceiling is its square root.

Model: each arena model's log-length = shared prompt-driven signal T + independent model noise.
Pick var(T) so the two models correlate at r=0.590 (the repo's measured agreement), then ask what
correlation the *best possible prompt-only predictor* (T itself) can reach with (a) one model's
length and (b) the two-model average -- the repo's length target.
"""
import numpy as np
from scipy.stats import spearmanr

r = 0.590
rng = np.random.default_rng(0)
n = 400_000
T = rng.normal(0, np.sqrt(r), n)
e1, e2 = rng.normal(0, np.sqrt(1 - r), (2, n))
x1, x2 = T + e1, T + e2
avg = (x1 + x2) / 2

R = 2 * r / (1 + r)
print(f"models agree: spearman {spearmanr(x1, x2).statistic:.3f}  (target {r})")
print(f"Spearman-Brown reliability of the average R = {R:.3f}   <- what target_ceiling() returns")
print(f"best possible predictor (T) vs ONE model : spearman {spearmanr(T, x1).statistic:.3f}"
      f"   (sqrt(r)={np.sqrt(r):.3f}; docs use {r})")
print(f"best possible predictor (T) vs the AVERAGE: spearman {spearmanr(T, avg).statistic:.3f}"
      f"   (sqrt(R)={np.sqrt(R):.3f}; docs use {R:.3f})")
head = 0.573
print(f"\nshipped head rho={head}: {head/R:.0%} of the docs' 'ceiling' {R:.3f} "
      f"vs {head/np.sqrt(R):.0%} of the correlation ceiling {np.sqrt(R):.3f}")
