"""Render the headline recovery-rate figure for the README.

The numbers mirror Table 1 in results/RESULTS.md (recovery rate by model and
refresh policy, with Wilson 95% confidence intervals, n=144 per cell).

Run:  python3 docs/plot_recovery.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "recovery.png")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

policies = ["never", "batch", "immediate"]
models = ["Llama-3.2-1B", "Qwen2.5-3B", "Qwen2.5-7B"]

# rate, ci_low, ci_high  indexed [model][policy]
data = {
    "Llama-3.2-1B": [(0.00, 0.00, 0.03), (0.04, 0.02, 0.09), (0.42, 0.34, 0.50)],
    "Qwen2.5-3B":   [(0.00, 0.00, 0.03), (0.25, 0.19, 0.33), (0.58, 0.50, 0.66)],
    "Qwen2.5-7B":   [(0.00, 0.00, 0.03), (0.25, 0.14, 0.41), (0.38, 0.30, 0.46)],
}

x = np.arange(len(policies))
width = 0.26
colors = ["#9ecae1", "#4292c6", "#08519c"]

fig, ax = plt.subplots(figsize=(6.4, 3.6))
for i, m in enumerate(models):
    rates = [data[m][p][0] for p in range(len(policies))]
    low = [data[m][p][0] - data[m][p][1] for p in range(len(policies))]
    high = [data[m][p][2] - data[m][p][0] for p in range(len(policies))]
    ax.bar(x + (i - 1) * width, rates, width, label=m, color=colors[i],
           yerr=[low, high], capsize=3, error_kw={"elinewidth": 1})

ax.set_xticks(x)
ax.set_xticklabels(policies)
ax.set_ylabel("Recovery rate (higher = fresher)")
ax.set_xlabel("Refresh policy")
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
