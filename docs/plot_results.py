"""Render the headline figure for the README: the recency-order fix is model-dependent.

Numbers mirror the 6-trial headline (immediate recovery, TF-IDF, Wilson 95% CI, n=144) reported
in results/RESULTS.md. Run:  python3 docs/plot_results.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "recency_fix.png")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

labels = ["Q2.5-1.5B", "Q2.5-3B", "Q2.5-7B", "L3.2-1B", "L3.2-3B",
          "L3.1-8B", "G2-2B", "G2-9B", "G4-E2B", "G4-E4B"]
naive   = [0.50, 0.62, 0.51, 0.33, 0.50, 0.54, 0.50, 0.46, 0.54, 0.50]
recency = [0.04, 0.75, 0.54, 0.54, 0.62, 1.00, 0.67, 0.92, 0.79, 0.33]

x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w/2, naive, w, label="naive order", color="#bdbdbd")
ax.bar(x + w/2, recency, w, label="recency order (newest last)", color="#08519c")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Immediate recovery (higher = fresher)"); ax.set_ylim(0, 1.08)
ax.axhline(0.5, color="grey", ls=":", lw=0.7)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
