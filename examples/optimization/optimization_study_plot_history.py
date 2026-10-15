"""
Optimization history of optimization_study_three_point_bending.py: compliance normalized by the initial design and
mean relative density (kriging surrogate used in the constraint) at each SLSQP iteration.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

save_dir = Path(__file__).resolve().parents[2] / "data" / "outputs" / "optimization_study" / "Three_point_bending"
cases = {"hybrid": ("Hybrid (2 var.)", "#2a78d6", "o"),
         "graded": ("Graded (3 var.)", "#eb6834", "s"),
         "unit_cell": ("Cell-wise (500 var.)", "#1baf7a", "^")}

fig, (ax_c, ax_rho) = plt.subplots(1, 2, figsize=(10, 3.8))
target = None
for name, (label, color, marker) in cases.items():
    case = json.loads((save_dir / f"{name}.json").read_text())
    history = case["history"]
    C0 = case["compliance"]["initial_DDM_RBF"]
    target = case["relative_density"]["target"]
    # First record: initial design before any objective evaluation (compliance = C0)
    compliance = [1.0] + [c / C0 for c in history["objective"][1:]]
    iterations = range(len(compliance))
    every = max(1, len(compliance) // 12)
    for ax, values in ((ax_c, compliance), (ax_rho, history["relative_density"])):
        ax.plot(iterations, values, color=color, lw=2, marker=marker, ms=6, markevery=every, label=label)
    ax_c.annotate(f"{compliance[-1]:.3f}", (iterations[-1], compliance[-1]), textcoords="offset points",
                  xytext=(4, 0), va="center", fontsize=8, color="#444444")

ax_rho.axhline(target, color="#888888", lw=1, ls="--", label=f"Target $\\bar\\rho$ = {target}")
ax_c.set_ylabel("Compliance $C/C_0$ (DDM + RBF)")
ax_rho.set_ylabel(r"Mean relative density $\bar\rho$")
for ax in (ax_c, ax_rho):
    ax.set_xlabel("SLSQP iteration")
    ax.grid(color="#e5e5e5", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
ax_c.set_ylim(bottom=0)
ax_c.legend(frameon=False, fontsize=9)
ax_rho.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(save_dir / "optimization_history.png", dpi=200)
fig.savefig(save_dir / "optimization_history.pdf")
print("Saved", save_dir / "optimization_history.png")
