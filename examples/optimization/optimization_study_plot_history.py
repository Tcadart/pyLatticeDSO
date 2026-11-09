"""
Optimization history of optimization_study_three_point_bending.py: compliance normalized by the initial design and
mean relative density (kriging surrogate used in the constraint) at each SLSQP iteration.
Solid lines: DDM + RBF surrogate, dashed lines: direct FEM with analytic gradient ("<name>_FEM" cases, when available).
Study folder: data/outputs/optimization_study/Three_point_bending by default, or `--input=optimization/<name>`.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

name_file = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--input=")), "optimization/Three_point_bending")
save_dir = Path(__file__).resolve().parents[2] / "data" / "outputs" / "optimization_study" / Path(name_file).name
cases = {"hybrid": ("Hybrid", "#2a78d6", "o"),
         "graded": ("Graded", "#eb6834", "s"),
         "unit_cell": ("Cell-wise", "#1baf7a", "^")}

fig, (ax_c, ax_rho) = plt.subplots(1, 2, figsize=(10, 3.8))
target = None
runs = [(name, label, color, marker, "-") for name, (label, color, marker) in cases.items()
        if (save_dir / f"{name}.json").exists()]
runs += [(f"{name}_FEM", f"{label} FEM", color, marker, "--") for name, (label, color, marker) in cases.items()
         if (save_dir / f"{name}_FEM.json").exists()]
for name, label, color, marker, linestyle in runs:
    case = json.loads((save_dir / f"{name}.json").read_text())
    label = f"{label} ({case['n_design_variables']} var.)"
    history = case["history"]
    C0 = case["compliance"][f"initial_{case.get('model', 'DDM_RBF')}"]
    target = case["relative_density"]["target"]
    # First record: initial design before any objective evaluation (compliance = C0)
    compliance = [1.0] + [c / C0 for c in history["objective"][1:]]
    iterations = range(len(compliance))
    every = max(1, len(compliance) // 12)
    for ax, values in ((ax_c, compliance), (ax_rho, history["relative_density"])):
        ax.plot(iterations, values, color=color, lw=2, ls=linestyle, marker=marker, ms=6, markevery=every,
                label=label)
    ax_c.annotate(f"{compliance[-1]:.3f}", (iterations[-1], compliance[-1]), textcoords="offset points",
                  xytext=(4, 0), va="center", fontsize=8, color="#444444")

ax_rho.axhline(target, color="#888888", lw=1, ls="--", label=f"Target $\\bar\\rho$ = {target}")
ax_c.set_ylabel("Compliance $C/C_0$ (optimization model)")
ax_rho.set_ylabel(r"Mean relative density $\bar\rho$")
for ax in (ax_c, ax_rho):
    ax.set_xlabel("SLSQP iteration")
    ax.grid(color="#e5e5e5", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
ax_c.set_ylim(bottom=0)
ax_c.legend(frameon=False, fontsize=9)
ax_rho.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(save_dir / "optimization_history.png", dpi=200)
fig.savefig(save_dir / "optimization_history.pdf")
print("Saved", save_dir / "optimization_history.png")
