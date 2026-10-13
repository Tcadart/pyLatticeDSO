"""
Convergence study of the greedy reduced basis of the Schur complement with pyLatticeSim.

(a) Decay of the greedy stopping criterion with the basis size.
(b) Relative projection error on the full snapshot set versus basis size, for reduced bases built from
    sub-sampled training grids (different numbers of snapshots).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyLatticeSim.utils_schur import load_schur_complement_dataset
from pyLatticeSim.greedy_algorithm import reduce_basis_greedy
from pyLatticeSim.lattice_sim import LatticeSim

name_file = "simulation/hybrid_cell_simulation"
tolerance_greedy = 1e-12  # Low tolerance to observe the full decay
tolerances_marked = [1e-3, 1e-6]
grid_steps = [0.03, 0.02, 0.01]  # Sub-sampling of the training grid (radius step)

lattice_Sim_object = LatticeSim(name_file)
schur_data = load_schur_complement_dataset(lattice_Sim_object)

keys_all = sorted(schur_data.keys())
radii_all = np.array(keys_all)
snapshots_all = np.stack([np.ravel(schur_data[k], order='F') for k in keys_all]).T  # (n_dof, n_snapshots)
norm_snapshots_all = np.linalg.norm(snapshots_all, axis=0)
r_min = radii_all.min()

results = {}
for step in grid_steps:
    # Keep the grid points r = r_min + k * step
    on_grid = np.all(np.isclose(np.mod(np.round((radii_all - r_min) / 0.01), round(step / 0.01)), 0), axis=1)
    schur_subset = {k: schur_data[k] for k, keep in zip(keys_all, on_grid) if keep}

    residual_history = []
    _, _, _, basis, _, _, _ = reduce_basis_greedy(schur_subset, tolerance_greedy, verbose=0,
                                                  residual_history=residual_history)

    # Re-orthonormalization (same nested subspaces) to remove the loss of orthogonality of the last greedy vectors
    basis = np.linalg.qr(basis)[0]

    # Projection error of all snapshots on the first N basis vectors
    residual_proj = snapshots_all.copy()
    coefficients = basis.T @ snapshots_all
    projection_error_max = np.zeros(basis.shape[1])
    projection_error_mean = np.zeros(basis.shape[1])
    for n in range(basis.shape[1]):
        residual_proj -= np.outer(basis[:, n], coefficients[n])
        relative_error = np.linalg.norm(residual_proj, axis=0) / norm_snapshots_all
        projection_error_max[n] = relative_error.max()
        projection_error_mean[n] = relative_error.mean()

    results[step] = {"n_snapshots": len(schur_subset),
                     "residual_greedy": np.array(residual_history),
                     "projection_error_max": projection_error_max,
                     "projection_error_mean": projection_error_mean}
    print(f"Step {step}: {len(schur_subset)} snapshots, {basis.shape[1]} basis vectors")
    for tol in tolerances_marked:
        n_tol = np.argmax(np.array(residual_history) < tol) + 1
        print(f"    tol = {tol:.0e} -> N = {n_tol}, max projection error = {projection_error_max[n_tol - 1]:.3e}")

# Plot
fig, (ax_res, ax_proj) = plt.subplots(1, 2, figsize=(11, 4.2))

ref = results[min(grid_steps)]
N_ref = np.arange(1, ref["residual_greedy"].size + 1)
ax_res.semilogy(N_ref, ref["residual_greedy"], "k-", label=f"{ref['n_snapshots']} snapshots")
for tol in tolerances_marked:
    n_tol = np.argmax(ref["residual_greedy"] < tol) + 1
    ax_res.axhline(tol, color="grey", ls=":", lw=0.8)
    ax_res.plot(n_tol, ref["residual_greedy"][n_tol - 1], "o", color="tab:red")
    ax_res.annotate(f"tol = {tol:.0e}, N = {n_tol}", (n_tol, tol), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
ax_res.set_xlabel("Reduced basis size N")
ax_res.set_ylabel("Greedy stopping criterion $\\max_i \\|r_i\\|_1 / \\max_i \\|s_i\\|_1$")
ax_res.set_title("(a) Greedy residual decay")
ax_res.grid(True, which="both", alpha=0.3)

colors = ["tab:blue", "tab:orange", "k"]
for color, step in zip(colors, grid_steps):
    res = results[step]
    N = np.arange(1, res["projection_error_max"].size + 1)
    ax_proj.semilogy(N, res["projection_error_max"], "-", color=color,
                     label=f"{res['n_snapshots']} snapshots (max)")
    ax_proj.semilogy(N, res["projection_error_mean"], "--", color=color,
                     label=f"{res['n_snapshots']} snapshots (mean)")
ax_proj.set_xlabel("Reduced basis size N")
ax_proj.set_ylabel("Relative projection error $\\|S - \\Pi_N S\\|_F / \\|S\\|_F$")
ax_proj.set_title(f"(b) Projection error on the {snapshots_all.shape[1]} snapshots")
ax_proj.grid(True, which="both", alpha=0.3)
ax_proj.legend(fontsize=8)

fig.tight_layout()

save_dir = Path(__file__).parents[2] / "data" / "outputs" / "reduced_basis_convergence"
save_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(save_dir / "greedy_convergence.png", dpi=300)
fig.savefig(save_dir / "greedy_convergence.pdf")
np.savez(save_dir / "greedy_convergence.npz",
         **{f"{key}_step_{step}": value for step, res in results.items() for key, value in res.items()})
plt.show()
