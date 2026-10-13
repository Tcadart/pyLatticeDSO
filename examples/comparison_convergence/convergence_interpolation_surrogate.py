"""
Accuracy of the Schur complement surrogate models (reduced basis + interpolation) with pyLatticeSim.

The surrogate prediction is compared with exact Schur complements computed on an independent test set
(Latin hypercube sampling of the parameter space, not on the training grid), as a function of:
    - the reduced basis size N,
    - the number of training snapshots (sub-sampled training grids),
    - the interpolation method (nearest neighbor, linear, thin-plate spline RBF).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.stats import qmc

from pyLatticeSim.utils_schur import load_schur_complement_dataset, get_schur_complement
from pyLatticeSim.greedy_algorithm import reduce_basis_greedy
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_rbf import ThinPlateSplineRBF

name_file = "simulation/hybrid_cell_simulation"
n_test = 100
radius_bounds = (0.01, 0.10)
tolerance_greedy = 1e-12  # Full greedy, the basis for a given tolerance is obtained by truncation (nested bases)
tolerances_table = [1e-3, 1e-6]
grid_steps = [0.03, 0.02, 0.01]  # Sub-sampling of the training grid (radius step)
methods = ["nearest_neighbor", "linear", "RBF"]

save_dir = Path(__file__).parents[2] / "data" / "outputs" / "reduced_basis_convergence"
save_dir.mkdir(parents=True, exist_ok=True)

lattice_Sim_object = LatticeSim(name_file)
schur_data = load_schur_complement_dataset(lattice_Sim_object)
n_params = len(lattice_Sim_object.geom_types)

# =============================================================================
# Test set: exact Schur complements on Latin hypercube samples (cached)
# =============================================================================
test_file = save_dir / f"schur_test_LHS_{n_test}.npz"
sampler = qmc.LatinHypercube(d=n_params, seed=0)
radii_test = qmc.scale(sampler.random(n_test), *radius_bounds)
schur_test = list(np.load(test_file)["schur_matrices"]) if test_file.exists() else []
for i in range(len(schur_test), n_test):
    lattice_Sim_object.reset_cell_with_new_radii(list(radii_test[i]))
    schur_test.append(get_schur_complement(lattice_Sim_object))
    np.savez(test_file, radius_values=radii_test[:len(schur_test)], schur_matrices=np.array(schur_test))
    print(f"Test sample {i + 1}/{n_test}: {np.round(radii_test[i], 4)}", flush=True)
S_test = np.stack([np.ravel(S, order='F') for S in schur_test], axis=1)  # (n_dof, n_test)
norm_test = np.linalg.norm(S_test, axis=0)


def relative_error(S_pred):
    error = np.linalg.norm(S_pred - S_test, axis=0) / norm_test
    return error.mean(), error.max()


def interpolate_alphas(method, radii_train, alpha_train):
    if method == "nearest_neighbor":
        return NearestNDInterpolator(radii_train, alpha_train)(radii_test)
    if method == "linear":
        alpha = LinearNDInterpolator(radii_train, alpha_train)(radii_test)
        outside = np.any(np.isnan(alpha), axis=1)  # Fallback outside the convex hull (as in LatticeSim)
        alpha[outside] = NearestNDInterpolator(radii_train, alpha_train)(radii_test[outside])
        return alpha
    if method == "RBF":
        return ThinPlateSplineRBF(radii_train, alpha_train).evaluate(radii_test)
    raise ValueError(method)


# =============================================================================
# Surrogate errors
# =============================================================================
keys_all = sorted(schur_data.keys())
radii_all = np.array(keys_all)
r_min = radii_all.min()

results = {}
for step in grid_steps:
    on_grid = np.all(np.isclose(np.mod(np.round((radii_all - r_min) / 0.01), round(step / 0.01)), 0), axis=1)
    schur_subset = {k: schur_data[k] for k, keep in zip(keys_all, on_grid) if keep}
    radii_train = np.array(sorted(schur_subset.keys()))
    S_train = np.stack([np.ravel(schur_subset[tuple(k)], order='F') for k in radii_train], axis=1)

    residual_history = []
    _, _, _, basis_full, _, _, _ = reduce_basis_greedy(schur_subset, tolerance_greedy, verbose=0,
                                                       residual_history=residual_history)
    residual_history = np.array(residual_history)
    basis_full = np.linalg.qr(basis_full)[0]  # Same nested subspaces, orthonormal up to machine precision

    n_max = basis_full.shape[1]
    N_values = np.unique(np.concatenate([np.unique(np.geomspace(1, n_max, 25).astype(int)),
                                         [np.argmax(residual_history < tol) + 1 for tol in tolerances_table
                                          if np.any(residual_history < tol)]]))
    res = {"n_snapshots": len(schur_subset), "N": N_values, "projection": np.zeros((len(N_values), 2)),
           "residual_greedy": residual_history}
    for method in methods:
        res[method] = np.zeros((len(N_values), 2))

    for i, N in enumerate(N_values):
        basis = basis_full[:, :N]
        # Best approximation in the reduced basis (no interpolation error)
        res["projection"][i] = relative_error(basis @ (basis.T @ S_test))
        alpha_train = (basis.T @ S_train).T  # (n_train, N)
        for method in methods:
            res[method][i] = relative_error(basis @ interpolate_alphas(method, radii_train, alpha_train).T)
    results[step] = res

    print(f"\nStep {step}: {len(schur_subset)} snapshots")
    for tol in tolerances_table:
        if not np.any(residual_history < tol):
            print(f"    tol = {tol:.0e}: not reached (N max = {n_max})")
            continue
        i = np.searchsorted(N_values, np.argmax(residual_history < tol) + 1)
        line = f"    tol = {tol:.0e}, N = {N_values[i]:3d} | projection {res['projection'][i, 0]:.2e}/{res['projection'][i, 1]:.2e}"
        for method in methods:
            line += f" | {method} {res[method][i, 0]:.2e}/{res[method][i, 1]:.2e}"
        print(line + "   (mean/max)")

# =============================================================================
# Plot
# =============================================================================
labels = {"projection": "Projection (no interpolation)", "nearest_neighbor": "Nearest neighbor",
          "linear": "Linear", "RBF": "RBF (thin-plate spline)"}
colors = {"projection": "grey", "nearest_neighbor": "tab:blue", "linear": "tab:orange", "RBF": "tab:green"}

fig, (ax_N, ax_snap) = plt.subplots(1, 2, figsize=(11, 4.2))

ref = results[min(grid_steps)]
for key in ["projection"] + methods:
    ax_N.loglog(ref["N"], ref[key][:, 0], "-", color=colors[key], label=f"{labels[key]} (mean)")
    ax_N.loglog(ref["N"], ref[key][:, 1], "--", color=colors[key], lw=0.9, label=f"{labels[key]} (max)")
ax_N.set_xlabel("Reduced basis size N")
ax_N.set_ylabel("Relative error $\\|S_{pred} - S\\|_F / \\|S\\|_F$")
ax_N.set_title(f"(a) Test error vs N ({ref['n_snapshots']} snapshots)")
ax_N.grid(True, which="both", alpha=0.3)
ax_N.legend(fontsize=7)

for key in ["projection"] + methods:
    n_snap = [results[step]["n_snapshots"] for step in grid_steps]
    best = [results[step][key][:, 0].min() for step in grid_steps]  # Best mean error over N
    ax_snap.loglog(n_snap, best, "o-", color=colors[key], label=labels[key])
ax_snap.set_xlabel("Number of training snapshots")
ax_snap.set_ylabel("Minimum mean relative error over N")
ax_snap.set_title("(b) Test error vs number of snapshots")
ax_snap.grid(True, which="both", alpha=0.3)
ax_snap.legend(fontsize=8)

fig.tight_layout()
fig.savefig(save_dir / "interpolation_convergence.png", dpi=300)
fig.savefig(save_dir / "interpolation_convergence.pdf")
np.savez(save_dir / "interpolation_convergence.npz",
         **{f"{key}_step_{step}": value for step, res in results.items() for key, value in res.items()})
plt.show()
