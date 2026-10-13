"""
Example script to compare the homogenization of a cell with the 3D solid model (reference), the beam model and
the beam models with penalized beams at nodes ("beam mod"), for a range of radii.
The penalized zone at a node has a length L_mod given by the penalization strategy:
- "angle": L_mod = r_neighbor / tan(angle / 2) (Cadart et al. 2025)
- "length" (strategy 1): L_mod = beam length / 10
- "radius" (strategy 2): L_mod = beam radius
- "sqrt3_radius" (strategy 3): L_mod = sqrt(3) * beam radius

The homogenized stiffness matrices are saved in data/outputs/homogenization_comparaison/results.json after each
simulation, per cell (geometry, cell size, material and periodicity of the input file): the simulations already
saved for this cell are not computed again.
The static error of a beam model is eps_static = ||S_solid - S_beam||_F / ||S_solid||_F, with S the homogenized
stiffness matrix in engineering Voigt notation (shear strains gamma = 2 eps, so that C44 = G).
The homogenization classes build the matrix with tensor shear strains (C44 = 2G): its shear columns are divided by 2
before computing the error.
"""
import fcntl
import json
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pyLatticeDesign.utils import open_lattice_parameters
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_simulation import get_homogenized_properties, get_homogenized_properties_solid

name_file = "simulation/homogenization_Cubic"
radii = np.round(np.arange(0.025, 0.2 + 1e-9, 0.025), 4)

# Solid model settings (part of the result key, so that changing them triggers new simulations)
mesh_refinement = 0.5
element_degree = 2

# Beam models with penalized beams at nodes: model name -> penalization strategy
penalization_strategies = {
    "beam_mod": "angle",
    "beam_mod_length": "length",
    "beam_mod_radius": "radius",
    "beam_mod_sqrt3_radius": "sqrt3_radius",
}

# Model name -> key of its results in the results file
models = {
    "solid": f"solid_P{element_degree}_curved_refinement_{mesh_refinement}",  # curved geometry of order P
    "beam": "beam",
    **{model: model for model in penalization_strategies},
}
beam_models = ["beam", "beam_mod_length", "beam_mod_radius", "beam_mod_sqrt3_radius", "beam_mod"]  # strategy order
labels = {"solid": "Solid", "beam": "Without strategy", "beam_mod_length": "Strategy 1: L/10",
          "beam_mod_radius": "Strategy 2: Radius", "beam_mod_sqrt3_radius": r"Strategy 3: $\sqrt{3}\times$Radius",
          "beam_mod": r"Strategy 4: Radius / tan($\gamma$/2)"}
colors = {"solid": "tab:gray", "beam": "crimson", "beam_mod_length": "deepskyblue", "beam_mod_radius": "mediumseagreen",
          "beam_mod_sqrt3_radius": "orange", "beam_mod": "darkviolet"}
markers = {"solid": "x", "beam": "o", "beam_mod_length": "s", "beam_mod_radius": "^", "beam_mod_sqrt3_radius": "o",
           "beam_mod": "*"}

save_dir = Path(__file__).resolve().parents[2] / "data" / "outputs" / "homogenization_comparaison"
save_dir.mkdir(parents=True, exist_ok=True)
results_path = save_dir / "results.json"


def get_cell_key(file_name: str) -> str:
    """Identifier of the homogenized cell built from the input file content (the radius is swept separately)."""
    parameters = open_lattice_parameters(file_name)
    geometry = parameters["geometry"]
    simulation = parameters.get("simulation_parameters", {})
    size = geometry["cell_size"]
    key = f"{'_'.join(geometry['geom_types'])}_cell_{size['x']}x{size['y']}x{size['z']}"
    key += f"_{simulation.get('material', 'VeroClear')}"
    if simulation.get("periodicity", False):
        key += "_periodic"
    return key


cell_key = get_cell_key(name_file)
print("Cell:", cell_key)


def load_results() -> dict:
    if results_path.exists():
        with open(results_path, "r") as file:
            return json.load(file)
    return {}


def save_result(cell: str, model_key: str, radius_key: str, entry: dict) -> dict:
    """
    Add one simulation to the results file and return its up-to-date content.
    The file is locked, read again and only completed, so that several scripts running at the same time do not
    erase the results of each other.
    """
    with open(results_path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = load_results()
        data.setdefault(cell, {}).setdefault(model_key, {})[radius_key] = entry
        tmp_path = results_path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp_path, "w") as file:
            json.dump(data, file, indent=2)
        tmp_path.replace(results_path)  # never leave a half-written results file
        fcntl.flock(lock, fcntl.LOCK_UN)
    return data


def homogenize(model: str, radius: float) -> dict:
    """Homogenization of the cell with the given model and radius."""
    lattice = LatticeSim(name_file)
    if model in penalization_strategies:
        lattice.penalization_strategy = penalization_strategies[model]  # applied when the cell is rebuilt below
    lattice.reset_cell_with_new_radii([float(radius)])
    start = time.time()
    if model == "solid":
        mat_S, analysis = get_homogenized_properties_solid(lattice, mesh_refinement=mesh_refinement,
                                                           element_degree=element_degree, verbose=0)
        relative_density = analysis.get_relative_density()
    else:
        if model == "beam":
            lattice.reset_penalized_beams()
        mat_S, analysis = get_homogenized_properties(lattice)
        relative_density = None
    return {"stiffness": analysis.homogenizeMatrix.tolist(),
            "compliance_orthotropic": np.asarray(mat_S).tolist(),
            "relative_density": relative_density,
            "time": time.time() - start}


# Run the missing simulations
for model, key in models.items():
    for radius in radii:
        radius_key = f"{radius:.4f}"
        if radius_key in load_results().get(cell_key, {}).get(key, {}):
            continue
        print(f"Homogenization {model} r = {radius_key}")
        save_result(cell_key, key, radius_key, homogenize(model, radius))
results = load_results()[cell_key]

def to_engineering_voigt(stiffness) -> np.ndarray:
    """Stiffness matrix with tensor shear strains (C44 = 2G) -> engineering Voigt notation (C44 = G)."""
    C = np.array(stiffness, dtype=float)
    C[:, 3:] /= 2.0
    return 0.5 * (C + C.T)


# Static error with the solid model as reference
# eps^2 = eps_normal^2 + eps_shear^2 + eps_coupling^2, with each block of S_solid - S_beam measured against ||S_solid||_F
normal, shear = slice(0, 3), slice(3, 6)
reference = results[models["solid"]]
errors, errors_normal, errors_shear = {}, {}, {}
for radius in radii:
    radius_key = f"{radius:.4f}"
    S_solid = to_engineering_voigt(reference[radius_key]["stiffness"])
    norm_solid = np.linalg.norm(S_solid, "fro")
    for model in beam_models:
        delta = to_engineering_voigt(results[models[model]][radius_key]["stiffness"]) - S_solid
        errors.setdefault(model, []).append(np.linalg.norm(delta, "fro") / norm_solid)
        errors_normal.setdefault(model, []).append(np.linalg.norm(delta[normal, normal], "fro") / norm_solid)
        errors_shear.setdefault(model, []).append(np.linalg.norm(delta[shear, shear], "fro") / norm_solid)

# Relative density of the solid model (exact volume of the fused beams), used as x-axis of the figures
relative_densities = np.array([reference[f"{r:.4f}"]["relative_density"] for r in radii])

print(f"\nStatic error eps_static (%) - {cell_key}")
print(f"{'Radius':>7s} {'rho':>7s} " + " ".join(f"{model:>22s}" for model in beam_models))
for i, radius in enumerate(radii):
    print(f"{radius:7.3f} {relative_densities[i]:7.4f} "
          + " ".join(f"{100 * errors[model][i]:21.2f}%" for model in beam_models))

with open(save_dir / f"static_errors_{cell_key}.json", "w") as file:
    json.dump({"cell": cell_key, "radii": radii.tolist(), "relative_densities": relative_densities.tolist(),
               "reference": models["solid"], "penalization_strategies": penalization_strategies,
               **{model: values for model, values in errors.items()},
               **{f"{model}_normal_block": values for model, values in errors_normal.items()},
               **{f"{model}_shear_block": values for model, values in errors_shear.items()}}, file, indent=2)


def stiffness_ratio(model: str, i: int, j: int) -> np.ndarray:
    """C_ij of a model divided by C_ij of the solid model."""
    values = [results[models[model]][f"{r:.4f}"]["stiffness"][i][j] for r in radii]
    values_solid = [reference[f"{r:.4f}"]["stiffness"][i][j] for r in radii]
    return np.array(values) / np.array(values_solid)


# Figure
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax_error, ax_split, ax_normal, ax_shear = axes.flatten()

for model in beam_models:
    style = {"marker": markers[model], "color": colors[model]}
    ax_error.plot(relative_densities, 100 * np.array(errors[model]), label=labels[model], **style)
    ax_split.plot(relative_densities, 100 * np.array(errors_normal[model]), label=f"{labels[model]} - normal",
                  **style)
    ax_split.plot(relative_densities, 100 * np.array(errors_shear[model]), linestyle="--", mfc="none",
                  label=f"{labels[model]} - shear", **style)
    ax_normal.plot(relative_densities, stiffness_ratio(model, 0, 0), label=f"{labels[model]} $C_{{11}}$", **style)
    ax_normal.plot(relative_densities, stiffness_ratio(model, 2, 2), linestyle="--", mfc="none",
                   label=f"{labels[model]} $C_{{33}}$", **style)
    ax_shear.plot(relative_densities, stiffness_ratio(model, 3, 3), label=f"{labels[model]} $C_{{44}}$", **style)
    ax_shear.plot(relative_densities, stiffness_ratio(model, 5, 5), linestyle="--", mfc="none",
                  label=f"{labels[model]} $C_{{66}}$", **style)

ax_error.set_ylabel(r"$\varepsilon_{static}$ (%)")
ax_error.set_title(r"$\|S^{solid} - S^{beam}\|_F \,/\, \|S^{solid}\|_F$")
ax_split.set_ylabel("Contribution to the error (%)")
ax_split.set_title(r"Blocks $C_{11..33}$ (normal, solid line) and $C_{44..66}$ (shear, dashed)")
ax_normal.set_title(r"Normal stiffness / solid: $C_{11}$ (solid line), $C_{33}$ (dashed)")
ax_shear.set_title(r"Shear stiffness / solid: $C_{44}$ (solid line), $C_{66}$ (dashed)")
for ax in [ax_normal, ax_shear]:
    ax.axhline(1.0, color=colors["solid"], linewidth=1.5, label="Solid")
    ax.set_ylabel("Beam / solid")

for ax in axes.flatten():
    ax.set_xlabel("Relative density")
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.05, 0.1, 0.3, 0.5])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(alpha=0.3)
ax_error.legend(fontsize=8)
for ax in [ax_split, ax_normal, ax_shear]:
    ax.legend(fontsize=6, ncol=2)

fig.suptitle(cell_key)
fig.tight_layout()
fig.savefig(save_dir / f"static_error_{cell_key}.png", dpi=150)
print("Results saved to", save_dir)
plt.show()
