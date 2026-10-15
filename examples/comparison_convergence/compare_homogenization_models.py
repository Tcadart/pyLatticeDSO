"""
Validation of the homogenization module on 4 periodic cells (Cubic, BCC, Octet, BCCZ): comparison of the effective
stiffness given by
- the 3D solid model (periodic solid FE simulation, reference),
- the beam model,
- the beam model with penalized beams at nodes ("beam mod", angle strategy of Cadart et al. 2025).

Figures
-------
- The directional Young's modulus surface E(n) is cut in 4 quadrants around the z-axis: solid, beam, solid, beam
  penalized. The cells have the cubic (or tetragonal for BCCZ) symmetry, so each quadrant contains all the information
  of the surface, and each beam model lies between two solid quadrants: the jumps at the borders show the differences
  with the reference.
- Sections of these surfaces in the plane (x, y) and in the diagonal plane ([110], z), all models superimposed.
- Ex / Es, Gxy / Es and the relative error against the reference as a function of the relative density.

The homogenizations are read from (and missing ones are added to) data/outputs/homogenization_comparaison/results.json,
shared with compare_homogenization_solid_beam.py.
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

project_root = Path(__file__).resolve().parents[2]

cells = {
    "Cubic": "simulation/homogenization_Cubic",
    "BCC": "simulation/homogenization_BCC",
    "Octet": "simulation/homogenization_Octet",
    "BCCZ": "simulation/homogenization_BCCZ",
}
radii = np.round(np.arange(0.025, 0.2 + 1e-9, 0.025), 4)
# Octet: the solid mesh at r = 0.025 does not fit in 8 GB of memory and r > 0.15 gives rho > 0.7
cell_radii = {"Octet": radii[(radii >= 0.05) & (radii <= 0.15)]}
radii_surface = [0.05, 0.1, 0.15]  # radii of the directional modulus surfaces

# Solid model settings (part of the result key, so that changing them triggers new simulations)
mesh_refinement = 0.5
element_degree = 2

# Model name -> key of its results in the results file
models = {
    "solid": f"solid_P{element_degree}_curved_refinement_{mesh_refinement}",
    "beam": "beam",
    "beam_mod": "beam_mod",
}
model_order = ["solid", "beam", "beam_mod"]
quadrant_order = ["solid", "beam", "solid", "beam_mod"]  # quadrants of the surfaces (90 deg: cubic symmetry)
labels = {"solid": "Solid", "beam": "Beam", "beam_mod": "Beam penalized"}
colors = {"solid": "tab:gray", "beam": "crimson", "beam_mod": "darkviolet"}
markers = {"solid": "x", "beam": "o", "beam_mod": "*"}

save_dir = project_root / "data" / "outputs" / "homogenization_comparaison"
figure_dir = save_dir / "models_comparison"
figure_dir.mkdir(parents=True, exist_ok=True)
results_path = save_dir / "results.json"


# =============================================================================
# Results file (same format as compare_homogenization_solid_beam.py)
# =============================================================================
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


def load_results() -> dict:
    if results_path.exists():
        with open(results_path, "r") as file:
            return json.load(file)
    return {}


def save_result(cell: str, model_key: str, radius_key: str, entry: dict) -> None:
    """Add one simulation to the results file (locked, read again and only completed)."""
    with open(results_path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = load_results()
        data.setdefault(cell, {}).setdefault(model_key, {})[radius_key] = entry
        tmp_path = results_path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp_path, "w") as file:
            json.dump(data, file, indent=2)
        tmp_path.replace(results_path)
        fcntl.flock(lock, fcntl.LOCK_UN)


def homogenize(name_file: str, model: str, radius: float) -> dict:
    """Homogenization of the cell with the given model and radius."""
    lattice = LatticeSim(name_file)
    if model == "beam_mod":
        lattice.penalization_strategy = "angle"
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


def to_engineering_voigt(stiffness) -> np.ndarray:
    """Stiffness matrix with tensor shear strains (C44 = 2G) -> engineering Voigt notation (C44 = G)."""
    C = np.array(stiffness, dtype=float)
    C[:, 3:] /= 2.0
    return 0.5 * (C + C.T)


def load_material(name: str) -> float:
    """Young's modulus of the material of the cell."""
    with open(project_root / "src" / "pyLatticeDesign" / "materials" / f"{name}.json", "r") as file:
        return json.load(file)["Young_modulus"]


# =============================================================================
# Directional Young's modulus
# =============================================================================
def directional_modulus_surface(C_eng: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Points E(n) n of the directional Young's modulus surface, 1 / E(n) = v^T S v with S = C^-1 (engineering Voigt)
    and v = (n1^2, n2^2, n3^2, n1 n2, n1 n3, n2 n3).
    """
    T, P = np.meshgrid(theta, phi, indexing="ij")
    n = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)])
    v = np.stack([n[0] ** 2, n[1] ** 2, n[2] ** 2, n[0] * n[1], n[0] * n[2], n[1] * n[2]])
    S = np.linalg.inv(C_eng)
    E_dir = 1.0 / np.einsum("i...,ij,j...->...", v, S, v)
    return E_dir * n


def plot_split_surface(ax, stiffness: dict, title: str, cmap=plt.cm.jet, elev: float = 35, azim: float = None):
    """
    Directional modulus surface cut in quadrants (around z), one model per quadrant (quadrant_order). The quadrants
    are rotated so that the stiffest direction of the reference (solid) model lies in the middle of a quadrant: the
    lobes are not cut. Without azim, the view faces the first solid quadrant, with the two beam models on its sides.
    """
    names = quadrant_order
    sector = 2 * np.pi / len(names)
    theta = np.linspace(0, np.pi, 91)

    phi_full = np.linspace(0, 2 * np.pi, 145)
    reference = directional_modulus_surface(stiffness["solid"], theta, phi_full)
    i_max = np.unravel_index(np.argmax(np.linalg.norm(reference, axis=0)), reference.shape[1:])
    phi_offset = phi_full[i_max[1]] % (np.pi / 2) - sector / 2

    # One single surface (sectors side by side along phi) so that matplotlib sorts all the faces together; the
    # jump between two sectors appears as a wall at their common border
    surfaces = []
    for i, name in enumerate(names):
        phi = phi_offset + np.linspace(i * sector, (i + 1) * sector, 46)
        surfaces.append(directional_modulus_surface(stiffness[name], theta, phi))
    X, Y, Z = np.concatenate(surfaces + [surfaces[0][:, :, :1]], axis=2)  # close the surface
    radius = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    E_max = radius.max()
    norm = plt.Normalize(vmin=0, vmax=E_max)
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(radius)), rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False)

    for i, name in enumerate(names):
        # Label below the middle of the sector (from the top view: next to the sector)
        phi_mid = phi_offset + (i + 0.5) * sector
        z_label = 0.0 if elev == 90 else -0.95 * E_max
        ax.text(0.95 * E_max * np.cos(phi_mid), 0.95 * E_max * np.sin(phi_mid), z_label, labels[name],
                ha="center", va="center", fontsize=7, fontweight="bold", color=colors[name], zorder=10)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    for set_lim in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        set_lim(-0.8 * E_max, 0.8 * E_max)
    if azim is None:
        azim = np.degrees(phi_offset + 0.5 * sector)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    return mappable


# =============================================================================
# Simulations and comparison
# =============================================================================
all_results = {}
for cell_name, name_file in cells.items():
    cell_key = get_cell_key(name_file)
    for model, key in models.items():
        for radius in cell_radii.get(cell_name, radii):
            radius_key = f"{radius:.4f}"
            if radius_key in load_results().get(cell_key, {}).get(key, {}):
                continue
            print(f"Homogenization {cell_name} {model} r = {radius_key}")
            try:
                save_result(cell_key, key, radius_key, homogenize(name_file, model, radius))
            except Exception as error:  # e.g. solid mesh failing for overlapping beams at high density
                print(f"  failed: {error}")
    all_results[cell_name] = load_results()[cell_key]

summary = {}
for cell_name, name_file in cells.items():
    parameters = open_lattice_parameters(name_file)
    E_s = load_material(parameters.get("simulation_parameters", {}).get("material", "VeroClear"))
    results = all_results[cell_name]

    summary[cell_name] = {"radii": [], "relative_density": [], **{name: [] for name in model_order}}
    for radius in cell_radii.get(cell_name, radii):
        radius_key = f"{radius:.4f}"
        if not all(radius_key in results.get(models[model], {}) for model in models):
            continue
        summary[cell_name]["radii"].append(float(radius))
        summary[cell_name]["relative_density"].append(results[models["solid"]][radius_key]["relative_density"])
        for model in models:
            summary[cell_name][model].append(to_engineering_voigt(results[models[model]][radius_key]["stiffness"]))
    summary[cell_name]["E_s"] = E_s


def relative_error(C: np.ndarray, C_ref: np.ndarray) -> float:
    return np.linalg.norm(C - C_ref, "fro") / np.linalg.norm(C_ref, "fro")


# Table: C11, C12, C44 and relative error ||C - C_ref||_F / ||C_ref||_F
table = {}
for cell_name, data in summary.items():
    print(f"\n=== {cell_name} ===")
    print(f"{'r':>6s} {'rho':>7s} {'model':>15s} {'C11':>10s} {'C12':>10s} {'C44':>10s} {'err/solid':>10s}")
    table[cell_name] = []
    for i, radius in enumerate(data["radii"]):
        for name in model_order:
            C = data[name][i]
            err_solid = relative_error(C, data["solid"][i])
            print(f"{radius:6.3f} {data['relative_density'][i]:7.4f} {labels[name]:>15s} {C[0, 0]:10.4f} "
                  f"{C[0, 1]:10.4f} {C[3, 3]:10.4f} {100 * err_solid:9.3f}%")
            table[cell_name].append({"radius": radius, "relative_density": data["relative_density"][i],
                                     "model": name, "C11": C[0, 0], "C12": C[0, 1], "C44": C[3, 3],
                                     "error_vs_solid": err_solid})
with open(figure_dir / "comparison_solid_beam.json", "w") as file:
    json.dump(table, file, indent=2, default=float)


# Figure 1: directional modulus surfaces cut in sectors (rows: cells, columns: radii)
fig = plt.figure(figsize=(4.2 * len(radii_surface), 3.8 * len(cells)))
for row, (cell_name, data) in enumerate(summary.items()):
    for col, radius in enumerate(radii_surface):
        if radius not in data["radii"]:
            continue
        i = data["radii"].index(radius)
        stiffness = {name: data[name][i] for name in model_order}
        ax = fig.add_subplot(len(cells), len(radii_surface), row * len(radii_surface) + col + 1, projection="3d")
        mappable = plot_split_surface(ax, stiffness, f"{cell_name}, r = {radius:.3f}, "
                                                     rf"$\rho$ = {data['relative_density'][i]:.3f}")
        colorbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02)
        colorbar.set_label("Directional stiffness [MPa]", fontsize=7)
        colorbar.ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(figure_dir / "directional_modulus_split.png", dpi=300, bbox_inches="tight")

# One figure per cell and radius (for the article)
for cell_name, data in summary.items():
    for radius in radii_surface:
        if radius not in data["radii"]:
            continue
        i = data["radii"].index(radius)
        fig_cell = plt.figure(figsize=(9, 4.2))
        stiffness = {name: data[name][i] for name in model_order}
        for k, (elev, azim, view) in enumerate([(35, None, "3D view"), (90, -90, "top view")]):
            ax = fig_cell.add_subplot(1, 2, k + 1, projection="3d")
            mappable = plot_split_surface(ax, stiffness, f"{cell_name}, r = {radius:.3f}, "
                                                         rf"$\rho$ = {data['relative_density'][i]:.3f} ({view})",
                                          elev=elev, azim=azim)
        fig_cell.colorbar(mappable, ax=fig_cell.axes, shrink=0.7).set_label("Directional stiffness [MPa]")
        fig_cell.savefig(figure_dir / f"directional_modulus_split_{cell_name}_r{radius:.3f}.png", dpi=300,
                         bbox_inches="tight")
        plt.close(fig_cell)


# Figure 2: sections of the directional modulus surface (all models superimposed)
#   - plane (x, y): theta = 90 deg
#   - diagonal plane containing z and [110]: phi = 45 deg (contains <100>, <110> and <111>)
fig, axes = plt.subplots(len(radii_surface), 2 * len(cells), figsize=(3.2 * 2 * len(cells), 3.4 * len(radii_surface)),
                         subplot_kw={"projection": "polar"}, squeeze=False)
angles = np.linspace(0, 2 * np.pi, 361)
for col, (cell_name, data) in enumerate(summary.items()):
    for row, radius in enumerate(radii_surface):
        if radius not in data["radii"]:
            continue
        i = data["radii"].index(radius)
        for k, plane in enumerate(["xy", "diag"]):
            ax = axes[row, 2 * col + k]
            for name in model_order:
                if plane == "xy":
                    points = directional_modulus_surface(data[name][i], np.array([np.pi / 2]), angles)[:, 0, :]
                else:  # angle alpha from [110] towards z: n = cos(alpha) [110] / sqrt(2) + sin(alpha) z
                    points = directional_modulus_surface(data[name][i], np.pi / 2 - angles,
                                                         np.array([np.pi / 4]))[:, :, 0]
                ax.plot(angles, np.linalg.norm(points, axis=0), color=colors[name], label=labels[name], linewidth=1.2)
            ax.set_title(f"{cell_name} r = {radius:.3f}\n" + ("plane (x, y)" if plane == "xy" else
                                                              "plane ([110], z)"), fontsize=8)
            ax.tick_params(labelsize=6)
            if plane == "diag":
                ax.set_xticks(np.deg2rad([0, 35.26, 90, 180]), ["[110]", "[111]", "[001]", ""], fontsize=6)
axes[0, 0].legend(fontsize=6, loc="lower left", bbox_to_anchor=(-0.3, -0.25))
fig.suptitle("Directional stiffness [MPa]")
fig.tight_layout()
fig.savefig(figure_dir / "directional_modulus_sections.png", dpi=200)


# Figure 3: Ex / Es, Gxy / Es and error against the solid model as a function of the relative density
fig, axes = plt.subplots(3, len(cells), figsize=(4 * len(cells), 10), squeeze=False)
for col, (cell_name, data) in enumerate(summary.items()):
    rho = np.array(data["relative_density"])
    for name in model_order:
        style = {"color": colors[name], "marker": markers[name], "label": labels[name]}
        Ex = np.array([1 / np.linalg.inv(C)[0, 0] for C in data[name]]) / data["E_s"]
        Gxy = np.array([C[3, 3] for C in data[name]]) / data["E_s"]
        axes[0, col].plot(rho, Ex, **style)
        axes[1, col].plot(rho, Gxy, **style)
        if name != "solid":
            errors = [relative_error(C, C_solid) for C, C_solid in zip(data[name], data["solid"])]
            axes[2, col].plot(rho, 100 * np.array(errors), **style)
    axes[0, col].set_title(cell_name)
    axes[0, col].set_ylabel(r"$E_x / E_s$")
    axes[1, col].set_ylabel(r"$G_{xy} / E_s$")
    axes[2, col].set_ylabel(r"$\|C - C^{solid}\|_F / \|C^{solid}\|_F$ (%)")
    for ax in axes[:2, col]:
        ax.set_yscale("log")
    for ax in axes[:, col]:
        ax.set_xscale("log")
        ax.set_xlabel("Relative density")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([0.01, 0.05, 0.1, 0.3, 0.5])
        ax.grid(alpha=0.3)
    axes[0, col].legend(fontsize=8)
fig.tight_layout()
fig.savefig(figure_dir / "effective_properties_vs_density.png", dpi=200)
print("\nFigures saved to", figure_dir)
plt.show()
