"""
Effect of the surface roughness on the mechanical response of a BCC lattice: virtual compression test on rough
meshes generated with Pyrough, compared with smooth meshes.

Pipeline (the same for the smooth and the rough samples, so that only the roughness differs):
1. Pyrough generates the surface mesh of every beam (cylindrical wire, self-affine roughness of RMS Sq and Hurst
   exponent eta), and the beams are fused (Lattice.generate_mesh_lattice_rough). The smooth samples are generated
   with Sq = 0.
2. The mesh is cut by the flat bounding box of the lattice (planar faces in contact with the platens), remeshed
   isotropically and meshed with tetrahedra (SolidModel with a surface mesh file).
3. Compression between bonded rigid platens (Zmin fixed, Zmax displaced along -z, see BCC_rough_compression.json),
   solved in linear elasticity with quadratic elements.

Outputs for each sample: relative density, apparent Young's modulus E* = (F / A) / (delta / H), and the 99th
percentile of the von Mises stress normalized by the macroscopic stress (stress concentrations at the roughness, the
band of elements next to the platens is excluded).

Pyrough adds the roughness outward (the surface is moved by z - min(z) >= 0): the rough struts are thicker than the
nominal ones. The rough samples are then compared with the smooth samples:
- at the same nominal radius (as-designed lattice),
- at the same relative density (smooth samples generated on a radius sweep): stiffness lost because of the roughness
  itself, and not because of the change of volume. The equivalent smooth radius with the same stiffness is also given.

Requirements: Pyrough (branch pyLattice_comp_V2, https://github.com/jamodeo12/Pyrough), its root folder being given by
the environment variable PYROUGH_PATH (or pyLatticeDSO installed as a submodule of Pyrough), manifold3d, pymeshlab,
trimesh.

Results are saved in data/outputs/roughness_compression (results.json is completed on each run, so the script can be
stopped and run again; delete it to restart from scratch).
"""
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

if os.environ.get("PYROUGH_PATH"):
    sys.path.insert(0, os.environ["PYROUGH_PATH"])

from pyLatticeDesign.materials import MatProperties
from pyLatticeSim.export_simulation_results import exportSimulationResults
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.solid_lattice_simulation import SolidLatticeSimulation
from pyLatticeSim.solid_model import SolidModel

project_root = Path(__file__).resolve().parents[2]

# Lattice and loading (cell 200 x 200 x 200, nominal strut radius 20)
name_file = "Pyrough/BCC_rough_compression"

# Roughness (Pyrough parameters of the wires, the RMS is given and C1 is computed by Pyrough)
sq_over_radius = [0.025, 0.05, 0.1, 0.15]  # RMS roughness Sq / nominal radius
n_seeds = 3  # realizations of the random roughness for each amplitude
roughness_parameters = {"beam_type": "cwire", "eta": 0.8, "N": 30, "M": 30, "Mesh_size": 5, "Refine_factor": 0.75,
                        "Raw_stl": "na"}

# Smooth samples: radius sweep for the comparison at the same relative density
radii_smooth_over_radius = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# Solid model
element_degree = 2
platen_band = 0.05  # elements closer to the platens than platen_band * height are excluded from the stress statistics

save_dir = project_root / "data" / "outputs" / "roughness_compression"
mesh_dir = save_dir / "meshes"
mesh_dir.mkdir(parents=True, exist_ok=True)
results_path = save_dir / "results.json"
parameters_file = "_roughness_parameters_tmp.json"  # Pyrough input written for each sample
parameters_path = project_root / "data" / "inputs" / "preset_lattice" / "Pyrough" / parameters_file


# =============================================================================
# Mesh generation
# =============================================================================
def generate_surface_mesh(radius: float, sq: float, seed: int, name: str) -> Path:
    """
    Rough (or smooth, sq = 0) surface mesh of the lattice with Pyrough, cut by the flat bounding box, remeshed
    isotropically, with the vertices of the cut faces exactly on the planes of the box.
    """
    import manifold3d
    import pymeshlab
    import trimesh

    with open(parameters_path, "w") as file:
        json.dump({"lattice": dict(roughness_parameters, RMS=float(sq)), "trim_boundary_beams": False}, file,
                  indent=2)
    lattice = LatticeSim(name_file)
    lattice.reset_cell_with_new_radii([float(radius)])
    np.random.seed(seed)  # Pyrough draws the roughness with the global numpy generator
    lattice.generate_mesh_lattice_rough(parameters_file, name, print_volume=False, cut_mesh_at_boundary=False)
    parameters_path.unlink()
    mesh = trimesh.load(project_root / "data" / "outputs" / "Pyrough" / f"{name}.stl")

    # Cut by the flat bounding box
    mins = np.array([lattice.x_min, lattice.y_min, lattice.z_min], dtype=float)
    maxs = np.array([lattice.x_max, lattice.y_max, lattice.z_max], dtype=float)
    solid = manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                                                tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    solid = solid ^ manifold3d.Manifold.cube(tuple(maxs - mins)).translate(tuple(mins))
    cut = solid.to_mesh()

    # Put back on the planes the vertices of the cut faces (float32 precision of manifold3d)
    vertices = np.asarray(cut.vert_properties[:, :3], dtype=np.float64)
    tol = 1e-5 * float(np.max(maxs - mins))
    for axis in range(3):
        for value in (mins[axis], maxs[axis]):
            vertices[np.abs(vertices[:, axis] - value) < tol, axis] = value

    # The cut leaves slivers on the planar faces, that make the tetrahedral mesher fail: isotropic remeshing (the
    # vertices are projected on the input surface, so that the cut faces stay planar)
    target = float(np.median(mesh.edges_unique_length))
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=np.asarray(cut.tri_verts, dtype=np.int32)))
    meshset.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(target), iterations=5, featuredeg=30)

    # The union of the Pyrough wires leaves a few self-intersecting faces and needle triangles at the junctions
    # (rejected by the tetrahedral mesher): needles are removed by edge flips and collapses, self-intersecting faces
    # with their neighbours, and the holes are closed
    for _ in range(10):
        meshset.meshing_remove_t_vertices()
        meshset.meshing_remove_folded_faces()
        meshset.meshing_remove_null_faces()
        meshset.meshing_remove_unreferenced_vertices()
        meshset.meshing_repair_non_manifold_edges()  # non manifold edges and holes left by the collapses
        meshset.meshing_repair_non_manifold_vertices()
        meshset.meshing_close_holes(maxholesize=1000)
        meshset.compute_selection_by_self_intersections_per_face()
        if meshset.current_mesh().selected_face_number() == 0:
            break
        meshset.apply_selection_dilatation()
        meshset.meshing_remove_selected_faces()
    else:
        raise RuntimeError(f"Self-intersections of the surface mesh of {name} could not be removed.")
    remeshed = meshset.current_mesh()

    # Vertices at the border of the cut faces are slightly moved by the remeshing (~1e-8 L): put back on the planes
    vertices = remeshed.vertex_matrix().copy()
    for axis in range(3):
        for value in (mins[axis], maxs[axis]):
            vertices[np.abs(vertices[:, axis] - value) < tol, axis] = value
    surface = trimesh.Trimesh(vertices=vertices, faces=remeshed.face_matrix(), process=False)
    if not surface.is_watertight or np.any(surface.area_faces < 1e-12 * target ** 2):
        raise RuntimeError(f"The surface mesh of {name} is not closed or has degenerate faces.")
    path = mesh_dir / f"{name}.stl"
    surface.export(path)
    return path


# =============================================================================
# Compression test
# =============================================================================
def compression_test(surface_mesh: Path, name: str, export: bool = False) -> dict:
    """Compression of the meshed sample, apparent modulus and stress statistics."""
    import ufl
    from dolfinx import fem

    start = time.time()
    lattice = LatticeSim(name_file)
    mins = np.array([lattice.x_min, lattice.y_min, lattice.z_min], dtype=float)
    maxs = np.array([lattice.x_max, lattice.y_max, lattice.z_max], dtype=float)
    lengths = maxs - mins
    solid_model = SolidModel(MPI.COMM_SELF, lattice, surface_mesh_file=str(surface_mesh), verbose=0)
    simulation = SolidLatticeSimulation(solid_model, element_degree=element_degree, verbose=0)
    simulation.apply_boundary_conditions_from_lattice()
    simulation.solve_problem(solver="iterative", rtol=1e-10)  # direct solver: too much memory for P2 elements

    displacement = lattice.boundary_conditions["Displacement"]["Compression"]["Value"][2]
    force = simulation.compute_reaction_forces()["Compression"][2]
    macro_stress = force / (lengths[0] * lengths[1])
    macro_strain = displacement / lengths[2]

    # Volume weighted von Mises stress far from the platens
    von_mises = simulation.compute_von_mises_stress()
    W = von_mises.function_space
    cell_volumes = fem.Function(W)
    cell_volumes.interpolate(fem.Expression(ufl.CellVolume(solid_model.domain), W.element.interpolation_points()))
    centers = fem.Function(fem.functionspace(solid_model.domain, ("DG", 0, (3,))))
    centers.interpolate(lambda x: x)
    z = centers.x.array.reshape(-1, 3)[:, 2]
    keep = (z > mins[2] + platen_band * lengths[2]) & (z < maxs[2] - platen_band * lengths[2])
    stress, weights = von_mises.x.array[keep], cell_volumes.x.array[keep]
    order = np.argsort(stress)
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    stress_p99 = float(stress[order][np.searchsorted(cumulative, 0.99)])

    if export:
        exporter = exportSimulationResults(simulation, f"roughness_compression/{name}")
        exporter.export_solid_displacement()
        exporter.export_von_mises()
        exporter.export_finalize()

    return {"relative_density": float(solid_model.get_volume() / np.prod(lengths)),
            "E": float(macro_stress / macro_strain),
            "von_mises_p99": float(stress_p99 / abs(macro_stress)),
            "n_tetrahedra": int(solid_model.domain.topology.index_map(3).size_global),
            "time": time.time() - start}


def load_results() -> dict:
    if results_path.exists():
        with open(results_path, "r") as file:
            return json.load(file)
    return {"smooth": {}, "rough": {}}


def save_results(results: dict) -> None:
    with open(results_path, "w") as file:
        json.dump(results, file, indent=2)


results = load_results()
radius = LatticeSim(name_file).radii[0]

samples = [("smooth", f"{ratio:.3f}", ratio * radius, 0.0, 0) for ratio in radii_smooth_over_radius]
samples += [("rough", f"{ratio:.3f}_seed{seed}", radius, ratio * radius, seed)
            for ratio in sq_over_radius for seed in range(n_seeds)]
for kind, key, r, sq, seed in samples:
    if key in results[kind]:
        continue
    name = f"BCC_{kind}_{key}"
    start = time.time()
    surface_mesh = generate_surface_mesh(r, sq, seed, name)
    entry = compression_test(surface_mesh, name, export=(seed == 0 and kind == "rough") or
                             (kind == "smooth" and np.isclose(r, radius)))
    entry.update({"radius": r, "Sq": sq, "seed": seed, "total_time": time.time() - start})
    results[kind][key] = entry
    save_results(results)
    print(f"{name}: rho = {entry['relative_density']:.4f}, E* = {entry['E']:.4g}, "
          f"sigma_VM,99% / sigma = {entry['von_mises_p99']:.3f} ({entry['total_time']:.0f} s)")


# =============================================================================
# Post-processing
# =============================================================================
E_s = float(MatProperties(LatticeSim(name_file).material_name).young_modulus)

smooth = sorted(results["smooth"].values(), key=lambda e: e["radius"])
r_smooth = np.array([e["radius"] for e in smooth])
rho_smooth = np.array([e["relative_density"] for e in smooth])
E_smooth = np.array([e["E"] for e in smooth]) / E_s
vm_smooth = np.array([e["von_mises_p99"] for e in smooth])
nominal = next(e for e in smooth if np.isclose(e["radius"], radius))

# Smooth samples: E*(rho) fitted in log-log (quadratic), E*(r) interpolated for the equivalent radius
fit_E = np.polyfit(np.log(rho_smooth), np.log(E_smooth), 2)
fit_vm = np.polyfit(np.log(rho_smooth), np.log(vm_smooth), 2)


def E_smooth_at_density(rho):
    return np.exp(np.polyval(fit_E, np.log(rho)))


def vm_smooth_at_density(rho):
    return np.exp(np.polyval(fit_vm, np.log(rho)))


summary = {0.0: {"rho": np.array([nominal["relative_density"]]), "E": np.array([nominal["E"] / E_s]),
                 "vm": np.array([nominal["von_mises_p99"]])}}
for ratio in sq_over_radius:
    entries = [e for e in results["rough"].values() if np.isclose(e["Sq"], ratio * radius)]
    if entries:
        summary[ratio] = {"rho": np.array([e["relative_density"] for e in entries]),
                          "E": np.array([e["E"] for e in entries]) / E_s,
                          "vm": np.array([e["von_mises_p99"] for e in entries])}
for ratio, s in summary.items():
    s["rho_nominal"] = s["rho"] / nominal["relative_density"]
    s["E_nominal"] = s["E"] / (nominal["E"] / E_s)
    s["E_same_density"] = s["E"] / E_smooth_at_density(s["rho"])
    s["vm_nominal"] = s["vm"] / nominal["von_mises_p99"]
    s["vm_same_density"] = s["vm"] / vm_smooth_at_density(s["rho"])
    s["r_eq_stiffness"] = np.interp(s["E"], E_smooth, r_smooth) / radius
    s["r_eq_volume"] = np.interp(s["rho"], rho_smooth, r_smooth) / radius

columns = {"rho_nominal": "rho/rho0", "E_nominal": "E/E0", "E_same_density": "E/E_smooth(rho)",
           "r_eq_volume": "r_eq,vol/r", "r_eq_stiffness": "r_eq,E/r", "vm_nominal": "VM99/VM99_0",
           "vm_same_density": "VM99/VM99_smooth(rho)"}
print("\nSq/r   " + "".join(f"{label:<19}" for label in columns.values()))
for ratio, s in summary.items():
    print(f"{ratio:<6.3f} " + "".join(f"{s[k].mean():.3f} +- {s[k].std():.3f}   " for k in columns))
print("(mean +- standard deviation over the realizations; E/E_smooth(rho): smooth sample with the same density)")

# Figure 1: apparent modulus against the relative density
cmap = plt.cm.viridis
max_ratio = max(sq_over_radius)
fig, ax = plt.subplots(figsize=(6, 4.5))
rho_plot = np.linspace(rho_smooth.min(), rho_smooth.max(), 100)
ax.plot(rho_plot, E_smooth_at_density(rho_plot), "-", color="tab:gray", label="Smooth (radius sweep)")
ax.plot(rho_smooth, E_smooth, "o", color="tab:gray", markersize=4)
ax.plot(nominal["relative_density"], nominal["E"] / E_s, "o", color="k", label="Smooth, nominal radius")
for ratio, s in summary.items():
    if ratio > 0:
        ax.plot(s["rho"], s["E"], "D", color=cmap(ratio / max_ratio), markeredgecolor="k",
                label=fr"Rough $S_q / r$ = {ratio:g}")
ax.set_xlabel(r"Relative density $\rho^*$")
ax.set_ylabel(r"Apparent modulus $E^* / E_s$")
ax.set_title("BCC in compression: smooth vs rough struts")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(save_dir / "roughness_modulus_vs_density.png", dpi=300)

# Figure 2: ratios rough / smooth against the roughness amplitude
ratios = np.array(list(summary))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for key, style, label in (("rho_nominal", ":", r"$\rho^* / \rho^*_0$ (same nominal radius)"),
                          ("E_nominal", "o-", r"$E^* / E^*_0$ (same nominal radius)"),
                          ("E_same_density", "s--", r"$E^* / E^*_{smooth}(\rho^*)$ (same density)")):
    mean = np.array([summary[r][key].mean() for r in ratios])
    std = np.array([summary[r][key].std() for r in ratios])
    axes[0].errorbar(ratios, mean, yerr=std, fmt=style, capsize=3, label=label)
axes[0].axhline(1.0, color="k", linewidth=0.8)
axes[0].set_ylabel("Rough / smooth")
axes[0].set_title("Density and stiffness")
for key, style, label in (("r_eq_volume", "o-", "Same volume"), ("r_eq_stiffness", "s--", "Same stiffness")):
    mean = np.array([summary[r][key].mean() for r in ratios])
    std = np.array([summary[r][key].std() for r in ratios])
    axes[1].errorbar(ratios, mean, yerr=std, fmt=style, capsize=3, label=label)
axes[1].axhline(1.0, color="k", linewidth=0.8)
axes[1].set_ylabel(r"Equivalent smooth radius $r_{eq} / r$")
axes[1].set_title("Equivalent smooth strut")
for ax in axes:
    ax.set_xlabel(r"RMS roughness $S_q / r$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(save_dir / "roughness_knockdown.png", dpi=300)

# Figure 3: stress concentrations
fig, ax = plt.subplots(figsize=(6, 4.5))
for key, style, label in (("vm_nominal", "o-", "Same nominal radius"), ("vm_same_density", "s--", "Same density")):
    mean = np.array([summary[r][key].mean() for r in ratios])
    std = np.array([summary[r][key].std() for r in ratios])
    ax.errorbar(ratios, mean, yerr=std, fmt=style, capsize=3, label=label)
ax.axhline(1.0, color="k", linewidth=0.8)
ax.set_xlabel(r"RMS roughness $S_q / r$")
ax.set_ylabel(r"$\sigma_{VM, 99\%}$ rough / smooth (same macroscopic stress)")
ax.set_title("Stress concentrations at the rough surface")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(save_dir / "roughness_stress_concentration.png", dpi=300)

print(f"\nFigures and results saved in {save_dir}")
plt.show()
