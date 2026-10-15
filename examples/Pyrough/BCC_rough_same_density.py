"""
Effect of the surface roughness on the mechanical response of a BCC cell at constant relative density: rough cells
generated with Pyrough with random roughness parameters, all with the relative density of the smooth reference cell,
loaded in compression and in shear (3D solid model).

Pipeline (the same for the smooth and the rough cells, so that only the roughness differs):
1. Pyrough generates the surface mesh of every beam (cylindrical wire with a self-affine roughness of RMS Sq, Hurst
   exponent eta and N x M Fourier modes) and the beams are fused (Lattice.generate_mesh_lattice_rough).
2. The mesh is cut by the flat bounding box of the cell, cleaned and meshed with tetrahedra (SolidModel with a surface
   mesh file).
3. Pyrough adds the roughness outward (the surface is moved by z - min(z) >= 0): the strut radius of each rough cell
   is decreased (Newton iterations on the volume of the surface mesh) until its relative density matches the one of
   the smooth cell. The same random seed is used at each iteration, so that mainly the radius changes.
4. Two loading cases between bonded rigid platens, Zmin fixed: compression (Zmax displaced along -z, apparent modulus
   E*) and shear (Zmax displaced along x, apparent shear modulus G*), linear elasticity with quadratic elements.

The residual difference of density (< max_density_error) is corrected with the local power law of the smooth cells
(radii r0 and r0 (1 +- 5%)). The figure shows, for each rough cell, the change of E* and G* relative to the smooth
cell, with an image of the rough cell above its data points.

Requirements: Pyrough (branch pyLattice_comp_V2, https://github.com/jamodeo12/Pyrough), its root folder being given by
the environment variable PYROUGH_PATH, manifold3d, pymeshlab, trimesh.

Results are saved in data/outputs/roughness_same_density (results.json is completed on each run, so the script can be
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

from pyLatticeDesign.lattice import Lattice
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.solid_lattice_simulation import SolidLatticeSimulation
from pyLatticeSim.solid_model import SolidModel

project_root = Path(__file__).resolve().parents[2]

# Lattice (cell 200 x 200 x 200, nominal strut radius 20)
name_file = "Pyrough/BCC_rough_compression"

# Random roughness parameters of the rough cells
n_samples = 8
random_seed = 2026
sq_over_radius_range = (0.025, 0.15)  # RMS roughness Sq / smooth radius
eta_range = (0.3, 1.0)  # Hurst exponent
n_modes_choices = (20, 30, 40)  # number of Fourier modes N = M
fixed_parameters = {"beam_type": "cwire", "Mesh_size": 5, "Refine_factor": 0.75, "Raw_stl": "na"}

# Same density
density_tolerance = 3e-3  # relative difference of density to stop the iterations
max_density_error = 1.5e-2  # largest relative difference of density accepted (corrected by the smooth power law)
max_newton_iterations = 6

# Loading: imposed displacement of the Zmax face (1% of the cell height)
loading_cases = {"compression": (2, -1.0), "shear": (0, 1.0)}  # (displaced component, sign)
strain = 0.01

element_degree = 2

save_dir = project_root / "data" / "outputs" / "roughness_same_density"
mesh_dir = save_dir / "meshes"
mesh_dir.mkdir(parents=True, exist_ok=True)
results_path = save_dir / "results.json"
parameters_file = "_roughness_parameters_tmp.json"  # Pyrough input written for each sample
parameters_path = project_root / "data" / "inputs" / "preset_lattice" / "Pyrough" / parameters_file

lattice_reference = LatticeSim(name_file)
radius_smooth = float(lattice_reference.radii[0])
mins = np.array([lattice_reference.x_min, lattice_reference.y_min, lattice_reference.z_min], dtype=float)
maxs = np.array([lattice_reference.x_max, lattice_reference.y_max, lattice_reference.z_max], dtype=float)
lengths = maxs - mins


# =============================================================================
# Mesh generation
# =============================================================================
def generate_surface_mesh(radius: float, roughness: dict, seed: int, name: str) -> Path:
    """
    Rough (or smooth, RMS = 0) surface mesh of the lattice with Pyrough, cut by the flat bounding box, remeshed
    isotropically and cleaned, with the vertices of the cut faces exactly on the planes of the box.
    """
    import manifold3d
    import pymeshlab
    import trimesh

    with open(parameters_path, "w") as file:
        json.dump({"lattice": dict(fixed_parameters, **roughness), "trim_boundary_beams": False}, file, indent=2)
    # Design lattice: one wire per strut (the simulation lattice splits the struts into segments at the nodes, each
    # segment would get its own roughness)
    lattice = Lattice(name_file + ".json")
    for beam in lattice.beams:
        beam.radius = float(radius)
    lattice.generate_mesh_lattice_rough(parameters_file, name, print_volume=False, cut_mesh_at_boundary=False,
                                        seed=seed)
    parameters_path.unlink()
    mesh = trimesh.load(project_root / "data" / "outputs" / "Pyrough" / f"{name}.stl")

    # Cut by the flat bounding box
    solid = manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                                                tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    solid = solid ^ manifold3d.Manifold.cube(tuple(lengths)).translate(tuple(mins))
    cut = solid.to_mesh()

    # Put back on the planes the vertices of the cut faces (float32 precision of manifold3d)
    vertices = np.asarray(cut.vert_properties[:, :3], dtype=np.float64)
    tol = 1e-5 * float(np.max(lengths))
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


def relative_density(surface_mesh: Path) -> float:
    import trimesh
    return float(trimesh.load(surface_mesh).volume / np.prod(lengths))


def generate_at_density(roughness: dict, seed: int, name: str, target: float, slope: float) -> tuple[Path, float]:
    """
    Rough surface mesh with the target relative density: Newton iterations on the strut radius, with the derivative
    d ln(rho) / d ln(r) of the smooth cells. Pyrough does not reproduce exactly the same random surface when the
    radius changes exactly (meshes of different sizes), so that the density is a slightly noisy function of the radius:
    the iteration closest to the target is kept if it is within max_density_error.
    """
    # Initial guess: Pyrough moves the surface outward by about 2 Sq on average
    radius = radius_smooth - 2.0 * roughness["RMS"]
    best = None
    for _ in range(max_newton_iterations):
        path = generate_surface_mesh(radius, roughness, seed, f"{name}_iteration")
        rho = relative_density(path)
        print(f"  {name}: r = {radius:.3f}, rho = {rho:.5f} (target {target:.5f})")
        if best is None or abs(rho / target - 1.0) < abs(best[1] / target - 1.0):
            best = (radius, rho)
            path.replace(mesh_dir / f"{name}.stl")
        if abs(rho / target - 1.0) < density_tolerance:
            break
        radius *= (target / rho) ** (1.0 / slope)
    if abs(best[1] / target - 1.0) > max_density_error:
        raise RuntimeError(f"The density of {name} did not converge.")
    return mesh_dir / f"{name}.stl", best[0]


# =============================================================================
# Loading cases
# =============================================================================
def mechanical_tests(surface_mesh: Path) -> dict:
    """Apparent moduli in compression and in shear of the meshed cell, and its relative density."""
    start = time.time()
    lattice = LatticeSim(name_file)
    solid_model = SolidModel(MPI.COMM_SELF, lattice, surface_mesh_file=str(surface_mesh), verbose=0)
    entry = {"relative_density": float(solid_model.get_volume() / np.prod(lengths)),
             "n_tetrahedra": int(solid_model.domain.topology.index_map(3).size_global)}
    for case, (component, sign) in loading_cases.items():
        displacement = [0.0, 0.0, 0.0]
        displacement[component] = sign * strain * lengths[2]
        lattice.boundary_conditions = {"Displacement": {
            "Fixed": {"Surface": ["Zmin"], "DOF": ["X", "Y", "Z"], "Value": [0.0, 0.0, 0.0]},
            "Load": {"Surface": ["Zmax"], "DOF": ["X", "Y", "Z"], "Value": displacement}}}
        simulation = SolidLatticeSimulation(solid_model, element_degree=element_degree, verbose=0)
        simulation.apply_boundary_conditions_from_lattice()
        simulation.solve_problem(solver="iterative", rtol=1e-10)  # direct solver: too much memory for P2 elements
        force = simulation.compute_reaction_forces()["Load"][component]
        entry[case] = float((force / (lengths[0] * lengths[1])) / (sign * strain))
    entry["time"] = time.time() - start
    return entry


def load_results() -> dict:
    if results_path.exists():
        with open(results_path, "r") as file:
            return json.load(file)
    return {"smooth": {}, "rough": {}}


def save_results(results: dict) -> None:
    with open(results_path, "w") as file:
        json.dump(results, file, indent=2)


results = load_results()

# Smooth cells: reference and neighbours for the local power laws
for factor in (0.95, 1.0, 1.05):
    key = f"{factor:.2f}"
    if key in results["smooth"]:
        continue
    path = generate_surface_mesh(factor * radius_smooth, dict(RMS=0.0, eta=0.8, N=30, M=30), 0, f"smooth_{key}")
    results["smooth"][key] = dict(mechanical_tests(path), radius=factor * radius_smooth)
    save_results(results)
    print(f"Smooth r = {factor * radius_smooth:.2f}: {results['smooth'][key]}")

smooth = [results["smooth"][f"{f:.2f}"] for f in (0.95, 1.0, 1.05)]
reference = smooth[1]
log_rho = np.log([e["relative_density"] for e in smooth])
slope_density = np.polyfit(np.log([e["radius"] for e in smooth]), log_rho, 1)[0]
slopes = {case: np.polyfit(log_rho, np.log([e[case] for e in smooth]), 1)[0] for case in loading_cases}

# Rough cells with random parameters at the density of the smooth cell
rng = np.random.default_rng(random_seed)
for sample in range(n_samples):
    n_modes = int(rng.choice(n_modes_choices))
    roughness = {"RMS": float(rng.uniform(*sq_over_radius_range) * radius_smooth),
                 "eta": float(rng.uniform(*eta_range)), "N": n_modes, "M": n_modes}
    seed = int(rng.integers(0, 2 ** 31))
    key = f"sample_{sample}"
    if key in results["rough"]:
        continue
    start = time.time()
    path, radius = generate_at_density(roughness, seed, key, reference["relative_density"], slope_density)
    entry = mechanical_tests(path)
    entry.update({"radius": radius, "seed": seed, "Sq_over_radius": roughness["RMS"] / radius_smooth,
                  "eta": roughness["eta"], "N": n_modes, "total_time": time.time() - start})
    results["rough"][key] = entry
    save_results(results)
    print(f"{key}: Sq/r = {entry['Sq_over_radius']:.3f}, eta = {entry['eta']:.2f}, N = {n_modes}, "
          f"rho = {entry['relative_density']:.5f}, E* = {entry['compression']:.4g}, G* = {entry['shear']:.4g} "
          f"({entry['total_time']:.0f} s)")


# =============================================================================
# Post-processing
# =============================================================================
def change_vs_smooth(entry: dict, case: str) -> float:
    """Relative change (%) of a modulus against the smooth cell, corrected for the residual change of density."""
    smooth_value = reference[case] * (entry["relative_density"] / reference["relative_density"]) ** slopes[case]
    return 100.0 * (entry[case] / smooth_value - 1.0)


samples = sorted(results["rough"].items(), key=lambda item: item[1]["Sq_over_radius"])
changes = {case: np.array([change_vs_smooth(e, case) for _, e in samples]) for case in loading_cases}

print(f"\nSmooth cell: rho = {reference['relative_density']:.4f}, E* = {reference['compression']:.4g}, "
      f"G* = {reference['shear']:.4g}")
print("sample    Sq/r    eta   N   r/r0    rho/rho0   dE* (%)  dG* (%)")
for (key, e), dE, dG in zip(samples, changes["compression"], changes["shear"]):
    print(f"{key:<9} {e['Sq_over_radius']:.3f}  {e['eta']:.2f}  {e['N']:<3} {e['radius'] / radius_smooth:.3f}  "
          f"{e['relative_density'] / reference['relative_density']:.4f}     {dE:+.2f}    {dG:+.2f}")
for case in loading_cases:
    print(f"{case}: mean {changes[case].mean():+.2f} %, standard deviation {changes[case].std():.2f} %")


def render_cell(path: Path, image_path: Path, limit: float = 0.3) -> None:
    """Image of a cell, the strut surface colored by its distance to the axes relative to the smooth radius."""
    import trimesh
    from matplotlib.colors import LightSource, to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = trimesh.load(path)
    A = np.array([beam.point1.coordinates for beam in lattice_reference.beams], dtype=float)
    B = np.array([beam.point2.coordinates for beam in lattice_reference.beams], dtype=float)
    centers = mesh.triangles_center
    AB = B - A
    t = np.clip(np.einsum("pbj,bj->pb", centers[:, None] - A[None], AB) / np.sum(AB ** 2, axis=1)[None], 0, 1)
    distance = np.linalg.norm(centers[:, None] - (A[None] + t[..., None] * AB[None]), axis=2).min(axis=1)
    deviation = (distance - radius_smooth) / radius_smooth
    on_cut = np.any(np.isclose(centers, mins[None], atol=1e-3) | np.isclose(centers, maxs[None], atol=1e-3), axis=1)
    colors = plt.cm.RdBu_r(np.clip(0.5 + 0.5 * deviation / limit, 0, 1))[:, :3]
    colors[on_cut] = to_rgb("#b9c2cc")
    shading = 0.45 + 0.55 * LightSource(azdeg=225, altdeg=45).shade_normals(mesh.face_normals, fraction=1.0)
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolors=np.clip(colors * shading[:, None], 0, 1),
                                         edgecolor="none"))
    ax.set_xlim(mins[0], maxs[0]), ax.set_ylim(mins[1], maxs[1]), ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-60)
    ax.set_axis_off()
    ax.dist = 7
    fig.savefig(image_path, dpi=200, transparent=True)
    plt.close(fig)


image_dir = save_dir / "images"
image_dir.mkdir(exist_ok=True)
deviation_limit = 0.3
cells = [("smooth", mesh_dir / "smooth_1.00.stl")] + [(key, mesh_dir / f"{key}.stl") for key, _ in samples]
for key, path in cells:
    if not (image_dir / f"{key}.png").exists():
        render_cell(path, image_dir / f"{key}.png", deviation_limit)

# Figure: change of the moduli of each rough cell against the smooth cell, with the image of the cell
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
x = np.arange(len(samples))
styles = {"compression": ("o", "#1f4e79", r"Compression, $E^*$"), "shear": ("s", "#c0504d", r"Shear, $G^*$")}
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.axhline(0.0, color="#6b6b6b", linewidth=1.0)
offsets = {"compression": -0.12, "shear": 0.12}
for case, (marker, color, label) in styles.items():
    ax.plot(x + offsets[case], changes[case], marker, color=color, markersize=7, markeredgecolor="white",
            markeredgewidth=1.0, label=label, linestyle="none")
    ax.axhline(changes[case].mean(), color=color, linewidth=0.8, linestyle="--")
values = np.concatenate(list(changes.values()))
span = max(np.abs(values).max(), 1.0)
ax.set_ylim(-1.6 * span, 1.6 * span)
ax.set_xlim(-0.6, len(samples) - 0.4)
ax.set_xticks(x)
ax.set_xticklabels([f"{e['Sq_over_radius']:.3f}\n{e['eta']:.2f}\n{e['N']}" for _, e in samples], fontsize=7.5)
ax.text(-0.6, -1.6 * span, "$S_q/r$\n$\\eta$\n$N$", fontsize=7.5, ha="right", va="top",
        transform=ax.transData)
ax.set_ylabel("Change against the smooth cell (%)")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.legend(loc="lower right", frameon=False, ncol=2)
ax.set_title(f"BCC cells at constant relative density $\\rho^*$ = {reference['relative_density']:.3f}",
             fontsize=9, pad=70)

# Image of each cell above its data points, and of the smooth cell on the left
for i, (key, _) in enumerate(samples):
    image = plt.imread(image_dir / f"{key}.png")
    inset = ax.inset_axes([i - 0.45, 1.6 * span * 1.02, 0.9, 0.9 * 2 * 1.6 * span * 0.42], transform=ax.transData)
    inset.set_clip_on(False)
    inset.imshow(image)
    inset.set_axis_off()
smooth_inset = fig.add_axes([0.0, 0.70, 0.11, 0.2])
smooth_inset.imshow(plt.imread(image_dir / "smooth.png"))
smooth_inset.set_axis_off()
smooth_inset.set_title("Smooth", fontsize=8, pad=0)
colorbar_ax = fig.add_axes([0.93, 0.70, 0.01, 0.17])
colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(-deviation_limit,
                                                                                     deviation_limit)),
                        cax=colorbar_ax)
colorbar.ax.set_title(r"$\frac{d - r_0}{r_0}$", fontsize=8)
colorbar.ax.tick_params(labelsize=7)
fig.subplots_adjust(left=0.14, right=0.9, top=0.68, bottom=0.2)
fig.savefig(save_dir / "roughness_same_density.pdf")
fig.savefig(save_dir / "roughness_same_density.png", dpi=300)
print(f"\nFigure and results saved in {save_dir}")
plt.show()
