"""
Speed-accuracy comparison of the simulation methods on a three-point bending lattice (half model, symmetry at Xmax):
    - 3D solid FEM (tetrahedral mesh of the beams volume), optional (run_solid)
    - full-scale beam FEM
    - DDM with exact Schur complements (no surrogate)
    - DDM + nearest neighbor surrogate
    - DDM + linear interpolation surrogate
    - DDM + RBF surrogate

Errors are computed with respect to the reference model (reference_method: full-scale beam FEM or 3D solid) on:
    - the displacements of the free lattice nodes (relative L2 and max errors),
    - the reaction force on the loaded line,
    - the strain energy. The only non-zero imposed displacement being the deflection, W = 1/2 u_imposed F_reaction for
      every model: the energy error is therefore equal to the reaction force error (it is kept as a consistency check).

The radii are drawn randomly in each cell ("randomness_hybrid") so that the surrogates are evaluated away from the
points of the training grid; with the uniform radii of "Three_point_bending" the nearest neighbor surrogate is exact.
The support is put on the Xmin/Zmin edge of the lattice since the solid model cannot apply "SurfaceCells".

The meshes and fields of every simulation are saved for Paraview (solid and full-scale beam: PVD/VTU of the FE mesh,
DDM: VTK of the lattice nodes and beams), with the nodal results in a npz file.
"""
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyLatticeDesign.utils import open_lattice_parameters
from pyLatticeSim.export_simulation_results import exportSimulationResults
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX, solve_FEM_solid_FenicsX

name_file = "simulation/Three_point_bending_comparison"
loaded_condition = "Displacement"  # name of the imposed displacement condition in the input file
dof = 2  # Z

# The solid model is temporarily disabled: its boundary conditions on the edges of the bounding box are imposed on a
# few mesh points (singular, mesh dependent reaction force)
run_solid = False
reference_method = "Full-scale"  # "Full-scale" or "Solid"
reference_labels = {"Full-scale": "full-scale beam FEM", "Solid": "3D solid"}

solid_mesh_refinement = 0.5
solid_element_degree = 2
solid_solver = "iterative"  # "direct" runs out of memory on large lattices

ddm_methods = {"DDM": "exact",
               "DDM + near": "nearest_neighbor",
               "DDM + linear": "linear",
               "DDM + RBF": "RBF"}

save_dir = Path(__file__).resolve().parents[2] / "data" / "outputs" / "simulation_results" / name_file
save_dir.mkdir(parents=True, exist_ok=True)


def lattice_DDM(schur_complement_type: str) -> LatticeSim:
    """DDM lattice with the Schur complement computation method replaced by schur_complement_type."""
    class LatticeSimDDM(LatticeSim):
        def define_simulation_parameters(self, name_file_sim: str):
            super().define_simulation_parameters(name_file_sim)
            self.type_schur_complement_computation = schur_complement_type
            if schur_complement_type not in ["exact", "FE2"]:
                schur_parameters = open_lattice_parameters(name_file_sim)["simulation_parameters"]["DDM"]
                self.precision_greedy = schur_parameters["schur_complement_computation"]["precision_greedy"]

    return LatticeSimDDM(name_file, enable_domain_decomposition_solver=True)


def key_position(node) -> tuple:
    return tuple(np.round([node.x, node.y, node.z], 6))


def unique_nodes(lattice):
    """Nodes of the lattice (shared by several cells), without the nodes added by the penalization."""
    nodes = {id(node): node for cell in lattice.cells for node in cell.points_cell}.values()
    return [node for node in nodes if not getattr(node, "node_mod", False)]


def free_boundary_nodes(lattice):
    """
    Cell boundary nodes (the only ones solved by the DDM) without any imposed DOF: the imposed displacements are equal
    in all models and would artificially lower the displacement error.
    """
    return [n for n in unique_nodes(lattice) if n.index_boundary is not None and not any(n.fixed_DOF[:3])]


def node_displacements(lattice):
    """Translation of the free cell boundary nodes, keyed by their rounded position."""
    return {key_position(n): np.array(n.displacement_vector[:3], dtype=float) for n in free_boundary_nodes(lattice)}


def loaded_nodes(lattice):
    return [n for n in unique_nodes(lattice) if n.fixed_DOF[dof] and n.displacement_vector[dof] != 0]


def beam_reaction(lattice):
    """Total reaction force along dof on the loaded nodes."""
    return float(sum(n.reaction_force_vector[dof] for n in loaded_nodes(lattice)))


def beam_strain_energy(lattice):
    """Strain energy from the work of the reactions on the imposed displacements (Clapeyron, no applied force)."""
    return 0.5 * float(sum(n.displacement_vector[d] * n.reaction_force_vector[d]
                           for n in unique_nodes(lattice) for d in range(6) if n.fixed_DOF[d]))


def imposed_value(lattice) -> float:
    """Imposed displacement of the loaded line, read from the input file."""
    condition = lattice.boundary_conditions["Displacement"][loaded_condition]
    return float(condition["Value"][condition["DOF"].index("XYZ"[dof])])


def save_lattice_results(lattice, name: str, solved_nodes: set | None = None):
    """
    Save the lattice nodes and beams with the nodal results:
        - <name>.vtk: legacy VTK (lines) for Paraview,
        - <name>.npz: positions, displacements, rotations, reaction forces and connectivity.
    Nodes not in solved_nodes (DDM: nodes inside the cells) get NaN displacements.
    """
    nodes = list({id(n): n for cell in lattice.cells for n in cell.points_cell}.values())
    index = {id(n): i for i, n in enumerate(nodes)}
    beams = {id(b): b for cell in lattice.cells for b in cell.beams_cell}.values()
    lines = np.array([[index[id(b.point1)], index[id(b.point2)]] for b in beams], dtype=int)
    radii = np.array([b.radius for b in beams], dtype=float)
    positions = np.array([[n.x, n.y, n.z] for n in nodes], dtype=float)
    displacements = np.array([n.displacement_vector for n in nodes], dtype=float)
    reactions = np.array([n.reaction_force_vector for n in nodes], dtype=float)
    if solved_nodes is not None:
        unsolved = np.array([id(n) not in solved_nodes for n in nodes])
        displacements[unsolved] = np.nan

    np.savez(save_dir / f"{name}.npz", positions=positions, displacements=displacements, reactions=reactions,
             beams=lines, radii=radii)

    def vector_field(field_name, values):
        return f"VECTORS {field_name} double\n" + "\n".join(" ".join(f"{v:.10e}" for v in row) for row in values)

    with open(save_dir / f"{name}.vtk", "w") as file:
        file.write(f"# vtk DataFile Version 3.0\n{name}\nASCII\nDATASET POLYDATA\n")
        file.write(f"POINTS {len(positions)} double\n")
        file.write("\n".join(" ".join(f"{v:.10e}" for v in p) for p in positions) + "\n")
        file.write(f"LINES {len(lines)} {3 * len(lines)}\n")
        file.write("\n".join(f"2 {i} {j}" for i, j in lines) + "\n")
        file.write(f"POINT_DATA {len(positions)}\n")
        file.write(vector_field("Displacement", displacements[:, :3]) + "\n")
        file.write(vector_field("Rotation", displacements[:, 3:]) + "\n")
        file.write(vector_field("ReactionForce", reactions[:, :3]) + "\n")
        file.write(f"CELL_DATA {len(lines)}\nSCALARS BeamRadii double 1\nLOOKUP_TABLE default\n")
        file.write("\n".join(f"{r:.10e}" for r in radii) + "\n")


results = {}

# 3D solid model
if run_solid:
    print("\n=== 3D solid FEM ===")
    start = time.time()
    lattice_solid = LatticeSim(name_file)
    simulation_solid = solve_FEM_solid_FenicsX(lattice_solid, mesh_refinement=solid_mesh_refinement,
                                               element_degree=solid_element_degree, solver=solid_solver)
    elapsed = time.time() - start
    # Displacement of the solid evaluated at the lattice nodes (NaN outside the mesh, removed from the comparison)
    nodes_solid = free_boundary_nodes(lattice_solid)
    disp_solid = simulation_solid.evaluate_displacement(np.array([[n.x, n.y, n.z] for n in nodes_solid]))
    reaction_solid = simulation_solid.compute_reaction_forces()[loaded_condition][dof]
    results["Solid"] = {"time": elapsed, "setup": np.nan, "solve": np.nan, "iterations": np.nan,
                        "reaction": reaction_solid,
                        "energy": simulation_solid.compute_strain_energy(),
                        "displacements": {key_position(n): d for n, d in zip(nodes_solid, disp_solid)
                                          if not np.any(np.isnan(d))}}
    print(f"Strain energy: {results['Solid']['energy']:.6e} (volume integral), "
          f"{0.5 * imposed_value(lattice_solid) * reaction_solid:.6e} (1/2 u F)")

    export_solid = exportSimulationResults(simulation_solid, f"{name_file}/solid")
    export_solid.export_solid_displacement()
    export_solid.export_von_mises()
    export_solid.export_finalize()

# Full-scale beam FEM
print("\n=== Full-scale beam FEM ===")
start = time.time()
lattice_beam = LatticeSim(name_file)
setup = time.time() - start
_, simulation_beam = solve_FEM_FenicsX(lattice_beam)
results["Full-scale"] = {"time": time.time() - start, "setup": setup, "solve": time.time() - start - setup,
                         "iterations": np.nan,
                         "reaction": beam_reaction(lattice_beam),
                         "energy": beam_strain_energy(lattice_beam),
                         "displacements": node_displacements(lattice_beam)}
print(f"Strain energy: {results['Full-scale']['energy']:.6e} (from reactions), "
      f"{simulation_beam.calculate_strain_energy():.6e} (from the FE model)")

export_beam = exportSimulationResults(simulation_beam, f"{name_file}/full_scale")
export_beam.export_displacement_rotation()
export_beam.export_beamradii()
export_beam.export_finalize()
save_lattice_results(lattice_beam, "full_scale_nodes")

# DDM, without and with surrogate models of the Schur complements
for name, schur_complement_type in ddm_methods.items():
    print(f"\n=== {name} ===")
    start = time.time()
    lattice_ddm = lattice_DDM(schur_complement_type)  # includes the computation of the Schur complements
    setup = time.time() - start
    _, info, _, _ = lattice_ddm.solve_DDM()
    elapsed = time.time() - start
    if info != 0:
        print(f"Warning: CG did not converge for {name}")
    results[name] = {"time": elapsed, "setup": setup, "solve": elapsed - setup, "iterations": lattice_ddm.iteration,
                     "reaction": beam_reaction(lattice_ddm),
                     "energy": beam_strain_energy(lattice_ddm),
                     "displacements": node_displacements(lattice_ddm)}
    solved = {id(n) for n in unique_nodes(lattice_ddm) if n.index_boundary is not None}
    save_lattice_results(lattice_ddm, name.replace(" + ", "_").replace(" ", "_"), solved_nodes=solved)


def errors(result: dict, reference: dict) -> dict:
    """Relative errors of result with respect to reference, on their common nodes."""
    positions = sorted(set(result["displacements"]) & set(reference["displacements"]))
    u = np.array([result["displacements"][p] for p in positions])
    u_ref = np.array([reference["displacements"][p] for p in positions])
    return {"e_disp_L2": np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref),
            "e_disp_max": np.max(np.linalg.norm(u - u_ref, axis=1)) / np.max(np.linalg.norm(u_ref, axis=1)),
            "e_reaction": abs(result["reaction"] / reference["reaction"] - 1),
            "e_energy": abs(result["energy"] / reference["energy"] - 1),
            "n_nodes": len(positions)}


if reference_method not in results:
    raise ValueError(f"Reference model '{reference_method}' was not simulated.")
reference = results[reference_method]
reference_label = reference_labels[reference_method]
time_DDM = results["DDM"]["time"]
time_full_scale = results["Full-scale"]["time"]
rows = []
for name, r in results.items():
    row = {"method": name, "time": r["time"], "setup": r["setup"], "solve": r["solve"], "iterations": r["iterations"],
           "speedup_vs_full_scale": time_full_scale / r["time"], "speedup_vs_DDM": time_DDM / r["time"],
           "reaction": r["reaction"], "energy": r["energy"]}
    row.update(errors(r, reference))
    rows.append(row)
compared = [row for row in rows if row["method"] != reference_method]

# Console summary
print(f"\nErrors relative to the {reference_label} ({compared[0]['n_nodes']} free nodes for the displacements)")
print(f"{'Method':14s} {'Time (s)':>9s} {'vs FS':>7s} {'vs DDM':>7s} {'CG it.':>7s} {'Reaction':>11s} "
      f"{'Energy':>11s} {'u L2':>8s} {'u max':>8s} {'Reaction':>9s} {'Energy':>8s}")
for row in rows:
    print(f"{row['method']:14s} {row['time']:9.2f} {row['speedup_vs_full_scale']:6.1f}x {row['speedup_vs_DDM']:6.1f}x "
          f"{row['iterations']:7.0f} {row['reaction']:11.4e} {row['energy']:11.4e} "
          f"{100 * row['e_disp_L2']:7.3f}% {100 * row['e_disp_max']:7.3f}% "
          f"{100 * row['e_reaction']:8.3f}% {100 * row['e_energy']:7.3f}%")
print("(speedups relative to the full-scale beam FEM (FS) and to the DDM without surrogate)")

# Save results
with open(save_dir / "comparison_methods.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with open(save_dir / "comparison_methods.json", "w") as file:
    json.dump(rows, file, indent=2, default=float)

# LaTeX table
with open(save_dir / "comparison_methods.tex", "w") as file:
    file.write("\\begin{tabular}{l r r r r r r}\n\\hline\n")
    file.write("Method & Time (s) & Speedup vs FS & Speedup vs DDM & $e_u$ & $e_{u,\\max}$ & $e_F$ \\\\\n\\hline\n")
    for row in rows:
        errors_row = ("--- & --- & ---" if row["method"] == reference_method else
                      " & ".join(f"{100 * row[key]:.2f}\\%" for key in ["e_disp_L2", "e_disp_max", "e_reaction"]))
        file.write(f"{row['method']} & {row['time']:.1f} & {row['speedup_vs_full_scale']:.1f}$\\times$ & "
                   f"{row['speedup_vs_DDM']:.1f}$\\times$ & {errors_row} \\\\\n")
    file.write("\\hline\n\\end{tabular}\n")
print("\nResults saved to", save_dir)

# Speed-accuracy figure
fig, (ax_speed, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.5))
quantities = {"e_disp_L2": ("Displacement (rel. L2)", "o"), "e_disp_max": ("Displacement (max)", "s"),
              "e_reaction": ("Reaction force", "^")}
for key, (label, marker) in quantities.items():
    ax_speed.scatter([row["time"] for row in compared], [100 * row[key] for row in compared], marker=marker,
                     label=label)
for row in compared:
    ax_speed.annotate(row["method"], (row["time"], 100 * row["e_disp_L2"]), textcoords="offset points",
                      xytext=(5, 5), fontsize=8)
ax_speed.axvline(reference["time"], color="gray", linestyle="--", label=f"{reference_label} time")
ax_speed.set_xscale("log")
ax_speed.set_yscale("log")
ax_speed.set_xlabel("Wall-clock time (s)")
ax_speed.set_ylabel(f"Relative error vs {reference_label} (%)")
ax_speed.set_title("Speed-accuracy")
ax_speed.legend(fontsize=8)

x = np.arange(len(compared))
width = 0.25
for i, (key, (label, _)) in enumerate(quantities.items()):
    ax_bar.bar(x + (i - 1) * width, [100 * row[key] for row in compared], width, label=label)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([row["method"] for row in compared], rotation=30, ha="right")
ax_bar.set_ylabel(f"Relative error vs {reference_label} (%)")
ax_bar.set_yscale("log")
ax_bar.set_title(f"Errors relative to the {reference_label}")
ax_bar.legend(fontsize=8)
fig.tight_layout()
fig.savefig(save_dir / "comparison_methods.png", dpi=150)
print("Figure saved to", save_dir / "comparison_methods.png")
plt.show()
