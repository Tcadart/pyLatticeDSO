"""
Speed-accuracy comparison of the simulation methods on a three-point bending lattice (half model, symmetry at Xmax):
    - 3D solid FEM (tetrahedral mesh of the beams volume)
    - full-scale beam FEM
    - DDM with exact Schur complements (no surrogate)
    - DDM + nearest neighbor surrogate
    - DDM + linear interpolation surrogate
    - DDM + RBF surrogate

Errors are reported with respect to the full-scale beam FEM (error of the DDM and of the surrogates alone) and with
respect to the 3D solid model (total error, including the beam modelling error) on:
    - the displacements of the lattice nodes (relative L2 and max errors),
    - the reaction force on the loaded line,
    - the strain energy (equal to half the compliance for an imposed displacement).

The radii are drawn randomly in each cell ("randomness_hybrid") so that the surrogates are evaluated away from the
points of the training grid; with the uniform radii of "Three_point_bending" the nearest neighbor surrogate is exact.
The support is put on the Xmin/Zmin edge of the lattice since the solid model cannot apply "SurfaceCells".
"""
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyLatticeDesign.utils import open_lattice_parameters
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX, solve_FEM_solid_FenicsX

name_file = "simulation/Three_point_bending_comparison"
loaded_condition = "Displacement"  # name of the imposed displacement condition in the input file
dof = 2  # Z

run_solid = True  # the solid model is by far the most expensive simulation
solid_mesh_refinement = 0.5
solid_element_degree = 2
solid_solver = "iterative"  # "direct" runs out of memory on the full lattice

ddm_methods = {"DDM": "exact",
               "DDM + NN": "nearest_neighbor",
               "DDM + linear": "linear",
               "DDM + RBF": "RBF"}


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


def unique_nodes(lattice):
    """Nodes of the lattice (shared by several cells), without the nodes added by the penalization."""
    nodes = {id(node): node for cell in lattice.cells for node in cell.points_cell}.values()
    return [node for node in nodes if not getattr(node, "node_mod", False)]


def boundary_node_displacements(lattice):
    """Translation of the cell boundary nodes (the only ones solved by the DDM), keyed by their rounded position."""
    return {tuple(np.round([n.x, n.y, n.z], 6)): np.array(n.displacement_vector[:3], dtype=float)
            for n in unique_nodes(lattice) if n.index_boundary is not None}


def loaded_nodes(lattice):
    return [n for n in unique_nodes(lattice) if n.fixed_DOF[dof] and n.displacement_vector[dof] != 0]


def beam_reaction(lattice):
    """Total reaction force along dof on the loaded nodes."""
    return float(sum(n.reaction_force_vector[dof] for n in loaded_nodes(lattice)))


def beam_strain_energy(lattice):
    """Strain energy from the work of the reactions on the imposed displacements (Clapeyron, no applied force)."""
    return 0.5 * float(sum(n.displacement_vector[d] * n.reaction_force_vector[d]
                           for n in unique_nodes(lattice) for d in range(6) if n.fixed_DOF[d]))


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
    nodes = [n for n in unique_nodes(lattice_solid) if n.index_boundary is not None]
    disp = simulation_solid.evaluate_displacement(np.array([[n.x, n.y, n.z] for n in nodes]))
    results["Solid"] = {"time": elapsed, "setup": np.nan, "solve": np.nan, "iterations": np.nan,
                        "reaction": simulation_solid.compute_reaction_forces()[loaded_condition][dof],
                        "energy": simulation_solid.compute_strain_energy(),
                        "displacements": {tuple(np.round([n.x, n.y, n.z], 6)): d for n, d in zip(nodes, disp)
                                          if not np.any(np.isnan(d))}}

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
                         "displacements": boundary_node_displacements(lattice_beam)}
print(f"Strain energy: {results['Full-scale']['energy']:.6e} (from reactions), "
      f"{simulation_beam.calculate_strain_energy():.6e} (from the FE model)")

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
                     "displacements": boundary_node_displacements(lattice_ddm)}


def errors(result: dict, reference: dict) -> dict:
    """Relative errors of result with respect to reference, on their common nodes."""
    positions = sorted(set(result["displacements"]) & set(reference["displacements"]))
    u = np.array([result["displacements"][p] for p in positions])
    u_ref = np.array([reference["displacements"][p] for p in positions])
    return {"disp_L2": np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref),
            "disp_max": np.max(np.linalg.norm(u - u_ref, axis=1)) / np.max(np.linalg.norm(u_ref, axis=1)),
            "reaction": abs(result["reaction"] / reference["reaction"] - 1),
            "energy": abs(result["energy"] / reference["energy"] - 1),
            "n_nodes": len(positions)}


references = ["Full-scale"] + (["Solid"] if run_solid else [])
time_DDM = results["DDM"]["time"]
time_full_scale = results["Full-scale"]["time"]
rows = []
for name, r in results.items():
    row = {"method": name, "time": r["time"], "setup": r["setup"], "solve": r["solve"], "iterations": r["iterations"],
           "speedup_vs_full_scale": time_full_scale / r["time"], "speedup_vs_DDM": time_DDM / r["time"],
           "reaction": r["reaction"], "energy": r["energy"]}
    for reference in references:
        for key, value in errors(r, results[reference]).items():
            row[f"{key}_vs_{reference}"] = value
    rows.append(row)

# Console summary
for reference in references:
    print(f"\nErrors relative to the {reference} model "
          f"({rows[0][f'n_nodes_vs_{reference}']} common nodes for the displacements)")
    print(f"{'Method':14s} {'Time (s)':>9s} {'Speedup':>8s} {'CG it.':>7s} {'Reaction':>11s} {'Energy':>11s} "
          f"{'u L2':>8s} {'u max':>8s} {'Reaction':>9s} {'Energy':>8s}")
    for row in rows:
        print(f"{row['method']:14s} {row['time']:9.2f} {row['speedup_vs_DDM']:7.1f}x {row['iterations']:7.0f} "
              f"{row['reaction']:11.4e} {row['energy']:11.4e} "
              f"{100 * row[f'disp_L2_vs_{reference}']:7.3f}% {100 * row[f'disp_max_vs_{reference}']:7.3f}% "
              f"{100 * row[f'reaction_vs_{reference}']:8.3f}% {100 * row[f'energy_vs_{reference}']:7.3f}%")
print("(speedup relative to the DDM without surrogate)")

# Save results
save_dir = Path(__file__).resolve().parents[2] / "data" / "outputs" / "simulation_results" / name_file
save_dir.mkdir(parents=True, exist_ok=True)
with open(save_dir / "comparison_methods.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with open(save_dir / "comparison_methods.json", "w") as file:
    json.dump(rows, file, indent=2, default=float)

# LaTeX table (errors relative to the full-scale beam FEM, and to the solid model if available)
with open(save_dir / "comparison_methods.tex", "w") as file:
    columns = "".join(f" & $e_u$ & $e_F$ & $e_W$" for _ in references)
    file.write("\\begin{tabular}{l r r" + " r r r" * len(references) + "}\n\\hline\n")
    file.write(" & & " + "".join(f" & \\multicolumn{{3}}{{c}}{{vs {ref}}}" for ref in references) + " \\\\\n")
    file.write("Method & Time (s) & Speedup" + columns + " \\\\\n\\hline\n")
    for row in rows:
        file.write(f"{row['method']} & {row['time']:.1f} & {row['speedup_vs_DDM']:.1f}$\\times$")
        for ref in references:
            file.write("".join(f" & {100 * row[f'{key}_vs_{ref}']:.2f}\\%" for key in ["disp_L2", "reaction", "energy"]))
        file.write(" \\\\\n")
    file.write("\\hline\n\\end{tabular}\n")
print("\nResults saved to", save_dir)

# Speed-accuracy figure (errors relative to the full-scale beam FEM, which has no error by definition)
fig, axes = plt.subplots(1, len(references) + 1, figsize=(5.5 * (len(references) + 1), 4.5))
quantities = {"disp_L2": ("Displacement (rel. L2)", "o"), "reaction": ("Reaction force", "s"),
              "energy": ("Strain energy", "^")}
ax = axes[0]
for key, (label, marker) in quantities.items():
    selected = [row for row in rows if row["method"] in ddm_methods]
    ax.scatter([row["time"] for row in selected], [row[f"{key}_vs_Full-scale"] for row in selected],
               marker=marker, label=label)
for row in rows:
    if row["method"] in ddm_methods:
        ax.annotate(row["method"], (row["time"], row["disp_L2_vs_Full-scale"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
ax.axvline(time_full_scale, color="gray", linestyle="--", label="Full-scale FEM time")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Wall-clock time (s)")
ax.set_ylabel("Relative error vs full-scale beam FEM")
ax.set_title("Speed-accuracy of the DDM and surrogates")
ax.legend(fontsize=8)

# Errors relative to each reference, per method
for ax, reference in zip(axes[1:], references):
    methods = [row["method"] for row in rows if row["method"] != reference]
    x = np.arange(len(methods))
    width = 0.25
    for i, (key, (label, _)) in enumerate(quantities.items()):
        values = [100 * row[f"{key}_vs_{reference}"] for row in rows if row["method"] != reference]
        ax.bar(x + (i - 1) * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel(f"Relative error vs {reference} (%)")
    ax.set_yscale("log")
    ax.set_title(f"Errors relative to the {reference} model")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(save_dir / "comparison_methods.png", dpi=150)
print("Figure saved to", save_dir / "comparison_methods.png")
plt.show()
