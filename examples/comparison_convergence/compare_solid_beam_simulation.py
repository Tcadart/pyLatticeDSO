"""
Example script to compare the 3D solid simulation with the beam simulation, with and without penalized beams at
nodes ("beam mod"), on a BCC lattice in flexion.
The solid model is used as the reference.
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX, solve_FEM_solid_FenicsX

name_file = "simulation/comparison_BCC_flexion"
loaded_surface = "Flexion"  # name of the imposed displacement condition in the input file
dof = 2  # Z


def unique_nodes(lattice):
    """Nodes of the lattice (shared by several cells), without the nodes added by the penalization."""
    nodes = {id(node): node for cell in lattice.cells for node in cell.points_cell}.values()
    return [node for node in nodes if not getattr(node, "node_mod", False)]


def node_displacements(lattice):
    """Translation of each original lattice node, keyed by its rounded position."""
    return {tuple(np.round([n.x, n.y, n.z], 6)): np.array(n.displacement_vector[:3]) for n in unique_nodes(lattice)}


def beam_reaction(lattice, surface_value: float):
    """Total reaction force along dof on the nodes of the Xmax face."""
    return sum(n.reaction_force_vector[dof] for n in unique_nodes(lattice) if abs(n.x - surface_value) < 1e-9)


results = {}

# Beam model without penalization
start = time.time()
lattice_beam = LatticeSim(name_file)
lattice_beam.reset_penalized_beams()
_, simulation_beam = solve_FEM_FenicsX(lattice_beam)
results["Beam"] = {"time": time.time() - start,
                   "reaction": beam_reaction(lattice_beam, lattice_beam.x_max),
                   "energy": simulation_beam.calculate_strain_energy(),
                   "displacements": node_displacements(lattice_beam)}

# Beam model with penalized beams at nodes
start = time.time()
lattice_beam_mod = LatticeSim(name_file)
_, simulation_beam_mod = solve_FEM_FenicsX(lattice_beam_mod)
results["Beam mod"] = {"time": time.time() - start,
                       "reaction": beam_reaction(lattice_beam_mod, lattice_beam_mod.x_max),
                       "energy": simulation_beam_mod.calculate_strain_energy(),
                       "displacements": node_displacements(lattice_beam_mod)}

# 3D solid model (reference)
start = time.time()
lattice_solid = LatticeSim(name_file)
simulation_solid = solve_FEM_solid_FenicsX(lattice_solid, mesh_refinement=0.5, element_degree=2)
results["Solid"] = {"time": time.time() - start,
                    "reaction": simulation_solid.compute_reaction_forces()[loaded_surface][dof],
                    "energy": simulation_solid.compute_strain_energy(),
                    "displacements": node_displacements(lattice_solid)}

# Comparison with the solid model
reference = results["Solid"]
positions = sorted(set.intersection(*(set(r["displacements"]) for r in results.values())))
disp_ref = np.array([reference["displacements"][p] for p in positions])

print(f"\n{'Model':10s} {'Time (s)':>9s} {'Reaction':>11s} {'Diff':>8s} {'Energy':>11s} {'Diff':>8s} "
      f"{'Node disp. error':>17s}")
for name, r in results.items():
    disp = np.array([r["displacements"][p] for p in positions])
    r["disp_error"] = np.linalg.norm(disp - disp_ref) / np.linalg.norm(disp_ref)
    print(f"{name:10s} {r['time']:9.1f} {r['reaction']:11.5f} "
          f"{100 * (r['reaction'] / reference['reaction'] - 1):+7.1f}% {r['energy']:11.6f} "
          f"{100 * (r['energy'] / reference['energy'] - 1):+7.1f}% {100 * r['disp_error']:16.2f}%")
print(f"(node displacement error: relative L2 error on the {len(positions)} lattice nodes, solid as reference)")

# Figure
fig, (ax_force, ax_disp) = plt.subplots(1, 2, figsize=(11, 4.5))
names = list(results)
ax_force.bar(names, [abs(results[n]["reaction"]) for n in names], color=["tab:blue", "tab:orange", "tab:gray"])
ax_force.set_ylabel("Reaction force on Xmax (|Fz|)")
ax_force.set_title("Stiffness")

x_positions = np.array([p[0] for p in positions])
order = np.argsort(x_positions, kind="stable")
for name, marker in zip(names, ["o", "s", "x"]):
    uz = np.array([results[name]["displacements"][p][dof] for p in positions])
    ax_disp.scatter(x_positions[order], uz[order], marker=marker, label=name, alpha=0.7)
ax_disp.set_xlabel("x of the lattice node")
ax_disp.set_ylabel("Displacement uz")
ax_disp.set_title("Node displacements")
ax_disp.legend()
fig.tight_layout()

save_dir = Path(__file__).resolve().parents[2] / "data" / "outputs" / "simulation_results" / name_file
save_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(save_dir / "comparison_solid_beam.png", dpi=150)
print("Figure saved to", save_dir / "comparison_solid_beam.png")
plt.show()
