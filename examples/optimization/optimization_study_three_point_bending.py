"""
Optimization study on the three-point bending lattice (10x5x5 cells, half model) for three parameterizations of the
radii:
    - "hybrid":    one radius per geometry type, constant in the whole lattice (n_geom design variables),
    - "graded":    one radius per cell given by a linear field in x and z (3 design variables),
    - "unit_cell": one radius per geometry type and per cell (n_geom x 250 design variables).

The input file is optimization/Three_point_bending (BCC + Hybrid1) by default; another one is given with
`--input=optimization/<name>` (e.g. Three_point_bending_3geom for BCC + Hybrid1 + Hybrid4). The results are saved in
data/outputs/optimization_study/<name>.

Each parameterization is optimized with two methods:
    - "<name>":     surrogate-enhanced DDM (RBF Schur complements, node penalization),
    - "<name>_FEM": direct FEM solve of the full lattice (K u = f) with the analytic compliance gradient, without node
                    penalization (the cell-wise case uses the 'hybrid_size' parameterization).
Run only some cases with their names as arguments, e.g. `python optimization_study_three_point_bending.py hybrid_FEM`.

For each case:
    - the full optimization history (compliance, relative density, constraint violation) is saved,
    - the initial and optimized designs are re-simulated with the full-order beam FEM (FenicsX, every beam meshed,
      no domain decomposition, no surrogate),
    - the relative density constraint is checked without surrogate with optimization_study_direct_density.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from pyLatticeDesign.utils import open_lattice_parameters
from pyLatticeOpti.lattice_opti import LatticeOpti
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX

name_file = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--input=")), "optimization/Three_point_bending")
max_iterations = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--max-iterations=")), 100))

parameterizations = {
    "hybrid": {"type": "constant", "hybrid": True},
    "graded": {"type": "linear", "direction": ["x", "z"], "hybrid": False},
    "unit_cell": {"type": "unit_cell", "direction": "x", "hybrid": True},
}
# Same parameterizations solved with the direct FEM of the full lattice ('hybrid_size' = cell-wise radii)
cases = {}
for name, parameters in parameterizations.items():
    cases[name] = (parameters, "DDM")
    cases[f"{name}_FEM"] = ({"type": "hybrid_size"} if parameters["type"] == "unit_cell" else parameters, "FEM")
selected = [a for a in sys.argv[1:] if a in cases]  # run only these cases (all by default)
if selected:
    cases = {name: cases[name] for name in selected}

# Label of the model used during the optimization
model_label = {"DDM": "DDM_RBF", "FEM": "FEM_no_penalization"}
gradient_method = {
    "DDM": "analytical adjoint (DDM Schur complement sensitivities, RBF surrogate)",
    "FEM": "analytical (direct FEM, UFL derivative of the strain energy with respect to the radius field)",
}

save_dir = Path(__file__).resolve().parents[2] / "data" / "outputs" / "optimization_study" / Path(name_file).name
save_dir.mkdir(parents=True, exist_ok=True)


def lattice_opti(optimization_parameters: dict, simulation_type: str) -> LatticeOpti:
    """
    LatticeOpti on name_file with the optimization parameterization and the simulation type replaced. The FEM
    optimization disables the node penalization (required by its analytic gradient).
    """
    class LatticeOptiStudy(LatticeOpti):
        @staticmethod
        def _get_DDM_enable_simulation(name_file_opti) -> bool:
            return simulation_type == "DDM"

        def define_simulation_parameters(self, name_file_sim: str) -> None:
            super().define_simulation_parameters(name_file_sim)
            if simulation_type == "FEM":
                self.enable_node_penalization = False

        def _get_optimization_parameters(self, name_file_opti: str) -> None:
            super()._get_optimization_parameters(name_file_opti)
            self.optimization_parameters = dict(optimization_parameters)
            self.optim_max_iteration = max_iterations
            self._simulation_type = simulation_type
            if simulation_type == "FEM":
                self._check_FEM_gradient_inputs()

    return LatticeOptiStudy(name_file, verbose=0)


def lattice_verification(cell_radii: list) -> LatticeSim:
    """
    Full-order beam FEM lattice of name_file with the radii of each cell set before the penalized beams at the nodes
    are built.
    """
    class LatticeSimVerification(LatticeSim):
        def define_connected_beams_for_all_nodes(self, *args, **kwargs):
            for cell, radii in zip(self.cells, cell_radii):
                cell.change_beam_radius(list(radii))
            super().define_connected_beams_for_all_nodes(*args, **kwargs)

    return LatticeSimVerification(name_file)


def compliance(lattice) -> float:
    """C = f^T u on the loaded DOFs (same definition as LatticeOpti.compute_compliance)."""
    return float(sum(node.applied_force[k] * node.displacement_vector[k]
                     for node in lattice.nodes for k in range(6) if node.applied_force[k] != 0.0))


def verify(cell_radii: list) -> dict:
    """Compliance of a design with the full-order beam FEM."""
    out = {}
    start = time.time()
    lattice = lattice_verification(cell_radii)
    solve_FEM_FenicsX(lattice)
    out["FEM"] = {"compliance": compliance(lattice), "time": time.time() - start,
                  "n_beam_elements": len({id(b) for c in lattice.cells for b in c.beams_cell})}
    return out


target_density = open_lattice_parameters(name_file)["optimization_informations"]["constraints"]["relative_density"][
    "value"]
results = {}
verify_only = "--verify-only" in sys.argv  # reuse the optimized designs saved by a previous run

# Optimizations (DDM + RBF surrogate, or direct FEM)
for name, (parameters, simulation_type) in cases.items():
    if verify_only:
        results[name] = json.loads((save_dir / f"{name}.json").read_text())
        continue
    print(f"\n=== {name} ===")
    label = model_label[simulation_type]
    start = time.time()
    lattice = lattice_opti(parameters, simulation_type)
    setup_time = time.time() - start
    initial_radii = [list(map(float, cell.radii)) for cell in lattice.cells]
    rho_initial_surrogate = lattice.get_relative_density()

    start = time.time()
    lattice.optimize_lattice()
    optimization_time = time.time() - start
    solution = lattice.solution

    # Final design evaluated again with the optimization model
    lattice.set_optimization_parameters(list(solution.x))
    lattice._simulate_lattice_equilibrium()
    C_initial = float(lattice.initial_value_objective)
    C_final_model = lattice.compute_compliance()
    rho_final_surrogate = lattice.get_relative_density()
    final_radii = [list(map(float, cell.radii)) for cell in lattice.cells]

    x = np.asarray(solution.x, dtype=float)
    radii_flat = np.array(final_radii).ravel()
    bounded = parameters["type"] != "linear"
    case = {
        "parameterization": parameters,
        "simulation_type": simulation_type,
        "model": label,
        "n_design_variables": int(lattice.number_parameters),
        "n_cells": len(lattice.cells),
        "initial_radii": lattice.radii,
        "optimizer": {"method": "SLSQP", "maxiter": lattice.optim_max_iteration, "ftol": lattice.optim_ftol,
                      "bounds_normalized": [0.0, 1.0], "radius_bounds": [lattice.min_radius, lattice.max_radius],
                      "gradient": gradient_method[simulation_type],
                      "constraint_gradient": "analytical (gradient of the kriging mean of the cell relative density)"
                      if bounded else "central finite differences (kriging surrogate)"},
        "solution": {"success": bool(solution.success), "message": str(solution.message), "nit": int(solution.nit),
                     "nfev": int(solution.nfev), "njev": int(solution.njev)},
        "time": {"setup": setup_time, "optimization": optimization_time},
        "compliance": {f"initial_{label}": C_initial, f"final_{label}": C_final_model,
                       f"reduction_{label}": 1 - C_final_model / C_initial},
        "relative_density": {"target": target_density, "initial_surrogate": rho_initial_surrogate,
                             "final_surrogate": rho_final_surrogate},
        "design": {"x": x.tolist(), "n_at_lower_bound": int(np.sum(x <= 1e-6)) if bounded else None,
                   "n_at_upper_bound": int(np.sum(x >= 1 - 1e-6)) if bounded else None,
                   "radius_min": float(radii_flat.min()), "radius_max": float(radii_flat.max()),
                   "radius_mean": float(radii_flat.mean()), "radius_std": float(radii_flat.std()),
                   "cell_radii": final_radii, "initial_cell_radii": initial_radii},
        "history": lattice._history,
    }
    results[name] = case
    with open(save_dir / f"{name}.json", "w") as file:
        json.dump(case, file, indent=2, default=float)
    print(json.dumps({k: case[k] for k in ("n_design_variables", "solution", "time", "compliance",
                                           "relative_density")}, indent=2, default=float))
    del lattice

# Verification of the initial and optimized designs with the full-order beam FEM
# The initial design is the same for every case and every run: its FEM result is reused when available
if (save_dir / "FEM_initial.json").exists():
    initial_verification = json.loads((save_dir / "FEM_initial.json").read_text())
else:
    print("\n=== Full-order FEM: initial design ===")
    initial_verification = verify(next(iter(results.values()))["design"]["initial_cell_radii"])
    with open(save_dir / "FEM_initial.json", "w") as file:
        json.dump(initial_verification, file, indent=2, default=float)

for name, case in results.items():
    print(f"\n=== Full-order FEM: {name} ===")
    final_verification = verify(case["design"]["cell_radii"])
    case["verification"] = {"initial": initial_verification, "final": final_verification}
    C0, C1 = initial_verification["FEM"]["compliance"], final_verification["FEM"]["compliance"]
    direct_density = save_dir / "relative_density_direct.json"  # written by optimization_study_direct_density.py
    if direct_density.exists():
        case["relative_density"].update(json.loads(direct_density.read_text()).get(name, {}))
    label = case.get("model", "DDM_RBF")
    case["compliance"].update({"initial_FEM": C0, "final_FEM": C1, "reduction_FEM": 1 - C1 / C0,
                               f"error_initial_{label}_vs_FEM": case["compliance"][f"initial_{label}"] / C0 - 1,
                               f"error_final_{label}_vs_FEM": case["compliance"][f"final_{label}"] / C1 - 1})
    with open(save_dir / f"{name}.json", "w") as file:
        json.dump(case, file, indent=2, default=float)
    print(json.dumps(case["compliance"], indent=2))

summary_path = save_dir / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
summary.update({k: {kk: v for kk, v in case.items() if kk not in ("history", "design")} for k, case in results.items()})
summary_path.write_text(json.dumps(summary, indent=2, default=float))
print("\nResults saved to", save_dir)
