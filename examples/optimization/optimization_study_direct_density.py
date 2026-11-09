"""
Direct verification of the relative density constraint of the designs of optimization_study_three_point_bending.py:
the relative density of every cell is recomputed from the Gmsh CAD volume of a single cell with its optimized radii
(same procedure as the training data of the kriging surrogate used in the constraint), then averaged on the lattice.
Study folder: data/outputs/optimization_study/Three_point_bending by default, or `--input=optimization/<name>`; the
other arguments are the cases to check (all the saved cases by default).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.utils import open_lattice_parameters

name_file = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--input=")), "optimization/Three_point_bending")
save_dir = Path(__file__).resolve().parents[2] / "data" / "outputs" / "optimization_study" / Path(name_file).name
geom_types = open_lattice_parameters(name_file)["geometry"]["geom_types"]
cases = [a for a in sys.argv[1:] if not a.startswith("--")] or [f"{name}{suffix}" for suffix in ("", "_FEM") for name in ("hybrid", "graded", "unit_cell")
                         if (save_dir / f"{name}{suffix}.json").exists()]

cell_file = Path(tempfile.mkdtemp()) / ("cell_" + "_".join(geom_types) + ".json")
cell_file.write_text(json.dumps({"geometry": {"cell_size": {"x": 1, "y": 1, "z": 1},
                                              "number_of_cells": {"x": 1, "y": 1, "z": 1},
                                              "radii": [0.05] * len(geom_types), "geom_types": geom_types}}))
cell = Lattice(str(cell_file), verbose=-1)
cache = {}


def cell_density(radii) -> float:
    key = tuple(round(float(r), 10) for r in radii)
    if key not in cache:
        cell.change_beam_radius(list(key))
        cache[key] = float(cell.generate_mesh_lattice_Gmsh(volume_computation=True, cut_mesh_at_boundary=True,
                                                           save_STL=False, only_relative_density=True))
    return cache[key]


for name in cases:
    path = save_dir / f"{name}.json"
    case = json.loads(path.read_text())
    for label, key in (("initial", "initial_cell_radii"), ("final", "cell_radii")):
        rho = np.array([cell_density(r) for r in case["design"][key]])
        case["relative_density"][f"{label}_direct_Gmsh"] = float(rho.mean())
        case["relative_density"][f"{label}_direct_Gmsh_min_cell"] = float(rho.min())
        case["relative_density"][f"{label}_direct_Gmsh_max_cell"] = float(rho.max())
    rd = case["relative_density"]
    rd["final_violation_direct"] = rd["final_direct_Gmsh"] - rd["target"]
    rd["surrogate_error_final"] = rd["final_surrogate"] / rd["final_direct_Gmsh"] - 1
    # The current file may have been rewritten by the study (FEM verification) in the meantime: merge only this field
    latest = json.loads(path.read_text())
    latest["relative_density"] = rd
    path.write_text(json.dumps(latest, indent=2, default=float))
    print(name, json.dumps(rd, indent=1))
    summary_path = save_dir / "relative_density_direct.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[name] = rd
    summary_path.write_text(json.dumps(summary, indent=2))
