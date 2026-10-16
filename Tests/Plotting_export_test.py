"""
Smoke tests of the visualization and export functions: they must run without error (non-interactive backend,
nothing is displayed) and produce their output files.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.plotting_lattice import LatticePlotting


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def config_file(tmp_path):
    def _write(geom_types=("BCC",), simulation=False):
        config = {
            "geometry": {
                "cell_size": {"x": 1, "y": 1, "z": 1},
                "number_of_cells": {"x": 2, "y": 1, "z": 1},
                "radii": [0.05, 0.03][:len(geom_types)],
                "geom_types": list(geom_types),
            }
        }
        if simulation:
            config["simulation_parameters"] = {"enable": True, "material": "VeroClear", "periodicity": False}
            config["boundary_conditions"] = {
                "Displacement": {"Fixed": {"Surface": ["Xmin"], "DOF": ["X", "Y", "Z", "RX", "RY", "RZ"],
                                           "Value": [0, 0, 0, 0, 0, 0]}},
                "Force": {"Load": {"Surface": ["Xmax"], "DOF": ["Z"], "Value": [-0.01]}},
            }
        path = tmp_path / "lattice.json"
        path.write_text(json.dumps(config))
        return str(path)
    return _write


@pytest.mark.parametrize("beam_color_type", ["radii", "type", "material"])
def test_visualize_lattice(config_file, tmp_path, beam_color_type):
    lattice = Lattice(config_file(geom_types=("BCC", "Octet")))
    save_path = tmp_path / f"lattice_{beam_color_type}.png"
    LatticePlotting().visualize_lattice(lattice, beam_color_type=beam_color_type, cell_index=True, node_index=True,
                                        file_save_path=str(save_path), plotting=False)
    assert save_path.exists() and save_path.stat().st_size > 0


def test_plot_radius_distribution_and_hybrid_subplots(config_file):
    lattice = Lattice(config_file(geom_types=("BCC", "Octet")))
    plotter = LatticePlotting()
    plotter.plot_radius_distribution(lattice)
    LatticePlotting().subplot_lattice_hybrid_geometries(lattice)


@pytest.mark.simulation
def test_visualize_deformed_lattice_with_boundary_conditions(config_file):
    pytest.importorskip("dolfinx")
    from pyLatticeSim.lattice_sim import LatticeSim
    from pyLatticeSim.utils_simulation import solve_FEM_FenicsX

    lattice = LatticeSim(config_file(simulation=True))
    solve_FEM_FenicsX(lattice)
    LatticePlotting().visualize_lattice(lattice, deformed_form=True, enable_boundary_conditions=True,
                                        plotting=False)


@pytest.mark.simulation
def test_export_simulation_results(config_file, tmp_path):
    pytest.importorskip("dolfinx")
    from pyLatticeSim.export_simulation_results import exportSimulationResults
    from pyLatticeSim.lattice_sim import LatticeSim
    from pyLatticeSim.utils_simulation import solve_FEM_FenicsX

    lattice = LatticeSim(config_file(simulation=True))
    _, simulation_model = solve_FEM_FenicsX(lattice)
    export = exportSimulationResults(simulation_model, str(tmp_path / "export"))
    export.export_displacement_rotation()
    export.export_beamradii()
    export.export_finalize()
    assert any((tmp_path / "export").iterdir())
