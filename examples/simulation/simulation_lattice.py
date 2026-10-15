"""
Simple example of simulation of a beam in flexion using pyLatticeSim.
"""
import time

from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX
from pyLatticeSim.export_simulation_results import exportSimulationResults


name_file = "simulation/simulation_beam_flexion"
timer = time.time()

lattice_Sim_object = LatticeSim(name_file, verbose=1)

print("LatticeSim object created in ", time.time() - timer, " seconds.")
timer = time.time()
sol, simulation_lattice = solve_FEM_FenicsX(lattice_Sim_object)

print("Simulation completed in ", time.time() - timer, " seconds.")
timer = time.time()

# Visualization with matplotlib
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_Sim_object, beam_color_type="radii", deformed_form=True,
                             enable_boundary_conditions=True, plotting=False)

print("Visualization completed in ", time.time() - timer, " seconds.")
timer = time.time()

# Export the results to Paraview
export_results = exportSimulationResults(simulation_lattice, name_file)
export_results.export_displacement_rotation()
export_results.export_beamradii()
export_results.export_finalize()

print("Export completed in ", time.time() - timer, " seconds.")
timer = time.time()