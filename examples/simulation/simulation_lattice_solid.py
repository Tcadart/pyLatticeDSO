"""
Simple example of 3D solid simulation of a lattice in compression using pyLatticeSim.
The beams are meshed with tetrahedra and solved in linear elasticity, with the boundary conditions of the input file.
"""

from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeSim.utils_simulation import solve_FEM_solid_FenicsX
from pyLatticeSim.export_simulation_results import exportSimulationResults


name_file = "simulation/solid_BCC_compression"

lattice_Sim_object = LatticeSim(name_file)

simulation_lattice = solve_FEM_solid_FenicsX(lattice_Sim_object, mesh_refinement=0.5, element_degree=2)

print("Reaction forces:", simulation_lattice.compute_reaction_forces())
print("Strain energy:", simulation_lattice.compute_strain_energy())
print("Maximum displacement:", simulation_lattice.get_max_displacement())

# Visualization with matplotlib (displacement interpolated at the lattice nodes)
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_Sim_object, beam_color_type="radii", deformed_form=True,
                             enable_boundary_conditions=True)

# Export the results to Paraview
export_results = exportSimulationResults(simulation_lattice, name_file)
export_results.export_solid_displacement()
export_results.export_von_mises()
export_results.export_finalize()
