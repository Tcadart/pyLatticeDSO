"""
Example of a domain decomposition simulation using pyLatticeSim.
"""

from pyLatticeSim.lattice_sim import LatticeSim
from pyLattice.plotting_lattice import LatticePlotting


name_file = "simulation/simulation_DDM_surrogate"

solver_DDM = LatticeSim(name_file, verbose=1, enable_domain_decomposition_solver=True)

solver_DDM.solve_DDM()

# Visualization with matplotlib
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(solver_DDM, beam_color_type="radii", deformed_form=True, enable_boundary_conditions=True)


