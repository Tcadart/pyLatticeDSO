"""
Examples of a simple optimization case.
"""

from pyLattice.plotting_lattice import LatticePlotting
from pyLatticeOpti.lattice_opti import LatticeOpti
from pyLattice.utils import save_JSON_to_Grasshopper, save_lattice_object

path = "optimization/"
# name_file = "optimization_DDM_surrogate"
name_file = "Cantilever_L_beam"

lattice_object = LatticeOpti(path + name_file, verbose=1, convergence_plotting = True)

lattice_object.optimize_lattice()

lattice_object.reset_penalized_beams()
# lattice_object.delete_beams_under_radius_threshold(0.015)

save_JSON_to_Grasshopper(lattice_object, name_file + "_optimized")
save_lattice_object(lattice_object, name_file + "_optimized")

# Visualization optimized lattice
vizualizer = LatticePlotting()
# vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_boundary_conditions=True,
#                              deformedForm = True, use_radius_grad_color=True)
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", use_radius_grad_color=True)
