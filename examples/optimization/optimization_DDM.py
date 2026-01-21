"""
Examples of a simple optimization case.
"""
import time

from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting
from pyLatticeOpti.lattice_opti import LatticeOpti
from pyLattice.utils import save_JSON_to_Grasshopper, save_lattice_object

path = "optimization/"
# name_file = "optimization_DDM_surrogate"
name_file = "Three_point_bending"
# name_file = "Three_point_bending_adapted"
# name_file = "Cantilever_L_beam"
# name_file = "Inversion_mechanism"


start_time = time.time()

lattice_object = LatticeOpti(path + name_file, verbose=1, convergence_plotting = True)


lattice_object.optimize_lattice()

print("--- %s seconds ---" % (time.time() - start_time))

vizualizer = LatticePlotting()
# vizualizer.visualize_lattice(lattice_object, beam_color_type="radii")
# vizualizer.visualize_lattice_voxels(lattice_object, beam_color_type="radii")
vizualizer.subplot_lattice_hybrid_geometries(lattice_object)
# lattice_object.reset_penalized_beams()

# lattice_object.save_optimization_json(name_file = name_file + "_optimized")
# save_JSON_to_Grasshopper(lattice_object, name_file + "_optimized")
# save_lattice_object(lattice_object, name_file + "_optimized")

# Visualization optimized lattice
# vizualizer = LatticePlotting()
# vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_boundary_conditions=True,
#                              deformedForm = True, use_radius_grad_color=True)
# vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", use_radius_grad_color=True)
