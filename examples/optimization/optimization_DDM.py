"""
Examples of a simple optimization case with the Domain Decomposition Method (DDM) and surrogate models.
"""
import time

from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeOpti.lattice_opti import LatticeOpti
from pyLatticeDesign.utils import save_JSON_to_Grasshopper, save_lattice_object

path = "optimization/"
# name_file = "Three_point_bending"
name_file = "Cantilever_L_beam"


start_time = time.time()

lattice_object = LatticeOpti(path + name_file, verbose=1, convergence_plotting = True)


lattice_object.optimize_lattice()

print("--- %s seconds ---" % (time.time() - start_time))

vizualizer = LatticePlotting()
vizualizer.subplot_lattice_hybrid_geometries(lattice_object)

# Save procedures
lattice_object.reset_penalized_beams()

# Visualization optimized lattice
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_boundary_conditions=True)

# Save optimization results
# lattice_object.save_optimization_json(name_file = name_file + "_optimized")

# Save for Grasshopper to make the geometry
# save_JSON_to_Grasshopper(lattice_object, name_file + "_optimized")

# Save the lattice object to be re-used later
# save_lattice_object(lattice_object, name_file + "_optimized")


