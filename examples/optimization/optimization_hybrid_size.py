"""
Size optimization of a hybrid lattice (BCC + Hybrid1 + Hybrid4) in bending.

One radius per base geometry and per cell, direct FEM solve of the full lattice (K u = f) and analytic gradient of the
compliance. The node penalization is disabled so that the stiffness only depends on the radii through the beam
sections. The analytic gradient is compared with finite differences at the initial point before optimizing.
"""
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeOpti.lattice_opti import LatticeOpti

name_file = "optimization/optimization_hybrid_size"

lattice_object = LatticeOpti(name_file, verbose=0, convergence_plotting=True)

# Gradient check on the first iteration
lattice_object.check_gradient(eps=1e-3)

lattice_object.optimize_lattice()

# Visualization optimized lattice
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_boundary_conditions=True)
