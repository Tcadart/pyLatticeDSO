"""
This example shows how to load a saved lattice and visualize it.
"""
from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.plotting_lattice import LatticePlotting


name_file = "L_logo_saved.pkl"

# The complete structure is loaded from the pickle file
lattice = Lattice.open_pickle_lattice(name_file)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice)
