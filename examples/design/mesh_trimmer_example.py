"""
Example of lattice trimmed with a mesh with class MeshTrimmer.
"""
from src.pyLatticeDesign.plotting_lattice import LatticePlotting
from src.pyLatticeDesign.lattice import Lattice

from data.inputs.mesh_file.mesh_trimmer import MeshTrimmer

name_mesh = "CutedBone"  # get from https://anatomytool.org/content/thunthu-3d-model-bones-lower-limb
mesh_trimmer = MeshTrimmer(name_mesh)
mesh_trimmer.plot_mesh(zoom = 3)

name_lattice = "design/BCC_trimmed_example"
lattice_object = Lattice(name_lattice, mesh_trimmer)
lattice_object.cut_beam_with_mesh_trimmer()

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii")
