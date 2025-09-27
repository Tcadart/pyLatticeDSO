"""
Main
"""
from data.inputs.mesh_file.mesh_trimmer import MeshTrimmer
from pyLatticeSim.lattice_sim import LatticeSim
from src.pyLattice.plotting_lattice import LatticePlotting
from src.pyLattice.lattice import Lattice

from src.pyLattice.utils import save_JSON_to_Grasshopper
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX
from pyLatticeSim.export_simulation_results import exportSimulationResults


name_mesh = "CutedBone"  # get from https://anatomytool.org/content/thunthu-3d-model-bones-lower-limb
mesh_trimmer = MeshTrimmer(name_mesh)
mesh_trimmer.scale_mesh(1.5)
# mesh_trimmer.plot_mesh(zoom = 3, camera_position=(8.7, -178.7))

name_file = "design/"
name_lattice = "Bone_cuted_hybrid"
lattice_object = LatticeSim(name_file + name_lattice, mesh_trimmer)
# lattice_object.cut_beam_with_mesh_trimmer()
# lattice_object.print_statistics_lattice()

sol, simulation_lattice = solve_FEM_FenicsX(lattice_object)

# save_JSON_to_Grasshopper(lattice_object, name_lattice)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", voxelViz=False, camera_position=(8.7, -178.7),
                             enable_system_coordinates=False, deformedForm=True, enable_boundary_conditions=True)


export_results = exportSimulationResults(simulation_lattice, name_lattice)
export_results.export_displacement_rotation()
export_results.export_finalize()