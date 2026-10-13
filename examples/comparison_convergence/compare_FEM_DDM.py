"""
Example script to compare FEM simulation with DDM simulation for a beam under flexion.
"""
import numpy as np
import time

from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX

path = "simulation/"
name_file = "Cantilever_L_beam"
# name_file = "Three_point_bending"


start_time_FEM = time.time()
lattice_Sim_object = LatticeSim(path + name_file, verbose = 1)
print("Lattice generation time --- %s seconds ---" % (time.time() - start_time_FEM))


sol_FEM = solve_FEM_FenicsX(lattice_Sim_object)[0]
print("FEM simulation time --- %s seconds ---" % (time.time() - start_time_FEM))

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_Sim_object, beam_color_type="radii",
                             enable_boundary_conditions=True, deformed_form=True)

start_time_DDM= time.time()
lattice_object = LatticeSim(path + name_file, enable_domain_decomposition_solver = True, verbose=1)
print("Lattice generation time --- %s seconds ---" % (time.time() - start_time_DDM))


sol_DDM = lattice_object.solve_DDM()[0]
print("DDM simulation time --- %s seconds ---" % (time.time() - start_time_DDM))

relative_error = np.linalg.norm(sol_FEM - sol_DDM) / np.linalg.norm(sol_FEM)
print("Relative error between FEM and DDM", relative_error)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii",
                             enable_boundary_conditions=True, deformed_form=True,
                             domain_decomposition_simulation_plotting=True)