"""
Example of reducing the basis of a Schur complement using a greedy algorithm with pyLatticeSim.
"""
import re

from pyLatticeSim.utils_schur import load_schur_complement_dataset
from pyLatticeSim.greedy_algorithm import reduce_basis_greedy, find_name_file_reduced_basis
from pyLatticeSim.lattice_sim import LatticeSim

name_file = "simulation/hybrid_cell_simulation"
tolerance_greedy = 1e-6

lattice_Sim_object = LatticeSim(name_file)

schur_data = load_schur_complement_dataset(lattice_Sim_object)

file_name = find_name_file_reduced_basis(lattice_Sim_object, tol_greedy=tolerance_greedy)
reduce_basis_greedy(schur_data, tolerance_greedy, file_name)

