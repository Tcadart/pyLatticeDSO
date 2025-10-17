"""
Simple schur complement example of an hybrid cell
"""
from pathlib import Path

import numpy as np

from pyLattice.plotting_lattice import LatticePlotting
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_schur import get_schur_complement
from pyLatticeSim.utils_schur import save_schur_complement_npz


name_file = "simulation/hybrid_cell_simulation"

lattice_object = LatticeSim(name_file)

schur_complement = get_schur_complement(lattice_object)

print("Schur complement matrix:\n", schur_complement)
