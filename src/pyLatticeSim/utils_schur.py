# =============================================================================
# UTILS FOR SCHUR COMPLEMENT CALCULATION
#
# DESCRIPTION:
# This module provides utility functions for calculating, saving,
# loading, and normalizing the Schur complement of stiffness matrices
# for lattice structures in finite element simulations.
# =============================================================================

from pathlib import Path
import numpy as np
from mpi4py import MPI

from .beam_model import *
from .schur_complement import SchurComplement

if TYPE_CHECKING:
    from pyLatticeSim.lattice_sim import LatticeSim
    from pyLatticeDesign.cell import Cell


def get_schur_complement(lattice: "LatticeSim", cell_index: int = None):
    """
    Calculate the Schur complement of the stiffness matrix for a given lattice.

    Parameters:
    -----------
    lattice: Lattice object
        The lattice structure to be analyzed.

    cell_index: int, optional
        The index of the cell to be used for the Schur complement calculation.
        If None, the first cell is used.
    """
    if cell_index is None and lattice.get_number_cells() > 1:
        raise ValueError("The lattice must contain only one cell for Schur complement calculation or specify a cell_index.")

    cell = lattice.cells[0] if cell_index is None else lattice.cells[cell_index]
    cell.define_node_order_to_simulate()

    boundary_points = cell.node_in_order_simulation
    tag_boundary = []
    for point in boundary_points:
        tag_boundary.append(point.cell_local_tag[cell.index])

    cell_model = BeamModel(MPI.COMM_SELF, lattice=lattice, cell_index=cell_index)

    # Initialization simulation
    schur_complement_analysis = SchurComplement(cell_model)

    schur_complement, _ = schur_complement_analysis.calculate_schur_complement(tag_boundary)

    return schur_complement

def save_schur_complement_npz(lattice_object, radius_values: list, schur_matrices: list):
    """
    Save the Schur complement matrices and corresponding radius values to a .npz file.

    Parameters:
    -----------
    lattice_object: LatticeSim
        The lattice object containing geometry and material information.

    radius_values: list
        A list of radius values corresponding to each Schur complement matrix.

    schur_matrices: list
        A list of Schur complement matrices to be saved.
    """
    output_file = define_path_schur_complement(lattice_object)
    np.savez(output_file, radius_values=np.array(radius_values), schur_matrices=np.array(schur_matrices))
    print("Schur complement data saved to", output_file)

def define_path_schur_complement(lattice_object: "LatticeSim") -> Path:
    """
    Define the file path for the Schur complement dataset based on the lattice geometry.

    Parameters:
    -------------
    lattice_object: LatticeSim
        The lattice object containing geometry and material information.

    Returns:
    ---------
    path_file: Path
        The full path to the Schur complement dataset file.
    """
    path_dataset_schur = Path(__file__).parents[2] / "data" / "outputs" / "schur_complement"
    geom_type_str = "_" + "_".join(str(gt) for gt in lattice_object.geom_types)
    name_file = Path("Schur_complement" + geom_type_str)
    if name_file.suffix.lower() != ".npz":
        name_file = name_file.with_suffix(".npz")

    path_file = path_dataset_schur / name_file
    return path_file

def load_schur_complement_dataset(lattice_object: "LatticeSim", enable_normalization: bool = False) -> dict:
    """
    Load the Schur complement matrices from a file and normalize them if needed.

    Parameters:
    -------------
    lattice_object: LatticeSim
        The lattice object containing geometry and material information.

    enable_normalization: bool
        If True, normalize each Schur complement matrix.

    Returns:
    ---------
    Schur_complement: dict
        A dictionary with radius tuples as keys and corresponding Schur complement matrices as values.
    """
    file_name = define_path_schur_complement(lattice_object)

    data = np.load(file_name, allow_pickle=True)
    radius_values = data["radius_values"]
    schur_matrices = data["schur_matrices"]

    # Handle single or multiple entries
    if np.ndim(radius_values[0]) == 0 or np.ndim(radius_values) == 1:
        schur_complement_dict = {tuple(np.atleast_1d(radius_values)): schur_matrices}
    else:
        schur_complement_dict = {tuple(radius): matrix for radius, matrix in zip(radius_values, schur_matrices)}

    if enable_normalization:
        schur_complement_dict = normalize_schur_matrix(schur_complement_dict)

    return schur_complement_dict

def normalize_schur_matrix(schur_dict: dict) -> dict:
    """
    Normalize each Schur complement matrix in the dictionary.

    Parameters:
    ------------
    schur_dict: dict
        A dictionary with radius tuples as keys and corresponding Schur complement matrices as values.

    Returns:
    ------------
    schur_dict: dict
        The input dictionary with each Schur complement matrix normalized.
    """
    for rad in schur_dict:
        S = schur_dict[rad]
        schur_dict[rad] = S / np.linalg.norm(S)

    return schur_dict
