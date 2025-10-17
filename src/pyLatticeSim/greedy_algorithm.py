import re
from pathlib import Path
from typing import TYPE_CHECKING


import scipy.linalg as la
import numpy as np

if TYPE_CHECKING:
    from pyLatticeSim.lattice_sim import LatticeSim


def reduce_basis_greedy(schur_complement_dict_to_reduce: dict , tol_greedy: float, file_name: str = None,
                        verbose: int = 1):
    """
    Generated a reduced basis with a greedy algorithm from a set of projected fields.
    
    Parameters
    ----------
    schur_complement_dict_to_reduce : dict
        A dictionary where keys are identifiers and values are the Schur complements to be projected.
    tol_greedy : float
        Tolerance for the greedy algorithm convergence.
    file_name : str, optional
        If provided, the reduced basis and related data will be saved to a .npz file with this name.
    verbose : int, optional
        Verbosity level for logging information during the process.

    Returns
    -------
    mainelem : np.ndarray
        Indices of the selected main elements.
    reducedcoef : np.ndarray
        Coefficients for the reduced basis.
    projfieldpp : list
        List of projected fields corresponding to the main elements.
    basis_reduced_ortho : np.ndarray
        The orthonormal reduced basis.
    alpha_ortho : np.ndarray
        Coefficients for projecting original fields onto the reduced basis.
    matP_sorted : np.ndarray
        Upper triangular matrix from the QR decomposition of the reduced coefficients.
    norm_mainelem_sorted : np.ndarray
        Norms of the projected fields corresponding to the main elements.

    Notes
    -----
    The greedy algorithm iteratively selects the most significant projected field and orthogonalizes it against the
    current basis until convergence. The resulting basis is orthonormal and can be used for efficient projections.
    The function uses LAPACK and BLAS routines for efficient linear algebra operations.
    This implementation was provided by Thibaut Hirschler and adapted for pyLatticeSim.

    """
    if not isinstance(schur_complement_dict_to_reduce, dict):
        raise ValueError("schur_complement_dict_to_reduce should be a dict of Schur complements.")
    keys_list = sorted(schur_complement_dict_to_reduce.keys())
    list_elements = np.array(keys_list)
    matrix_schur = np.array([schur_complement_dict_to_reduce[k] for k in keys_list])

    if list_elements.shape[1] > 1:
        sizeData = list_elements.shape[0]
    else:
        sizeData = list_elements.size
    projfields = []
    projfieldsnorm = np.ones(sizeData)
    # Normalization of the projected fields
    for i in range(sizeData):
        projfield = np.ravel(matrix_schur[i], order='F')
        projfieldsnorm[i] = np.linalg.norm(projfield)
        projfields.append(projfield / projfieldsnorm[i])
    projfields = np.stack(projfields).T

    diffprojfields = np.copy(projfields, order='A')
    atol = tol_greedy * la.norm(diffprojfields.T, np.inf)
    diffnormInf = np.zeros(sizeData)
    cvg = False
    maxiter = sizeData
    count = 0
    reducedbasis = []
    reducedcoef = []
    mainelem = []
    while (not cvg) and (count < maxiter):
        count += 1
        diffnormInf[:] = la.norm(diffprojfields, np.inf, axis=0)
        sI = np.argmax(diffnormInf)

        newvec = diffprojfields[:, sI] / la.norm(diffprojfields[:, sI], 2)
        newcoef = la.blas.dgemv(1., diffprojfields, newvec, trans=1)
        la.blas.dger(-1, newvec, newcoef, a=diffprojfields, overwrite_a=1)

        cvg = True if (la.norm(diffprojfields[:, :].T, np.inf) < atol) else False
        reducedbasis.append(newvec.copy())
        reducedcoef.append(newcoef.copy())
        mainelem.append(sI)

    mainelem = np.array(mainelem)
    reducedcoef = np.asfortranarray(np.stack(reducedcoef))
    matP = np.asfortranarray(np.triu(reducedcoef[:, mainelem]))
    la.lapack.dtrtrs(matP, reducedcoef, lower=0, trans=0, unitdiag=0, overwrite_b=1)
    reducedcoef[:, :] *= la.blas.dger(1, 1. / projfieldsnorm[mainelem], projfieldsnorm)

    vsort = np.argsort(mainelem)

    basis_reduced_ortho = np.column_stack(reducedbasis) if len(reducedbasis) else np.empty((projfields.shape[0], 0))
    basisF = np.asfortranarray(basis_reduced_ortho, dtype=float)
    alpha_ortho = np.zeros((basis_reduced_ortho.shape[1], projfields.shape[1]))
    for S in range(projfields.shape[1]):
        projfield = np.ravel(matrix_schur[S], order='F')
        alpha_ortho[:, S] = la.lstsq(basisF, projfield)[0]

    # basis_reduced_ortho = basis_reduced_ortho[:, vsort]
    matP_sorted = matP[np.ix_(vsort, vsort)]
    norm_mainelem_sorted = projfieldsnorm[mainelem[vsort]]

    if file_name is not None:
        save_reduced_basis(file_name, basis_reduced_ortho, alpha_ortho, list_elements)

    projfieldpp = [matrix_schur[i] for i in mainelem[vsort]]
    if verbose >= 1:
        print("Number of elements in the reduced basis:", len(mainelem))
        print("Selected elements:", list_elements[mainelem[vsort]])

    return (mainelem[vsort], reducedcoef[vsort, :], projfieldpp,
            basis_reduced_ortho, alpha_ortho, matP_sorted, norm_mainelem_sorted)

def save_reduced_basis(file_name: str, basis_reduced_ortho, alpha_ortho, list_elements):
    """
    Save the reduced basis and related data to a .npz file.

    Parameters
    ----------
    file_name : str
        Name of the file to save the data (without extension).
    basis_reduced_ortho : np.ndarray
        The orthonormal reduced basis.
    alpha_ortho : np.ndarray
        Coefficients for projecting original fields onto the reduced basis.
    list_elements : np.ndarray
        List of elements corresponding to the reduced basis.
    """
    saving_path = (Path(__file__).parents[2] / "data" / "outputs" / "schur_complement" / "reduced_basis" / file_name)
    if saving_path.suffix != ".npz":
        saving_path = saving_path.with_suffix(".npz")
    dict_to_save = {
        'basis_reduced_ortho': basis_reduced_ortho,
        'alpha_ortho': alpha_ortho,
        'list_elements': list_elements
    }
    np.savez_compressed(saving_path, **dict_to_save)
    print(f"Reduced basis saved to {saving_path}")

def load_reduced_basis(lattice_object_sim: "LatticeSim", tol_greedy: float):
    """
    Load a previously saved reduced basis based on the lattice simulation parameters and tolerance.

    Parameters
    ----------
    lattice_object_sim : LatticeSim
        The lattice simulation object containing parameters.
    tol_greedy : float
        Tolerance used in the greedy algorithm.

    Returns
    -------
    dict
        A dictionary containing the loaded reduced basis and related data.

    """
    file_name = find_name_file_reduced_basis(lattice_object_sim, tol_greedy)
    loading_path = (Path(__file__).parents[2] / "data" / "outputs" / "schur_complement" / "reduced_basis" / file_name)
    if loading_path.suffix != ".npz":
        loading_path = loading_path.with_suffix(".npz")
    if not loading_path.is_file():
        raise FileNotFoundError(f"Reduced basis file not found: {loading_path}")
    loaded_dict = np.load(loading_path)
    return loaded_dict


def find_name_file_reduced_basis(lattice_object_sim: "LatticeSim", tol_greedy: float):
    """
    Construct the filename for the reduced basis based on the lattice simulation parameters.

    Parameters
    ----------
    lattice_object_sim : LatticeSim
        The lattice simulation object containing parameters.
    tol_greedy : float
        Tolerance used in the greedy algorithm.

    Returns
    -------
    str
        The constructed filename for the reduced basis.
    """
    suffix = "_".join(re.sub(r"\W+", "-", str(g)) for g in lattice_object_sim.geom_types)
    tol_str = re.sub(r"e([+-])0+(\d+)$", r"e\1\2", f"{tol_greedy:.0e}")
    file_name = f"reduced_basis_{suffix}_tol_{tol_str}"
    return file_name

def project_to_reduced_basis(schur_complement_dict_to_project: dict, basis_matrix_ortho: np.ndarray):
    """
    Project Schur complements into the previously built reduced basis.

    Parameters
    ----------
    schur_complement_dict_to_project : dict
        A dictionary where keys are identifiers and values are the Schur complements to be projected.
    basis_matrix_ortho : np.ndarray
        The orthonormal reduced basis obtained from the greedy algorithm.
    """
    if not isinstance(schur_complement_dict_to_project, dict):
        raise ValueError("schur_input should be a dict of Schur complements.")
    if basis_matrix_ortho.size == 0:
        raise ValueError("Empty basis_reduced_ortho: build the reduced basis with return_projection_data=True.")

    # Ensure Fortran-contiguous arrays to match LAPACK/BLAS usage above
    basisF = np.asfortranarray(basis_matrix_ortho, dtype=float)

    def _project_one_alpha(Schur_matrix):
        v = np.ravel(Schur_matrix, order='C').astype(float, copy=False)

        alphas_lstsq = la.lstsq(basisF, v)[0]
        return alphas_lstsq

    alphas_dict = {}
    for key, S in schur_complement_dict_to_project.items():
        x = _project_one_alpha(S)
        alphas_dict[key] = x
    return alphas_dict
