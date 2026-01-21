# =============================================================================
# UTILS FUNCTIONS FOR PYLATTICESIM
#
# DESCRIPTION:
# This module contains utility functions for the pyLatticeSim package,
# including functions to clear directories, compute directional stiffness
# modulus, and create homogenization figures.
# =============================================================================

import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def clear_directory(directoryPath):
    """
    Clear all files in a directory

    Parameters:
    ------------
    directoryPath: string
        Path to the directory to clear
    """
    # List all files in the directory
    files = [file for file in os.listdir(directoryPath) if os.path.isfile(os.path.join(directoryPath, file))]

    # Loop through and delete each file
    for file in files:
        os.remove(os.path.join(directoryPath, file))

def directional_modulus(matS: np.ndarray, theta: float, phi: float):
    """
    Compute the directional stiffness modulus in a spherical coordinate system.

    Parameters
    ----------
    matS : ndarray
        A 6-by-6 matrix defining the linear stress-to-strain relationship
        using the voigt notation.

    theta : float
        Polar angle in degree.

    phi : float
        Azimuthal angle in degree.

    Returns
    -------
    dirE : {float, ndarray}
        Directional stiffness modulus.
    """
    ct, st = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    cp, sp = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))
    vecu = np.array([st * cp, st * sp, ct])

    invEdir = 0.00
    for i in range(3):
        for j in range(3):
            IJ = i if i == j else 2 + i + j  # Voigt notation
            coefIJ = 1.0 if i == j else 2.0
            for k in range(3):
                for l in range(3):
                    KL = k if k == l else 2 + k + l  # Voigt notation
                    coefKL = 1.0 if k == l else 2.0
                    coef = 1 / (coefIJ * coefKL)
                    tensSijkl = coef * matS[IJ, KL]
                    invEdir += tensSijkl * vecu[i] * vecu[j] * vecu[k] * vecu[l]
    Edir = 1 / invEdir
    return Edir * vecu

def create_homogenization_figure(mat_S_orthotropic: np.ndarray, plot: bool = True, save: bool = False,
                                 name_file: str = "homogenization_figure"):
    """
    Create a 3D figure representing the directional stiffness modulus

    Parameters
    ----------
    mat_S_orthotropic : ndarray
        A 6-by-6 matrix defining the linear stress-to-strain relationship
        using the voigt notation.

    plot : bool, optional
        If True, displays the plot. Default is True.

    save : bool, optional
        If True, save the plot to a file. Default is True.

    name_file : str, optional
        Name of the file to save the plot. Default is "homogenization_figure".
    """

    n = 200  # number of values per direction
    thetavalues = np.linspace(0, 180, n + 1)
    phivalues = np.linspace(0, 360, 2 * n + 1)

    data = []
    for phi in phivalues:
        for theta in thetavalues:
            vecEdir = directional_modulus(mat_S_orthotropic, theta, phi)
            data.append(vecEdir.copy())
    data = np.array(data)

    shp = (thetavalues.size, phivalues.size)
    X = np.reshape(data[:, 0], shp, order='F')
    Y = np.reshape(data[:, 1], shp, order='F')
    Z = np.reshape(data[:, 2], shp, order='F')

    norm = plt.Normalize(vmin=0)
    facecolors = plt.cm.jet(norm(np.sqrt(X ** 2 + Y ** 2 + Z ** 2)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1)
    ax.set_proj_type('ortho')
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Delete axis
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    m = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])

    ax2 = fig.colorbar(m, ax=ax, shrink=1, aspect=20, orientation="vertical")
    ax2.set_label('Directional Stiffness [GPa]')
    if save:
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "simulation_results" / "figure_homogenization" / name_file
        if path.suffix != ".png":
            path = path.with_suffix('.png')

        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved image to {path}")
    if plot:
        plt.show()
    plt.close(fig)
