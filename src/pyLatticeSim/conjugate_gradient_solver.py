# =============================================================================
# FUNCTION: conjugate_gradient_solver
#
# DESCRIPTION:
# This function implements the Conjugate Gradient method to solve the
# linear system Ax = b. It supports preconditioning, convergence criteria,
# and allows for a callback function to monitor progress.
# =============================================================================

import numpy as np
from colorama import Style, Fore



def conjugate_gradient_solver(
    A_operator,
    b,
    M=None,
    maxiter=100,
    tol=1e-5,
    mintol=1e-5,
    restart_every=1000,
    alpha_max=0.1,
    callback=None
):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    Parameters
    ----------
    A_operator : callable
        Function that computes the matrix-vector product Ax.

    b : array_like
        Right-hand side vector.

    M : array_like, optional
        Preconditioner matrix (if None, no preconditioning is applied).

    maxiter : int, optional
        Maximum number of iterations (default is 100).

    tol : float, optional
        Tolerance for convergence (default is 1e-5).

    mintol : float, optional
        Minimum value of residue (default is 1e-5).

    restart_every : int, optional
        Number of iterations after which to restart the search direction (default is 5).

    alpha_max : float, optional
        Maximum step size for the search direction (default is 1).

    callback : callable, optional
        Function to call after each iteration with the current solution.
    """

    # Initialisation
    n = b.shape[0]
    x = np.zeros(n, dtype=np.float64)
    r = b - A_operator @ x

    # Preconditioning
    if M is not None:
        z = M @ r
    else:
        z = r

    p = z.copy()
    rz_old = np.dot(r, z)
    norm_b = np.linalg.norm(b)
    info = 1  # Default: did not converge

    for k in range(maxiter):
        # print(Fore.RED + "Iteration", k + 1, Style.RESET_ALL)
        Ap = A_operator @ p
        alpha = rz_old / np.dot(p, Ap)
        alpha = min(alpha, alpha_max)

        x += alpha * p
        r -= alpha * Ap

        # --- Callback function ---
        if callback is not None:
            callback(x)

        # --- Restart condition ---
        if k % restart_every == 0 and k > 0:
            p = z.copy()

        # --- Convergence test  ---
        residual_norm = np.linalg.norm(r)
        direction_norm = np.linalg.norm(p)
        solution_norm = np.linalg.norm(x)

        if residual_norm <= tol * norm_b:
            info = 0
            print(Fore.GREEN + "Convergence achieved: norm(r) <= tol * norm(b)" + Style.RESET_ALL)
            break

        if direction_norm < mintol * (solution_norm + 1e-12):  # Avoid division by 0
            info = 0
            print(Fore.GREEN + "Convergence achieved: direction norm too small" + Style.RESET_ALL)
            break

        if alpha < 1e-6:
            info = 2
            print(Fore.YELLOW + "Warning: step size alpha too small — possible stagnation." + Style.RESET_ALL)

        if M is not None:
            z = M @ r
        else:
            z = r
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p

        rz_old = rz_new


    return x, info