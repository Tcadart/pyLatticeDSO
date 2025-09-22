import numpy as np

class ThinPlateSplineRBF:
    """
    Thin-plate-spline (polyharmonic, k=2) RBF interpolator with a linear polynomial tail.
    Supports vector-valued outputs Y in R^{N x m}.
    Provides evaluation f(X) and analytic gradient ∇f(X).
    """

    def __init__(self, x_train, y_train, reg: float = 0.0):
        """
        Parameters
        ----------
        x_train : (N, d) array_like
            Training inputs (centers).
        y_train : (N,) or (N, m) array_like
            Training targets. If 1D, will be promoted to (N, 1).
        reg : float, optional
            Small Tikhonov regularization added to Phi's diagonal (stabilization on near-singular sets).
        """
        X = np.asarray(x_train, dtype=float)
        Y = np.asarray(y_train, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]                    # (N,) -> (N,1)

        self.x_train = X
        self.y_train = Y
        self.N, self.d = X.shape
        _, self.m = Y.shape

        # Assemble blocks
        Phi = self._phi_matrix(X, X)          # (N, N)
        if reg > 0.0:
            Phi = Phi + reg * np.eye(self.N)

        P = np.hstack([np.ones((self.N, 1)), X])          # (N, d+1)

        # Block system [[Phi, P], [P^T, 0]] [W; CP] = [Y; 0]
        A = np.block([
            [Phi, P],
            [P.T, np.zeros((self.d + 1, self.d + 1))]
        ])                                         # ((N+d+1) x (N+d+1))

        RHS = np.vstack([Y, np.zeros((self.d + 1, self.m))])  # ((N+d+1) x m)

        # Solve for weights W (N x m) and polynomial coeffs CP ((d+1) x m)
        sol = np.linalg.solve(A, RHS)
        self.W  = sol[:self.N, :]               # radial weights
        self.CP = sol[self.N:, :]               # polynomial coefficients [c0; c] stacked in rows

    # ---------- kernels and assembly ----------
    @staticmethod
    def _tps_phi(r):
        """Thin-plate spline kernel φ(r) = r^2 log(r), φ(0)=0 by continuity."""
        out = np.zeros_like(r, dtype=float)
        mask = r > 0
        out[mask] = (r[mask]**2) * np.log(r[mask])
        return out

    def _phi_matrix(self, X1, X2):
        """Pairwise φ(||x - y||) between two sets."""
        r = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=2)  # (N1, N2)
        return self._tps_phi(r)

    # ---------- evaluation ----------
    def evaluate(self, x):
        """
        Evaluate the interpolant at one or many points.

        Parameters
        ----------
        x : (d,) or (M, d) array_like

        Returns
        -------
        f : (m,) if x is (d,), else (M, m)
        """
        Xq = np.asarray(x, dtype=float)
        Xq = Xq[None, :] if Xq.ndim == 1 else Xq    # ensure (M, d)
        M = Xq.shape[0]

        # Phi at queries
        r = np.linalg.norm(Xq[:, None, :] - self.x_train[None, :, :], axis=2)   # (M, N)
        Phi_q = self._tps_phi(r)                                               # (M, N)

        # Polynomial tail at queries
        P_q = np.hstack([np.ones((M, 1)), Xq])                                 # (M, d+1)

        F = Phi_q @ self.W + P_q @ self.CP                                     # (M, m)
        return F[0] if M == 1 else F

    # ---------- gradient ----------
    def gradient(self, x):
        """
        Evaluate the gradient ∇f at one or many points.

        Parameters
        ----------
        x : (d,) or (M, d) array_like

        Returns
        -------
        G : (d, m) if x is (d,), else (M, d, m)
        """
        Xq = np.asarray(x, dtype=float)
        Xq = Xq[None, :] if Xq.ndim == 1 else Xq    # (M, d)
        M = Xq.shape[0]

        # Displacements and distances from each query to centers
        D = Xq[:, None, :] - self.x_train[None, :, :]          # (M, N, d)
        r = np.linalg.norm(D, axis=2)                          # (M, N)

        # Stable factor: (2 log r + 1), zeroed at r=0
        fac = np.zeros_like(r)
        mask = r > 0
        fac[mask] = 2.0 * np.log(r[mask]) + 1.0                # (M, N)

        # Sum_i w_i * (2 log r_i + 1) * (x - x_i)
        # First form (M, N, d) tensor with per-center vectors scaled by fac
        G_core = fac[:, :, None] * D                            # (M, N, d)

        # G_core: (M, N, d),  self.W: (N, m)  →  G: (M, d, m)
        G = np.einsum('qnd,nk->qdk', G_core, self.W)

        # Add gradient of the polynomial tail: CP = [c0; c], so tail grad is c \in R^{d x m}
        c = self.CP[1:, :]                                      # (d, m)
        G += c[None, :, :]                                      # broadcast to (M, d, m)

        return G[0] if M == 1 else G
