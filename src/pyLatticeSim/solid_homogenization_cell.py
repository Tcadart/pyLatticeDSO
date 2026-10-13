# =============================================================================
# CLASS: SolidHomogenizedCell
#
# DESCRIPTION:
# This class performs the homogenization of a lattice unit cell meshed as a 3D solid (SolidModel).
# The displacement is split into a macroscopic part E.x and a periodic fluctuation, and the homogenized
# stiffness matrix is obtained from the average stress of 6 unit strain cases.
# =============================================================================
from typing import TYPE_CHECKING

import numpy as np
import ufl
from colorama import Fore, Style

from pyLatticeDesign.timing import timing
from .homogenization_cell import HomogenizationPostProcessing
from .simulation_base import set_direct_lu_solver
from .solid_lattice_simulation import SolidLatticeSimulation

if TYPE_CHECKING:
    from .solid_model import SolidModel


def _import_dolfinx_fem():
    try:
        from dolfinx import fem as _fem  # type: ignore
        return _fem
    except Exception as e:
        err = f"{e!r}"
        class _Missing:
            def __getattr__(self, _name):
                raise RuntimeError(
                    "dolfinx is required at runtime. For documentation builds this import is mocked. "
                    f"Original import error: {err}"
                )
        return _Missing()


def _import_dolfinx_mpc():
    try:
        import dolfinx_mpc as _mpc  # type: ignore
        return _mpc
    except Exception as e:
        err = f"{e!r}"
        class _Missing:
            def __getattr__(self, _name):
                raise RuntimeError(
                    "dolfinx_mpc is required at runtime. For documentation builds this import is mocked. "
                    f"Original import error: {err}"
                )
        return _Missing()


fem = _import_dolfinx_fem()


class SolidHomogenizedCell(HomogenizationPostProcessing, SolidLatticeSimulation):
    """
    Homogenization of a lattice unit cell meshed as a 3D solid.

    For each unit macroscopic strain E (6 cases, same convention as the beam homogenization HomogenizedCell),
    the displacement u = E.x + u_per is computed with u_per periodic, and the column of the homogenized stiffness
    matrix is the average stress over the cell (bounding box volume, voids included).

    Parameters:
    -----------
    SolidModel: SolidModel
        Solid model of a single cell, built with periodic_mesh=True.

    element_degree: int
        Degree of the Lagrange elements for the displacement (1: linear, 2: quadratic).

    verbose: int
        Verbosity level.
    """

    def __init__(self, SolidModel: "SolidModel", element_degree: int = 2, verbose: int = 1):
        super().__init__(SolidModel, element_degree=element_degree, verbose=verbose)
        lattice = SolidModel.lattice
        self._mins = np.array([lattice.x_min, lattice.y_min, lattice.z_min], dtype=float)
        self._maxs = np.array([lattice.x_max, lattice.y_max, lattice.z_max], dtype=float)
        self.cell_volume = float(np.prod(self._maxs - self._mins))

        self._mpc = None
        self._solver = None
        self.homogenizeMatrix = None
        self.orthotropicMatrix = None
        self._symmetryError = None
        self.saveDataToExport = None

    # =============================================================================
    # SECTION: Boundary conditions
    # =============================================================================
    def _on_max_faces(self, x: np.ndarray) -> np.ndarray:
        tol = self.SolidModel.tolerance
        return np.logical_or.reduce([np.isclose(x[i], self._maxs[i], atol=tol) for i in range(3)])

    @timing.category("solid_homogenization")
    @timing.timeit
    def apply_dirichlet_for_homogenization(self):
        """
        Block the rigid body translation of the periodic fluctuation by fixing the mesh node closest to the cell
        center (excluding the max faces, whose nodes are periodic slaves).
        """
        x = self.domain.geometry.x
        candidates = np.flatnonzero(~self._on_max_faces(x.T))
        center = 0.5 * (self._mins + self._maxs)
        node = x[candidates[np.argmin(np.linalg.norm(x[candidates] - center, axis=1))]]
        tol = self.SolidModel.tolerance
        dofs = fem.locate_dofs_geometrical(
            self._V, lambda p: np.logical_and.reduce([np.isclose(p[i], node[i], atol=tol) for i in range(3)]))
        self._bcs = [fem.dirichletbc(np.zeros(3, dtype=np.float64), dofs, self._V)]

    @timing.category("solid_homogenization")
    @timing.timeit
    def periodic_boundary_condition(self):
        """
        Periodic constraint on the displacement fluctuation: every dof on a max face is tied to the dof at the
        translated position on the opposite min faces (edges and corners are mapped in one step to the min corner).
        """
        mpc = _import_dolfinx_mpc()
        lengths = self._maxs - self._mins
        tol = self.SolidModel.tolerance

        def periodic_relation(x):
            out = x.copy()
            for i in range(3):
                on_max = np.isclose(x[i], self._maxs[i], atol=tol)
                out[i][on_max] -= lengths[i]
            return out

        self._mpc = mpc.MultiPointConstraint(self._V)
        self._mpc.create_periodic_constraint_geometrical(self._V, self._on_max_faces, periodic_relation, self._bcs)
        self._mpc.finalize()

    # =============================================================================
    # SECTION: Solving methods
    # =============================================================================
    def find_imposed_strain(self, case: int):
        """
        Macroscopic displacement E.x of a unit strain case (same convention as HomogenizedCell).

        Parameters:
        -----------
        case: int from 1 to 6
            1: X, 2: Y, 3: Z, 4: XY, 5: XZ, 6: YZ
        """
        x = ufl.SpatialCoordinate(self.domain)
        imposed = {1: [x[0], 0, 0], 2: [0, x[1], 0], 3: [0, 0, x[2]],
                   4: [x[1], x[0], 0], 5: [x[2], 0, x[0]], 6: [0, x[2], x[1]]}
        if case not in imposed:
            raise ValueError("Invalid case number. Must be between 1 and 6.")
        return ufl.as_vector(imposed[case])

    @timing.category("solid_homogenization")
    @timing.timeit
    def initialize_solver(self):
        """
        Assemble the constrained stiffness matrix once and factorize it for the 6 loading cases.
        """
        from petsc4py import PETSc
        mpc = _import_dolfinx_mpc()
        if self._mpc is None:
            raise RuntimeError("periodic_boundary_condition must be called before solving.")
        if self._solver is None:
            self._a_form = fem.form(self._k_form)
            A = mpc.assemble_matrix(self._a_form, self._mpc, bcs=self._bcs)
            A.assemble()
            self._solver = PETSc.KSP().create(self._COMM)
            self._solver.setOperators(A)
            self._solver.setType(PETSc.KSP.Type.PREONLY)
            if PETSc.Sys.hasExternalPackage("mumps"):
                self._solver.getPC().setType(PETSc.PC.Type.LU)
                self._solver.getPC().setFactorSolverType("mumps")
            else:
                set_direct_lu_solver(self._solver.getPC())

    @timing.category("solid_homogenization")
    @timing.timeit
    def solve_loading_case(self, case: int) -> "fem.Function":
        """
        Solve the periodic fluctuation of a unit strain case.

        Returns:
        --------
        fem.Function
            Total displacement E.x + u_per.
        """
        from petsc4py import PETSc
        mpc = _import_dolfinx_mpc()
        self.initialize_solver()

        w = self.find_imposed_strain(case)
        l_form = fem.form(-ufl.inner(self.stress(w), self.strain(self._u_)) * self._dx)
        b = mpc.assemble_vector(l_form, self._mpc)
        mpc.apply_lifting(b, [self._a_form], [self._bcs], self._mpc)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self._bcs)

        u_per = fem.Function(self._mpc.function_space)
        self._solver.solve(b, u_per.x.petsc_vec)
        u_per.x.scatter_forward()
        self._mpc.backsubstitution(u_per)
        b.destroy()

        u_total = fem.Function(self._V, name="Displacement")
        u_total.interpolate(fem.Expression(w + u_per, self._V.element.interpolation_points()))
        return u_total

    @timing.category("solid_homogenization")
    @timing.timeit
    def calculate_macro_stress(self, u_total: "fem.Function") -> np.ndarray:
        """
        Average stress over the cell in Voigt order [xx, yy, zz, xy, xz, yz].
        """
        sigma = self.stress(u_total)
        components = [(0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1)]
        values = [self._COMM.allreduce(fem.assemble_scalar(fem.form(sigma[i, j] * self._dx)))
                  for i, j in components]
        return np.array(values) / self.cell_volume

    @timing.category("solid_homogenization")
    @timing.timeit
    def solve_full_homogenization(self) -> np.ndarray:
        """
        Solve the 6 unit strain cases and build the homogenized stiffness matrix.

        Returns:
        --------
        np.ndarray
            6x6 homogenized stiffness matrix (Voigt notation).
        """
        import dolfinx.fem.petsc  # noqa: F401  (fem.petsc used in solve_loading_case)
        columns = []
        self.saveDataToExport = []
        for case in range(1, 7):
            u_total = self.solve_loading_case(case)
            columns.append(self.calculate_macro_stress(u_total))
            self.saveDataToExport.append(u_total)
            if self._verbose > 0:
                print(f"Loading case {case}/6 solved")
        self.u = self.saveDataToExport[-1]
        self.homogenizeMatrix = np.column_stack(columns)
        self.convert_to_orthotropic_form()
        self.compute_errors()
        self.homogenizeMatrix = 0.5 * (self.homogenizeMatrix + self.homogenizeMatrix.T)  # Ensure symmetry
        if self._verbose > 0:
            print(Fore.GREEN + "Homogenization solved" + Style.RESET_ALL)
        return self.homogenizeMatrix

    def get_relative_density(self) -> float:
        """Volume of the solid divided by the volume of the cell."""
        return self.SolidModel.get_volume() / self.cell_volume
