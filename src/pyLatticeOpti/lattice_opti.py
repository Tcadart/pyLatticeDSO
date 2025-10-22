# =============================================================================
# CLASS: LatticeOpti
#
# DESCRIPTION:
# This class extends the LatticeSim class to include optimization capabilities.
# It allows for the optimization of lattice parameters using various objective functions and parameterizations.
# It supports constraints such as relative density and can be used with different simulation types (FEM or DDM).
# =============================================================================
import os
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING
import json
from datetime import datetime

import joblib
import numpy as np
from scipy.optimize import NonlinearConstraint, Bounds, minimize
from colorama import Fore, Style

from pyLattice.cell import Cell
from pyLattice.utils import open_lattice_parameters
from pyLatticeSim.conjugate_gradient_solver import conjugate_gradient_solver
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX
from pyLatticeOpti.plotting_lattice_optim import OptimizationPlotter
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeOpti.surrogate_model_relative_densities import _find_path_to_data

if TYPE_CHECKING:
    from data.inputs.mesh_file.mesh_trimmer import MeshTrimmer

from pyLattice.timing import *
timingOpti = Timing()

class LatticeOpti(LatticeSim):
    """
    Superclass of lattice for optimization purposes
    """

    def __init__(self, name_file: str, mesh_trimmer: "MeshTrimmer" = None, verbose: int = 0,
                 convergence_plotting : bool = False):
        enable_domain_decomposition_solver = self._get_DDM_enable_simulation(name_file)
        super().__init__(name_file, mesh_trimmer, verbose, enable_domain_decomposition_solver)
        self._convergence_plotting = convergence_plotting
        self._simulation_flag = True
        self._optimization_flag = True

        self.solution = None
        self.actual_objective = None
        self.denorm_objective = None
        self.initial_value_objective = None
        self.initial_parameters = None
        self.bounds = None
        self.constraints = []
        self.iteration = 0
        self.optim_max_iteration = 10
        self.optim_ftol = 1e-6
        self.optim_disp = True
        self.optim_eps = 1e-3
        self.objectif_data = None
        self.objective_function = None
        self.objective_type = None
        self.actual_optimization_parameters = []
        self._sim_is_current = False # flag to indicate if the simulation is up-to-date with current parameters
        self.number_parameters = None
        self.enable_normalization = False
        self.optimization_parameters = None
        self.constraints_dict = {}
        self.kriging_model_relative_density = None
        self.kriging_model_geometries_types = None
        self.min_radius = 0.01
        self.max_radius = 0.1
        self.radius_field = None  # callable: (x,y,z) -> r
        self.radius_field_info = {}  # metadata (e.g., dirs, centers, widths)
        self._flag_plotting_initialized = False
        self.plotting_densities = []
        self.plotting_objective = []
        self.plotter = None

        self.relative_density_mode = "upper" # "upper" or "lower"
        self.relative_density_tolerance = 0.0
        self.initial_relative_density_constraint = None
        self.relative_density_direct_computation = False
        self.initial_continuity_constraint = None
        self.relative_density_poly = []
        self.relative_density_poly_deriv = []
        self.parameter_optimization = []
        self._Lx = float(self.size_x)
        self._Ly = float(self.size_y)
        self._Lz = float(self.size_z)
        self._x0 = 0.0
        self._y0 = 0.0
        self._z0 = 0.0

        self._history = {
            "iteration": [],
            "objective_norm": [],
            "objective": [],
            "relative_density": [],
            "relative_density_error": [],
            "parameters": [],
            "timestamp": [],
        }

        self._get_optimization_parameters(name_file)
        self._set_number_parameters_optimization()

        self.load_relative_density_model()

    def optimize_lattice(self):
        """
        Runs the optimization process using SLSQP.
        """
        self.initial_value_objective = None
        self.actual_objective = None
        self.denorm_objective = None
        self.iteration = 0
        self._sim_is_current = False
        self.plotting_objective.clear()
        self.plotting_densities.clear()

        self._reset_optimization_history()
        self._initialize_optimization_solver()
        self._add_constraint_density()
        minimize_kwargs = dict(
            fun=self.objective,
            x0=self.initial_parameters,
            method='SLSQP',
            bounds=self.bounds,
            constraints=self.constraints,
            callback=self.callback_function,
            options={
                'maxiter': self.optim_max_iteration,
                'ftol': self.optim_ftol,
                'disp': self.optim_disp,
                'eps': self.optim_eps
            }
        )
        if getattr(self, "enable_gradient_computing", False):
            minimize_kwargs["jac"] = self.gradient

        self._record_iteration(self.initial_parameters)
        self.solution = minimize(**minimize_kwargs)
        self.set_optimization_parameters(self.solution.x)

        if self._convergence_plotting and self.plotter is not None:
            try:
                self.plotter.update(self.actual_objective, self.solution.x)
            except Exception:
                pass
            self.plotter.finalize(block=False)

        if self.solution.success:
            print("\n✅ Optimization succeeded!")

        else:
            print("\n⚠️ Optimization failed!")
            print(self.solution.message)

        final_params = self.denormalize_optimization_parameters(list(map(float, self.actual_optimization_parameters)))
        print("Optimal parameters:", self.solution.x)
        print("Denormalized optimal parameters:", final_params)
        if self.objective_type == "compliance":
            print(f"Final compliance (non-normalized): {self.denorm_objective}")

    def redefine_optim_parameters(self, max_iteration: int = None, ftol: float = None, disp: bool = None,
                                  eps: float = None) -> None:
        """
        Redefine optimization parameters

        Parameters:
        -----------
        max_iteration: int
            Maximum number of iterations for the optimizer
        ftol: float
            Tolerance for termination by the optimizer
        disp: bool
            Whether to display optimization messages
        eps: float
            Step size for numerical approximation of the Jacobian
        """
        if max_iteration is not None:
            self.optim_max_iteration = max_iteration
        if ftol is not None:
            self.optim_ftol = ftol
        if disp is not None:
            self.optim_disp = disp
        if eps is not None:
            self.optim_eps = eps

    def _get_optimization_parameters(self, name_file: str) -> None:
        """
        Define optimization parameters from the input file

        Parameters:
        -----------
        name_file: str
            Name of the input file
        """
        lattice_parameters = open_lattice_parameters(name_file)

        optimization_informations = lattice_parameters.get("optimization_informations", {})
        self.objective_function = optimization_informations.get("objective_function", None)
        self.objective_type = optimization_informations.get("objective_type", None)
        self.objectif_data = optimization_informations.get("objective_data", None)
        self.optim_max_iteration = optimization_informations.get("max_iterations", 20)
        self.constraints_dict = optimization_informations.get("constraints", {})
        self.optimization_parameters = optimization_informations.get("optimization_parameters", None)
        if self.optimization_parameters is None:
            raise ValueError("No optimization parameters defined.")
        self._simulation_type = optimization_informations.get("simulation_type", None)
        if self._simulation_type not in {"FEM", "DDM"}:
            print("Simulation type for optimization:", self._simulation_type)
            raise ValueError("Invalid simulation type for optimization. Choose 'FEM' or 'DDM'.")

        self.enable_normalization = optimization_informations.get("enable_parameter_normalization", False)
        self.enable_gradient_computing = optimization_informations.get("enable_gradient_computing", False)
        if not self.enable_gradient_computing:
            print(Fore.YELLOW + "Warning: Gradient computing is disabled. Optimization may be slow." + Fore.RESET)

    def _get_DDM_enable_simulation(self, name_file) -> bool:
        """
        Check if DDM simulation is enabled
        """
        lattice_parameters = open_lattice_parameters(name_file)
        optimization_informations = lattice_parameters.get("optimization_informations", {}).get("simulation_type", None)
        if optimization_informations == "DDM":
            enable_domain_decomposition_solver = True
        else:
            enable_domain_decomposition_solver = False
        return enable_domain_decomposition_solver

    def _clamp_radius(self, v: float) -> float:
        return max(self.min_radius, min(self.max_radius, float(v)))

    def _initialize_optimization_solver(self):
        """
        Initialize the solver.
        """
        if self.enable_normalization:
            borneMin = 0
            borneMax = 1
        else:
            borneMin = self.min_radius
            borneMax = self.max_radius

        if self.optimization_parameters["type"] == "linear":
            # bornes : pentes dans [-1,1], intercept dans [0,1] (si normalization active)
            if self.enable_normalization:
                lb = [-1.0] * (self.number_parameters - 1) + [0.0]
                ub = [1.0] * (self.number_parameters - 1) + [1.0]
            else:
                # si pas de normalisation globale, garde pentes [-1,1] (unitaires), intercept entre min_radius et max_radius
                lb = [-1.0] * (self.number_parameters - 1) + [self.min_radius]
                ub = [1.0] * (self.number_parameters - 1) + [self.max_radius]
            self.bounds = Bounds(lb=lb, ub=ub)
        else:
            self.bounds = Bounds(lb=[borneMin] * self.number_parameters, ub=[borneMax] * self.number_parameters)

        initial_value = mean(self.normalize_optimization_parameters(self.radii))
        if self.optimization_parameters["type"] == "unit_cell":
            self.initial_parameters = [initial_value] * self.number_parameters
        elif self.optimization_parameters["type"] == "linear":
            if self.radius_field is None:
                self._build_radius_field()
            self.initial_parameters = [0.0] * (self.number_parameters - 1) + [initial_value]
        elif self.optimization_parameters["type"] == "constant":
            if self.optimization_parameters.get("hybrid", False):
                self.initial_parameters = self.normalize_optimization_parameters(self.radii)
            else:
                self.initial_parameters = [initial_value]
        else:
            raise ValueError("Invalid optimization parameters type.")

    def _build_radius_field(self):
        """
        Prepare a parametric radius field r(x,y,z) based on self.optimization_parameters.
        Supported fields:
          - field='linear' with 'direction' subset of ['x','y','z'] -> f=[a, b,c,d] (restricted to listed dirs +
          intercept)
          - field='poly2' with 'terms' subset of ['x','y','z','x2','y2','z2','xy','xz','yz'] + intercept
        """
        opt = self.optimization_parameters or {}
        field_type = opt.get("type", "linear")

        if field_type == "linear":
            dirs = opt.get("direction", ["x", "y", "z"])
            valid = {"x", "y", "z"}
            if any(d not in valid for d in dirs):
                raise ValueError(f"Invalid direction in {dirs}; valid are {valid}.")
            # parameter vector f = [coeff for each dir in order given] + [intercept]
            n = len(dirs) + 1

            def f(x, y, z, theta):
                """Evaluate linear field at (x,y,z) with parameters theta."""
                coeff = dict(zip(dirs, theta[:len(dirs)]))
                d0 = theta[-1]
                return (coeff.get("x", 0.0) * x
                        + coeff.get("y", 0.0) * y
                        + coeff.get("z", 0.0) * z
                        + d0)

            self.radius_field = f
            self.number_parameters = n
            self.radius_field_info = {"type": "linear", "dirs": dirs}

        elif field_type == "poly2":
            # polynomial up to quadratic with chosen terms + intercept
            terms = opt.get("terms", ["x", "y", "z"])  # add 'x2','y2','z2','xy','xz','yz' as needed
            valid = {"x", "y", "z", "x2", "y2", "z2", "xy", "xz", "yz"}
            if any(t not in valid for t in terms):
                raise ValueError(f"Invalid term in {terms}; valid are {valid}.")
            n = len(terms) + 1  # + intercept

            def f(x, y, z, theta):
                """Evaluate poly2 field at (x,y,z) with parameters theta."""
                coeffs = dict(zip(terms, theta[:len(terms)]))
                d0 = theta[-1]
                v = d0
                v += coeffs.get("x", 0.0) * x
                v += coeffs.get("y", 0.0) * y
                v += coeffs.get("z", 0.0) * z
                v += coeffs.get("x2", 0.0) * (x * x)
                v += coeffs.get("y2", 0.0) * (y * y)
                v += coeffs.get("z2", 0.0) * (z * z)
                v += coeffs.get("xy", 0.0) * (x * y)
                v += coeffs.get("xz", 0.0) * (x * z)
                v += coeffs.get("yz", 0.0) * (y * z)
                return v

            self.radius_field = f
            self.number_parameters = n
            self.radius_field_info = {"type": "poly2", "terms": terms}
        else:
            raise ValueError(f"Unknown field type '{field_type}'.")

    def _add_constraint_density(self):
        """
        Add the density constraint to the list of constraints.
        """
        if "relative_density" not in self.constraints_dict:
            return
        self.relative_density_objective = self.constraints_dict.get("relative_density", {}).get("value", None)
        self.relative_density_mode = self.constraints_dict.get("relative_density", {}).get("mode", "upper")
        self.relative_density_tolerance = self.constraints_dict.get("relative_density", {}).get("tolerance", 0.0)
        self.compute_gradient_relative_density = self.constraints_dict.get("relative_density", {}).get("compute_gradient", False)

        if self.relative_density_mode == "upper":
            lower_bound, upper_bound = -np.inf, 0.0  # rho - target <= 0
            function, jacobian = self.density_constraint, self.density_constraint_gradient
        elif self.relative_density_mode == "lower":
            lower_bound, upper_bound = 0.0, np.inf  # rho - target >= 0
            function = lambda r: -self.density_constraint(r)
            jacobian = lambda r: -self.density_constraint_gradient(r)
        elif self.relative_density_mode == "eq":
            lower_bound, upper_bound = 0.0, 0.0  # rho - target == 0
            function, jacobian = self.density_constraint, self.density_constraint_gradient
        elif self.relative_density_mode == "band":
            if self.relative_density_tolerance  <= 0.0:
                raise ValueError("For 'band' mode, a positive 'tolerance' is required.")
            lower_bound, upper_bound = -self.relative_density_tolerance , self.relative_density_tolerance   # |rho - target| <= tol
            function, jacobian = self.density_constraint, self.density_constraint_gradient
        else:
            raise ValueError(f"Invalid relative density mode '{self.relative_density_mode}'. Choose 'upper', 'lower', 'eq', or 'band'.")

        if self.compute_gradient_relative_density is False:
            self.relative_density_direct_computation = True
        density_nl_constraint = NonlinearConstraint(
            fun=function,
            lb=lower_bound,
            ub=upper_bound,
            jac=jacobian
        )
        self.constraints.append(density_nl_constraint)

    def density_constraint(self, r):
        """
        Density constraint function

        Parameters:
        r: list of float
            List of optimization parameters

        """
        print(Fore.LIGHTGREEN_EX + "Density constraint function" + Fore.RESET)
        self.set_optimization_parameters(r)
        densConstraint = self.get_relative_density_constraint()
        # if self.densConstraintInitial is None:
        #     self.densConstraintInitial = densConstraint
        # densConstraint = densConstraint/self.densConstraintInitial
        if self._verbose > 0:
            print("Density constraint: ", densConstraint)
        return densConstraint

    def density_constraint_gradient(self, r):
        """
        Density constraint gradient function

        Parameters:
        r: list of float
            List of optimization parameters
        """
        print(Fore.LIGHTBLUE_EX + "Density constraint gradient function" + Fore.RESET)
        self.set_optimization_parameters(r)
        # gradDensConstraint = self.get_relative_density_gradient_kriging()

        if self.optimization_parameters["type"] != "linear" and self.compute_gradient_relative_density:
            gradDensConstraint = self.get_relative_density_gradient_kriging()
        else:
            gradDensConstraint = self.finite_difference_density_gradient(r, eps=1e-2, scheme="central")
        print("Density constraint gradient: ", gradDensConstraint)
        # gradDensConstraint = gradDensConstraint/self.densConstraintInitial
        return gradDensConstraint


    def get_relative_density_constraint(self) -> float:
        """
        Get relative density of the lattice
        """
        relativeDensity = self.get_relative_density()
        error = relativeDensity - self.relative_density_objective
        if self._verbose > 0:
            print("Relative density: ", relativeDensity)
            print("Relative density maximum: ", self.relative_density_objective)
            print("Relative density error: ", error)
        return error

    def get_relative_density(self) -> float:
        """
        Get mean relative density of all cells in lattice

        Returns:
        --------
        meanRelDens: float
            Mean relative density of the lattice
        """
        cellRelDens = []
        for cell in self.cells:
            if self.relative_density_direct_computation:
                relative_dens = self.generate_mesh_lattice_Gmsh(volume_computation=True, cut_mesh_at_boundary=True,
                                                     save_STL=False, only_relative_density=True,
                                                     cell_index=cell.index)
                print(f"Cell {cell.index} relative density (direct computation): {relative_dens}")
                cellRelDens.append(relative_dens)
            elif self._simulation_flag and self.kriging_model_relative_density is not None:
                cellRelDens.append(cell.get_relative_density_kriging(self.kriging_model_relative_density))
            else:
                cellRelDens.append(cell.relative_density)
        meanRelDens = mean(cellRelDens)
        return meanRelDens

    def get_relative_density_gradient(self) -> list[float]:
        """
        Get relative density gradient of the lattice

        Returns:
        --------
        grad: list of float
            Gradient of relative density
        """
        if len(self.relative_density_poly) == 0:
            self.define_relative_density_function()
        if len(self.cells[0].radii) != len(self.relative_density_poly):
            raise ValueError("Invalid radii data.")

        grad = []
        for cell in self.cells:
            grad.append(cell.get_relative_density_gradient())
        return grad


    def get_relative_density_gradient_kriging(self) -> np.array:
        """
        Get relative density gradient of the lattice using kriging model

        Returns:
        --------
        grad: list of float
            Gradient of relative density
        """
        if self.optimization_parameters["type"] == "unit_cell":
            n_cells = len(self.cells)
            n_geom = len(self.geom_types)
            grad = np.zeros(self.number_parameters, dtype=float)
            scale = (self.max_radius - self.min_radius) if self.enable_normalization else 1.0

            for cell in self.cells:
                g_cell_exact = np.asarray(
                    cell.get_relative_density_gradient_kriging_exact(
                        self.kriging_model_relative_density,
                        self.kriging_model_geometries_types
                    ),
                    dtype=float
                )

                if g_cell_exact.size != n_geom:
                    raise ValueError(f"Gradient size mismatch: expected {n_geom}, got {g_cell_exact.size}")

                start = cell.index * n_geom
                grad[start:start + n_geom] = (g_cell_exact * scale) / n_cells

            return grad
        elif self.optimization_parameters["type"] == "linear":
            numberOfCells = len(self.cells)
            grad = np.zeros(self.number_parameters)
            dirs = self.optimization_parameters.get("direction", [])
            if not dirs:
                raise ValueError("No directions provided for linear optimization.")
            valid_dirs = {"x", "y", "z"}
            if any(d not in valid_dirs for d in dirs):
                raise ValueError(f"Invalid direction in {dirs}; valid are 'x', 'y', 'z'.")
            coeffs = {"x": 0.0, "y": 0.0, "z": 0.0}
            for i, dkey in enumerate(dirs):
                coeffs[dkey] = float(self.initial_parameters[i])
            d_intercept = float(self.initial_parameters[-1])

            for cell in self.cells:
                cx, cy, cz = cell.center_point
                value = coeffs["x"] * cx + coeffs["y"] * cy + coeffs["z"] * cz + d_intercept
                value = max(self.min_radius, min(self.max_radius, value))
                gradient3Geom = cell.get_relative_density_gradient_kriging(
                    self.kriging_model_relative_density, self.kriging_model_geometries_types) / numberOfCells
                for i, dkey in enumerate(dirs):
                    grad[i] += gradient3Geom[0] * cx if dkey == "x" else gradient3Geom[1] * cy if dkey == "y" else gradient3Geom[2] * cz
                grad[-1] += sum(gradient3Geom)
            return grad
        elif self.optimization_parameters["type"] == "constant":
            n_cells = len(self.cells)
            hybrid = bool(self.optimization_parameters.get("hybrid", False))
            n_geom = len(self.geom_types)
            scale = (self.max_radius - self.min_radius) if self.enable_normalization else 1.0

            if hybrid:
                grad = np.zeros(n_geom, dtype=float)
                for cell in self.cells:
                    g_cell = np.asarray(
                        cell.get_relative_density_gradient_kriging(
                            self.kriging_model_relative_density,
                            self.kriging_model_geometries_types
                        ),
                        dtype=float
                    )
                    grad += g_cell
                grad /= n_cells
                grad *= scale
                return grad
            else:
                g_total = 0.0
                for cell in self.cells:
                    g_cell = np.asarray(
                        cell.get_relative_density_gradient_kriging(
                            self.kriging_model_relative_density,
                            self.kriging_model_geometries_types
                        ),
                        dtype=float
                    )
                    g_total += float(np.sum(g_cell))
                g_total /= n_cells
                g_total *= scale
                return np.array([g_total], dtype=float)
        else:
            raise ValueError("Invalid optimization parameters type.")

    def finite_difference_density_gradient(self, r, eps: float = 1e-2, scheme: str = "central") -> np.ndarray:
        """
        Approximate the gradient of the density constraint g(θ) = ρ̄(θ) - ρ_target
        with finite differences in the SAME parameter space as the optimizer (θ).
        """
        r = np.asarray(r, dtype=float).copy()
        n = r.size
        g = np.zeros_like(r)

        def _clamp(val, i):
            return float(min(self.bounds.ub[i], max(self.bounds.lb[i], val)))

        f0 = None
        if scheme in ("forward", "backward"):
            f0 = float(self.density_constraint(r))

        for i in range(n):
            if scheme == "forward":
                rp = r.copy()
                rp[i] = _clamp(rp[i] + eps, i)
                if rp[i] == r[i]:
                    rm = r.copy()
                    rm[i] = _clamp(rm[i] - eps, i)
                    fm = float(self.density_constraint(rm))
                    denom = r[i] - rm[i]
                    g[i] = (f0 - fm) / max(denom, 1e-16)
                else:
                    fp = float(self.density_constraint(rp))
                    denom = rp[i] - r[i]
                    g[i] = (fp - f0) / max(denom, 1e-16)

            elif scheme == "backward":
                rm = r.copy()
                rm[i] = _clamp(rm[i] - eps, i)
                if rm[i] == r[i]:
                    rp = r.copy()
                    rp[i] = _clamp(rp[i] + eps, i)
                    fp = float(self.density_constraint(rp))
                    denom = rp[i] - r[i]
                    g[i] = (fp - f0) / max(denom, 1e-16)
                else:
                    fm = float(self.density_constraint(rm))
                    denom = r[i] - rm[i]
                    g[i] = (f0 - fm) / max(denom, 1e-16)

            else:  # central
                rp = r.copy(); rm = r.copy()
                rp[i] = _clamp(rp[i] + eps, i)
                rm[i] = _clamp(rm[i] - eps, i)

                if rp[i] == rm[i]:
                    rp2 = r.copy()
                    rp2[i] = _clamp(rp2[i] + eps, i)
                    f0c = float(self.density_constraint(r))
                    fp2 = float(self.density_constraint(rp2))
                    denom = rp2[i] - r[i]
                    g[i] = (fp2 - f0c) / max(denom, 1e-16)
                else:
                    fp = float(self.density_constraint(rp))
                    fm = float(self.density_constraint(rm))
                    denom = rp[i] - rm[i]
                    g[i] = (fp - fm) / max(denom, 1e-16)

        self.set_optimization_parameters(list(r))
        return g

    def objective(self, r) -> float:
        """
        Objective function for the optimization

        Parameters:
        -----------
        r: list of float
            List of optimization parameters

        Returns:
        --------
        objectiveValue: float
            Value of the objective function
        """
        if self._verbose >= 1:
            print(Fore.GREEN + "Objective function" + Fore.RESET)
        print("Parameters:", r)
        self.set_optimization_parameters(r)
        if not self._sim_is_current:
            self._simulate_lattice_equilibrium()
        objective = self.calculate_objective()
        print("objective", objective)
        self.denorm_objective = objective
        objectiveNorm = self.normalize_objective(objective)

        if self.objective_function == "max":
            objectiveNorm = -objectiveNorm
        elif self.objective_function == "min":
            pass
        else:
            raise ValueError("objective_function must be 'min' or 'max'")

        print("Normalized objective", objectiveNorm)
        self.actual_objective = objectiveNorm
        print("Actual objective", self.actual_objective)
        return self.actual_objective

    def _ensure_norm_scale_initialized(self, reference_value: float | None) -> None:
        """
        Initialize the normalization scale C_0 once (and only once).
        """
        if not self.enable_normalization:
            self.initial_value_objective = None
            return
        if self.initial_value_objective is None:
            s = abs(float(reference_value)) if reference_value is not None else 1.0
            if s == 0.0:
                s = 1.0  # robust fallback
            self.initial_value_objective = s
            print("Initial objective value (scale): ", self.initial_value_objective)

    def normalize_objective(self, value: float) -> float:
        """
        Return C/C_0 when normalization is enabled; otherwise return C.
        """
        if not self.enable_normalization:
            return float(value)
        if self.initial_value_objective is None:
            self._ensure_norm_scale_initialized(value)

        return float(value) / self.initial_value_objective

    def _to_normalized_theta_space(self, grad_dr: np.ndarray) -> np.ndarray:
        """
        Map gradient from physical radii space (dC/dr) to the optimizer's
        normalized parameter space (d(C/C0)/dθ), where:
          C0 = self.initial_value_objective (first objective value, >0)
          r  = min_radius + θ * (max_radius - min_radius)
        so: d(C/C0)/dθ = (1/C0) * dC/dr * (max_radius - min_radius)
        """
        if not self.enable_normalization:
            return grad_dr
        if self.initial_value_objective in (None, 0.0):
            raise RuntimeError("Normalization scale not initialized; call objective() once before grad.")
        return grad_dr * (self.max_radius - self.min_radius) / self.initial_value_objective


    def _simulate_lattice_equilibrium(self):
        """
        Simulate the lattice equilibrium using the internal solver.
        """
        self._initialize_simulation_parameters()
        if self._simulation_type == "FEM":
            solve_FEM_FenicsX(self)
        elif self._simulation_type == "DDM":
            self.solve_DDM()
        else:
            raise ValueError("Invalid simulation type for optimization. Choose 'FEM' or 'DDM'.")

        self._sim_is_current = True

    def get_radius_continuity_difference(self, delta: float = 0.01) -> list[float]:
        """
        Get the difference in radii between connected beams in the lattice

        Parameters:
        -----------
        delta: float
            Minimum difference in radii between connected cells
        """
        radiusContinuityDifference = []
        for cell in self.cells:
            radiusCell = cell.radii
            for neighbours in cell.get_all_cell_neighbours():
                for rad in range(len(radiusCell)):
                    radiusContinuityDifference.append((radiusCell[rad] - neighbours.radii[rad]) ** 2 - delta ** 2)
        return radiusContinuityDifference

    def get_radius_continuity_jacobian(self) -> np.ndarray:
        """
        Compute the Jacobian of the radii continuity constraint.

        Returns:
        --------
        np.ndarray
            Jacobian matrix of shape (num_constraints, num_radii)
        """
        rows = []
        cols = []
        values = []
        constraint_index = 0

        for cell in self.cells:
            radiusCell = cell.radii
            for neighbour in cell.get_all_cell_neighbours():
                radiusNeighbour = neighbour.radii
                for rad in range(len(radiusCell)):
                    i = cell.index * len(radiusCell) + rad
                    j = neighbour.index * len(radiusCell) + rad
                    diff = radiusCell[rad] - radiusNeighbour[rad]

                    rows.append(constraint_index)
                    cols.append(i)
                    values.append(2 * diff)

                    rows.append(constraint_index)
                    cols.append(j)
                    values.append(-2 * diff)

                    constraint_index += 1

        jacobian = np.zeros((constraint_index, self.get_number_parameters_optimization()))
        for r, c, v in zip(rows, cols, values):
            jacobian[r, c] = v

        return jacobian

    def define_relative_density_function(self, degree: int = 3) -> None:
        """
        Define relative density function
        Possible to define a more complex function with dependency on hybrid cells

        Parameters:
        -----------
        degree: int
            Degree of the polynomial function
        """
        if len(self.relative_density_poly) == 0:
            fictiveCell = Cell([0, 0, 0], [self.cell_size_x, self.cell_size_y, self.cell_size_z], [0, 0, 0],
                               self.geom_types, self.radii, self.grad_radius, self.grad_dim, self.grad_mat,
                               self.uncertainty_node, self._verbose)
            domainRadius = np.linspace(0.01, 0.1, 10)
            for idxRad in range(len(self.radii)):
                radius = np.zeros(len(self.radii))
                relativeDensity = []
                for domainIdx in domainRadius:
                    radius[idxRad] = domainIdx
                    fictiveCell.change_beam_radius([radius])
                    relativeDensity.append(fictiveCell.relative_density())
                poly_coeffs = np.polyfit(domainRadius, relativeDensity, degree).flatten()
                poly = np.poly1d(poly_coeffs)
                self.relative_density_poly.append(poly)
                self.relative_density_poly_deriv.append(poly.deriv())

    def set_optimization_parameters(self, optimization_parameters_actual: list[float]) -> None:
        """
        Set optimization parameters for the lattice

        Parameters:
        -----------
        optimizationParameters: list of float
            List of optimization parameters
        geomScheme: list of bool
            List of N boolean values indicating the scheme of geometry to optimize
        """

        if len(optimization_parameters_actual) != self.number_parameters:
            raise ValueError("Invalid number of optimization parameters.")
        if (    self.actual_optimization_parameters is not None
                and len(self.actual_optimization_parameters) == len(optimization_parameters_actual)
                and np.allclose(optimization_parameters_actual, self.actual_optimization_parameters,
                                rtol=1e-10, atol=1e-10)):
            return

        self._sim_is_current = False
        self.actual_optimization_parameters = optimization_parameters_actual.copy()

        if self._verbose >= 2:
            print(Style.DIM + "Optimization parameters: ", self.actual_optimization_parameters, Style.RESET_ALL)

        if self.optimization_parameters["type"] == "constant":
            hybrid = bool(self.optimization_parameters.get("hybrid", False))
            if hybrid:
                if len(optimization_parameters_actual) != len(self.geom_types):
                    raise ValueError(
                        f"Expected {len(self.geom_types)} parameters for hybrid constant mode, "
                        f"got {len(optimization_parameters_actual)}."
                    )
                per_geom = self.denormalize_optimization_parameters(
                    list(map(float, optimization_parameters_actual))
                )
                for cell in self.cells:
                    cell.change_beam_radius(per_geom)
            else:
                radius = self.denormalize_optimization_parameters(
                    [float(optimization_parameters_actual[0])]
                )[0]
                for cell in self.cells:
                    cell.change_beam_radius([radius] * len(self.geom_types))
        elif self.optimization_parameters["type"] == "unit_cell":
            number_parameters_per_cell = len(self.geom_types)
            for cell in self.cells:
                startIdx = cell.index * number_parameters_per_cell
                endIdx = (cell.index + 1) * number_parameters_per_cell
                param_slice = optimization_parameters_actual[startIdx:endIdx]
                radius = self.denormalize_optimization_parameters(list(map(float, param_slice)))
                cell.change_beam_radius(radius)
        elif self.optimization_parameters["type"] == "linear":
            dirs = self.optimization_parameters.get("direction", [])
            if not dirs:
                raise ValueError("No directions provided for linear optimization.")

            valid_dirs = {"x", "y", "z"}
            if any(d not in valid_dirs for d in dirs):
                raise ValueError(f"Invalid direction in {dirs}; valid are 'x', 'y', 'z'.")

            expected = len(dirs) + 1  # one coeff per listed direction + intercept d
            if len(optimization_parameters_actual) != expected:
                raise ValueError(
                    f"Invalid number of optimization parameters for linear optimization. "
                    f"Expected {expected} (one per {dirs} + intercept)."
                )

            # Build coefficients a, b, c mapped to x, y, z (missing ones = 0), and intercept d
            coeffs = {"x": 0.0, "y": 0.0, "z": 0.0}
            slopes = {dkey: float(optimization_parameters_actual[i]) for i, dkey in enumerate(dirs)}
            d_physical = self.denormalize_optimization_parameters([float(optimization_parameters_actual[-1])])[0]

            span = (self.max_radius - self.min_radius)

            for cell in self.cells:
                cx, cy, cz = cell.center_point
                # coords normalisées
                xh = (cx - self._x0) / max(self._Lx, 1e-16)
                yh = (cy - self._y0) / max(self._Ly, 1e-16)
                zh = (cz - self._z0) / max(self._Lz, 1e-16)

                s = 0.0
                if "x" in dirs: s += slopes["x"] * xh
                if "y" in dirs: s += slopes["y"] * yh
                if "z" in dirs: s += slopes["z"] * zh

                value = d_physical + span * s
                value = max(self.min_radius, min(self.max_radius, value))
                cell.change_beam_radius([float(value)] * len(self.geom_types))
        else:
            raise ValueError("Invalid optimization parameters type.")

        if self._simulation_type == "DDM":
            self._update_DDM_after_geometry_change()


    def denormalize_optimization_parameters(self, r_norm: list[float]) -> list[float]:
        """
        Denormalize optimization parameters

        Parameters:
        -----------
        r_norm: list of float
            List of normalized optimization parameters

        Returns:
        --------
        r: list of float
            List of denormalized optimization parameters
        """
        if not self.enable_normalization:
            return r_norm
        r = []
        for val in r_norm:
            denorm_val = self._clamp_radius(val * (self.max_radius - self.min_radius) + self.min_radius)
            r.append(denorm_val)
        return r

    def normalize_optimization_parameters(self, r: list[float]) -> list[float]:
        """
        Normalize optimization parameters

        Parameters:
        -----------
        r: list of float
            List of denormalized optimization parameters

        Returns:
        --------
        r_norm: list of float
            List of normalized optimization parameters
        """
        if not self.enable_normalization:
            return r
        r_norm = []
        for val in r:
            if val < self.min_radius or val > self.max_radius:
                raise ValueError("Optimization parameter out of bounds.")
            norm_val = (val - self.min_radius) / (self.max_radius - self.min_radius)
            r_norm.append(norm_val)
        return r_norm

    def calculate_objective(self) -> float:
        """
        Calculate objective function for the lattice optimization

        Parameters
        ----------
        typeObjective: str
            Type of objective function to calculate (Compliance...)

        Returns
        -------
        objectiveValue: float
            Objective function value
        """
        if self.objective_type == "compliance":
            objective = self.compute_compliance()
            # objective = self.compute_global_energy_ddm()
            if self._verbose > 2:
                print("Compliance: ", objective)
        elif self.objective_type == "displacement":
            setNode = self.find_point_on_lattice_surface(surfaceNames=self.objectif_data["Surface"])
            displacements = []
            dof_map = {"X": 0, "Y": 1, "Z": 2, "RX": 3, "RY": 4, "RZ": 5}

            for node in setNode:
                for dof in self.objectif_data["DOF"]:
                    if dof not in dof_map:
                        raise ValueError("Invalid degree of freedom index.")
                    displacements.append(node.displacement_vector[dof_map[dof]])

            displacements = np.array(displacements)

            mean_disp = np.mean(displacements)

            if self.objective_function == "max":
                objective = -mean_disp
            elif self.objective_function == "min":
                objective = mean_disp
            else:
                raise ValueError("objective_function must be 'min' or 'max'")

        elif self.objective_type == "displacement_ratio":
            bd_dict = self.boundary_conditions
            if bd_dict.get("Force", None) is not None:
                bd_dict = bd_dict["Force"]
            elif bd_dict.get("Displacement", None) is not None:
                bd_dict = bd_dict["Displacement"]
            else:
                raise ValueError("No boundary conditions defined for displacement ratio objective.")
            nodes_in = self.find_point_on_lattice_surface(surfaceNames=bd_dict["Load"]["Surface"])
            nodes_out = self.find_point_on_lattice_surface(surfaceNames=self.objectif_data["Surface"])
            dof_map = {"X": 0, "Y": 1, "Z": 2, "RX": 3, "RY": 4, "RZ": 5}

            u_in = np.mean(
                [n.displacement_vector[dof_map[d]] for n in nodes_in for d in bd_dict["Load"]["DOF"]])
            u_out = np.mean(
                [n.displacement_vector[dof_map[d]] for n in nodes_out for d in self.objectif_data["DOF"]])

            # Exemple : on veut u_out = - u_in
            # objective = (u_out + u_in) ** 2
            objective = -(u_out * u_in)

        elif self.objective_type == "stiffness":
            raise NotImplementedError("Stiffness objective not implemented yet.")
        else:
            raise ValueError("Invalid objective function type.")
        return objective

    def compute_compliance(self) -> float:
        """
        Compliance (external work of applied loads):
        C = sum_k f_k * u_k
        """
        total = 0.0
        for node in self.nodes:
            u = node.displacement_vector
            for k in range(self.n_DOF_per_node):
                if node.applied_force[k] != 0.0:
                    coefficient = node.applied_force[k] * u[k]
                    if coefficient < 0:
                        print(node)
                        print("Displacement at node", node.index, "DOF", k, ":", u[k])
                        print("Applied force at node", node.index, "DOF", k, ":", node.applied_force[k])
                        print(Fore.YELLOW + "Warning: negative contribution to compliance from node "
                              f"{node.index}, DOF {k}." + Fore.RESET)
                    total += coefficient
        return float(total)

    def compute_global_energy(self) -> float:
        """
        Global external work = global strain energy
        """
        total = 0.0
        for node in self.nodes:
            u = node.displacement_vector
            for k in range(6):
                if node.applied_force[k] != 0.0 or node.fixed_DOF[k]:
                    f = node.applied_force[k] if node.applied_force[k] != 0.0 else node.reaction_force_vector[k]
                    total += f * u[k]
        return 0.5 * total

    def compute_global_energy_ddm(self) -> float:
        x, _ = self.get_global_displacement_DDM()  # vecteur des DOF libres (ordre canonique)
        y = self.calculate_reaction_force_global(x)  # y = S_global @ x
        return 0.5 * float(np.dot(x, y))

    def _set_number_parameters_optimization(self):
        """
        Set number of parameters for optimization
        """
        if self.optimization_parameters["type"] == "unit_cell":
            numParameters = 0
            for cell in self.cells:
                numParameters += len(cell.radii)
            self.number_parameters = numParameters
        elif self.optimization_parameters["type"] == "linear":
            self._build_radius_field()
        elif self.optimization_parameters["type"] == "constant":
            if self.optimization_parameters.get("hybrid", False):
                self.number_parameters = len(self.geom_types)
            else:
                self.number_parameters = 1
        else:
            raise ValueError("Invalid optimization parameters type.")

    def gradient(self, r: list[float]) -> np.ndarray:
        """
        Gradient function for the optimization

        Parameters:
        r: list of float
            List of optimization parameters
        """
        if self._verbose >= 1:
            print(Fore.BLUE + "Gradient function"+ Fore.RESET)

        # Apply parameters and (re)solve to have consistent u on boundary nodes
        self.set_optimization_parameters(r)
        if not self._sim_is_current:
            self._simulate_lattice_equilibrium()

        g_raw = self.calculate_gradient()
        g_raw = - g_raw # i don't know why but a negative sign is needed here
        g = self._to_normalized_theta_space(g_raw)

        # finite_diff_g = self.finite_difference_gradient(r, eps=1e-2, scheme="central")
        # print("Raw gradient:", g)
        # print("Finite-difference gradient:", finite_diff_g)
        # diff = (g - finite_diff_g)
        # print("Gradient difference:", diff)

        # Optionally store for callbacks/plots
        self.actualGradient = g.copy()
        print("Gradient:", g)
        return g

    # def calculate_gradient(self):
    #     """
    #     Compute d(objective)/d(params) for the current state.
    #     Assumes objective_type == 'compliance' with imposed-DOF-only definition.
    #     """
    #     # if self.objective_type != "compliance":
    #     #     raise NotImplementedError("Gradient currently implemented for 'compliance' objective only.")
    #
    #     n_params = self.number_parameters
    #     grad = np.zeros(n_params, dtype=float)
    #     half_factor = 0.5
    #
    #     # Normalization chain rule (dr / dθ) if parameters are normalized in [0,1]
    #     norm_scale = (self.max_radius - self.min_radius) if self.enable_normalization else 1.0
    #
    #     n_geom = len(self.geom_types)
    #     opt_type = self.optimization_parameters["type"]
    #
    #     if opt_type == "unit_cell":
    #         for cell in self.cells:
    #             if cell.node_in_order_simulation is None:
    #                 cell.define_node_order_to_simulate()
    #
    #             # u_cell in the same order used by the Schur complement
    #             u_cell = np.array(cell.get_displacement_at_nodes(cell.node_in_order_simulation), dtype=float).ravel()
    #
    #             # For each local radius parameter (aligned with cell.radii / schur_complement_grads)
    #             for j_local, dS in enumerate(getattr(cell, "schur_complement_gradient", [])):
    #                 # dF_cell = dS/dr_j @ u_cell
    #                 dF_cell = dS @ u_cell  # vector length = (#bnd_nodes_in_order * 6)
    #                 contrib = u_cell @ dF_cell  # scalar
    #
    #                 p_idx = cell.index * n_geom + j_local
    #                 grad[p_idx] += contrib
    #
    #     elif opt_type == "constant":
    #         hybrid = bool(self.optimization_parameters.get("hybrid", False))
    #
    #         if hybrid:
    #             accum = np.zeros(n_geom, dtype=float)
    #             for cell in self.cells:
    #                 if cell.node_in_order_simulation is None:
    #                     cell.define_node_order_to_simulate()
    #                 u_cell = np.array(cell.get_displacement_at_nodes(cell.node_in_order_simulation),
    #                                   dtype=float).ravel()
    #                 for j_local, dS in enumerate(getattr(cell, "schur_complement_gradient", [])):
    #                     dF_cell = dS @ u_cell
    #                     accum[j_local] += float(u_cell @ dF_cell)
    #             grad[:n_geom] = accum
    #         else:
    #             total_contrib = 0.0
    #             for cell in self.cells:
    #                 if cell.node_in_order_simulation is None:
    #                     cell.define_node_order_to_simulate()
    #                 u_cell = np.array(cell.get_displacement_at_nodes(cell.node_in_order_simulation),
    #                                   dtype=float).ravel()
    #                 for dS in getattr(cell, "schur_complement_gradient", []):
    #                     dF_cell = dS @ u_cell
    #                     total_contrib += float(u_cell @ dF_cell)
    #             grad[0] = total_contrib
    #
    #     else:
    #         raise NotImplementedError("Gradient for optimization type '{opt_type}' not implemented yet.")
    #
    #     return grad

    # add these helpers inside class LatticeOpti

    def _build_displacement_rhs_global(self) -> np.ndarray:
        """
        Assemble q = ∂J/∂u for the displacement-type objectives
        in the *global FREE boundary-DOF ordering* used by the Schur solve.

        Supports:
        ---------
        - objective_type == "displacement":
            J = mean(|u_k|) over selected nodes/DOFs.
        - objective_type == "displacement_ratio":
            J = (u_out + u_in)^2  (forces u_out ≈ -u_in, i.e. inverse mechanism).

        Also builds a per-cell mapping to recover adjoint components back to each
        cell block in the *full* (nb_nodes*6) boundary ordering:
            self._adjoint_map = [
                {
                  "offset_full": int,                 # start index of cell block in full concatenation
                  "m_full": int,                      # block length = nb_nodes*6
                  "free_local_idx": List[int],        # positions (0..m_full-1) that are free in this cell
                }, ...
            ]
        """
        # --- 1) free-DOF vector and its index map ---
        x_free, free_idx = self.get_global_displacement_DDM()
        x_free = np.asarray(x_free, dtype=float)
        free_idx = np.asarray(free_idx, dtype=int)
        n_free = x_free.size

        dof_map = {"X": 0, "Y": 1, "Z": 2, "RX": 3, "RY": 4, "RZ": 5}
        q_blocks = []
        offset_full = 0
        adjoint_map = []

        # --- 2) case: standard displacement objective ---
        if self.objective_type == "displacement":
            set_nodes = self.find_point_on_lattice_surface(surfaceNames=self.objectif_data["Surface"])
            target = set(set_nodes)
            comps = [dof_map[d] for d in self.objectif_data["DOF"]]

            n_terms = 0
            for cell in self.cells:
                if cell.node_in_order_simulation is None:
                    cell.define_node_order_to_simulate()

                nb_nodes = len(cell.node_in_order_simulation)
                m_full = nb_nodes * 6
                q_cell = np.zeros(m_full, dtype=float)

                for i_node, node in enumerate(cell.node_in_order_simulation):
                    if node in target:
                        for k in comps:
                            n_terms += 1
                            val = node.displacement_vector[k]
                            q_cell[i_node * 6 + k] = np.sign(val)

                free_local_idx = []
                for i_node, node in enumerate(cell.node_in_order_simulation):
                    for k in range(6):
                        if not node.fixed_DOF[k]:
                            free_local_idx.append(i_node * 6 + k)

                q_blocks.append(q_cell)
                adjoint_map.append({
                    "offset_full": offset_full,
                    "m_full": m_full,
                    "free_local_idx": free_local_idx,
                })
                offset_full += m_full

            if n_terms > 0:
                q_blocks = [qb / max(1, np.sqrt(n_terms)) for qb in q_blocks]

        # --- 3) case: displacement ratio (inverse mechanism) ---
        elif self.objective_type == "displacement_ratio":
            bd_dict = self.boundary_conditions
            if bd_dict.get("Force", None) is not None:
                bd_dict = bd_dict["Force"]
            elif bd_dict.get("Displacement", None) is not None:
                bd_dict = bd_dict["Displacement"]
            else:
                raise ValueError("No boundary conditions defined for displacement ratio objective.")
            nodes_in = self.find_point_on_lattice_surface(surfaceNames=bd_dict["Load"]["Surface"])
            nodes_out = self.find_point_on_lattice_surface(surfaceNames=self.objectif_data["Surface"])
            comps_in = [dof_map[d] for d in bd_dict["Load"]["DOF"]]
            comps_out = [dof_map[d] for d in self.objectif_data["DOF"]]

            # compute mean displacements
            u_in = np.mean([n.displacement_vector[c] for n in nodes_in for c in comps_in])
            u_out = np.mean([n.displacement_vector[c] for n in nodes_out for c in comps_out])

            coeff_out = -u_in
            coeff_in = -u_out

            for cell in self.cells:
                if cell.node_in_order_simulation is None:
                    cell.define_node_order_to_simulate()

                nb_nodes = len(cell.node_in_order_simulation)
                m_full = nb_nodes * 6
                q_cell = np.zeros(m_full, dtype=float)

                for i_node, node in enumerate(cell.node_in_order_simulation):
                    if node in nodes_out:
                        for k in comps_out:
                            q_cell[i_node * 6 + k] += coeff_out / len(nodes_out)
                    if node in nodes_in:
                        for k in comps_in:
                            q_cell[i_node * 6 + k] += coeff_in / len(nodes_in)

                free_local_idx = []
                for i_node, node in enumerate(cell.node_in_order_simulation):
                    for k in range(6):
                        if not node.fixed_DOF[k]:
                            free_local_idx.append(i_node * 6 + k)

                q_blocks.append(q_cell)
                adjoint_map.append({
                    "offset_full": offset_full,
                    "m_full": m_full,
                    "free_local_idx": free_local_idx,
                })
                offset_full += m_full

        else:
            raise NotImplementedError(f"_build_displacement_rhs_global not implemented for {self.objective_type}")

        # --- 4) concatenate & restrict to free DOFs ---
        q_full = np.concatenate(q_blocks) if q_blocks else np.zeros(0, dtype=float)
        q_free = q_full[free_idx]

        if q_free.size != n_free:
            raise RuntimeError(f"Adjoint RHS size mismatch: got {q_free.size}, expected {n_free}")

        self._adjoint_map = adjoint_map
        return q_free

    def _solve_adjoint_vector(self, q: np.ndarray) -> np.ndarray:
        """
        Solve S λ = q using the same Schur matvec used during DDM solves.

        Parameters
        ----------
        q : np.ndarray
            RHS assembled in the global Schur ordering.

        Returns
        -------
        lam : np.ndarray
            Adjoint vector λ in the same global ordering.
        """
        from scipy.sparse.linalg import LinearOperator

        n = q.size

        def matvec(v: np.ndarray) -> np.ndarray:
            return self.calculate_reaction_force_global(v)

        Sop = LinearOperator((n, n), matvec=matvec, dtype=float)
        lam, info = conjugate_gradient_solver(Sop, q, tol=1e-10, maxiter=2000)
        if info != 0 and self._verbose > 0:
            print(Fore.YELLOW + f"Warning: adjoint CG did not fully converge (info={info})." + Fore.RESET)
        return lam

    def calculate_gradient(self):
        """
        Compute d(objective)/d(params) for the current state.
        - 'compliance': u^T (dS/dr) u (per parameter block).
        - 'displacement': λ^T (dS/dr) u with adjoint S λ = ∂J/∂u, J = average of |selected DOFs|.
          NOTE: gradient() applies a global negative sign afterwards.
        """
        if self.objective_type == "compliance":
            n_params = self.number_parameters
            grad = np.zeros(n_params, dtype=float)
            n_geom = len(self.geom_types)
            opt_type = self.optimization_parameters["type"]

            if opt_type == "unit_cell":
                for cell in self.cells:
                    if cell.node_in_order_simulation is None:
                        cell.define_node_order_to_simulate()
                    u_cell = np.array(cell.get_displacement_at_nodes(cell.node_in_order_simulation),
                                      dtype=float).ravel()
                    for j_local, dS in enumerate(getattr(cell, "schur_complement_gradient", [])):
                        dF_cell = dS @ u_cell
                        p_idx = cell.index * n_geom + j_local
                        grad[p_idx] += float(u_cell @ dF_cell)

            elif opt_type == "constant":
                hybrid = bool(self.optimization_parameters.get("hybrid", False))
                if hybrid:
                    accum = np.zeros(n_geom, dtype=float)
                    for cell in self.cells:
                        if cell.node_in_order_simulation is None:
                            cell.define_node_order_to_simulate()
                        u_cell = np.array(cell.get_displacement_at_nodes(cell.node_in_order_simulation),
                                          dtype=float).ravel()
                        for j_local, dS in enumerate(getattr(cell, "schur_complement_gradient", [])):
                            accum[j_local] += float(u_cell @ (dS @ u_cell))
                    grad[:n_geom] = accum
                else:
                    total = 0.0
                    for cell in self.cells:
                        if cell.node_in_order_simulation is None:
                            cell.define_node_order_to_simulate()
                        u_cell = np.array(cell.get_displacement_at_nodes(cell.node_in_order_simulation),
                                          dtype=float).ravel()
                        for dS in getattr(cell, "schur_complement_gradient", []):
                            total += float(u_cell @ (dS @ u_cell))
                    grad[0] = total
            elif opt_type == "linear":
                # Gradient w.r.t. linear field parameters θ = [a (for dirs...), intercept d]
                # r_cell = a_x*x + a_y*y + a_z*z + d, shared by all geometries in a cell
                dirs = self.optimization_parameters.get("direction", ["x", "y", "z"])
                valid_dirs = {"x", "y", "z"}
                if any(d not in valid_dirs for d in dirs):
                    raise ValueError(f"Invalid direction in {dirs}; valid are 'x', 'y', 'z'.")

                expected_n = len(dirs) + 1  # + intercept
                if self.number_parameters != expected_n:
                    raise ValueError(
                        f"Mismatch in number of linear parameters: got {self.number_parameters}, expected {expected_n}."
                    )

                # Rebuild current (denormalized) coefficients to detect clamping activity
                coeffs = {"x": 0.0, "y": 0.0, "z": 0.0}
                for i, dkey in enumerate(dirs):
                    coeffs[dkey] = self.denormalize_optimization_parameters(
                        [float(self.actual_optimization_parameters[i])]
                    )[0]
                d_intercept = self.denormalize_optimization_parameters(
                    [float(self.actual_optimization_parameters[-1])]
                )[0]

                tol = 1e-12  # small tolerance to decide if clamping is active

                for cell in self.cells:
                    if cell.node_in_order_simulation is None:
                        cell.define_node_order_to_simulate()

                    # Current displacement vector on the cell boundary (full local ordering)
                    u_cell = np.array(
                        cell.get_displacement_at_nodes(cell.node_in_order_simulation),
                        dtype=float
                    ).ravel()

                    # Sensitivity dC/dr_cell = sum_j u^T (dS_j/dr) u
                    dC_dr_cell = 0.0
                    for dS in getattr(cell, "schur_complement_gradient", []):
                        dC_dr_cell += float(u_cell @ (dS @ u_cell))

                    # Chain rule to linear parameters (ignore contribution if clamped)
                    cx, cy, cz = cell.center_point
                    r_unclamped = coeffs["x"] * cx + coeffs["y"] * cy + coeffs["z"] * cz + d_intercept
                    active = (self.min_radius + tol < r_unclamped < self.max_radius - tol)
                    if not active:
                        # When the radius is clamped at a bound, ∂r/∂θ ≈ 0 (no push outside the box)
                        continue

                    # Accumulate gradient for each coefficient in the order of 'dirs', then intercept
                    for i, dkey in enumerate(dirs):
                        axis_val = cx if dkey == "x" else cy if dkey == "y" else cz
                        grad[i] += dC_dr_cell * axis_val
                    grad[len(dirs)] += dC_dr_cell  # intercept contribution
            else:
                raise NotImplementedError(f"Gradient for optimization type '{opt_type}' not implemented yet.")
            return grad

        elif self.objective_type == "displacement" or self.objective_type == "displacement_ratio":
            # --- adjoint branch ---
            q_free = self._build_displacement_rhs_global()  # size = n_free
            lam_free = self._solve_adjoint_vector(q_free)  # size = n_free

            n_params = self.number_parameters
            grad = np.zeros(n_params, dtype=float)
            n_geom = len(self.geom_types)
            opt_type = self.optimization_parameters["type"]

            # map from full-position -> index in lam_free
            _, free_idx = self.get_global_displacement_DDM()
            free_idx = np.asarray(free_idx, dtype=int)
            fullpos_to_freepos = {int(fp): i for i, fp in enumerate(free_idx)}

            # walk cells and assemble contributions in the *full* boundary space
            for cell, amap in zip(self.cells, getattr(self, "_adjoint_map", [])):
                if cell.node_in_order_simulation is None:
                    cell.define_node_order_to_simulate()

                # Build Uc_full (length = nb_nodes*6) directly from node.displacement_vector
                Uc_full_list = []
                for node in cell.node_in_order_simulation:
                    Uc_full_list.append(np.asarray(node.displacement_vector, dtype=float))
                Uc_full = np.concatenate(Uc_full_list) if Uc_full_list else np.zeros(0, dtype=float)

                # Build λ_c_full by scattering lam_free entries into the cell full block
                lam_c_full = np.zeros(amap["m_full"], dtype=float)
                base = amap["offset_full"]
                for j_local in amap["free_local_idx"]:
                    gpos = base + j_local  # global full position
                    i_free = fullpos_to_freepos.get(gpos, None)
                    if i_free is not None:
                        lam_c_full[j_local] = lam_free[i_free]

                lam_norm = np.linalg.norm(lam_c_full)
                u_norm = np.linalg.norm(Uc_full)
                print(f"Cell {cell.index}: ||lam_c_full||={lam_norm:.2e}, ||Uc_full||={u_norm:.2e}")

                # accumulate gradient using the local Schur gradients (defined in the same full space)
                if opt_type == "unit_cell":
                    for j_local, dS in enumerate(getattr(cell, "schur_complement_gradient", [])):
                        contrib = float(lam_c_full @ (dS @ Uc_full))
                        p_idx = cell.index * n_geom + j_local
                        grad[p_idx] += contrib

                elif opt_type == "constant":
                    hybrid = bool(self.optimization_parameters.get("hybrid", False))
                    if hybrid:
                        for j_local, dS in enumerate(getattr(cell, "schur_complement_gradient", [])):
                            grad[j_local] += float(lam_c_full @ (dS @ Uc_full))
                    else:
                        total = 0.0
                        for dS in getattr(cell, "schur_complement_gradient", []):
                            total += float(lam_c_full @ (dS @ Uc_full))
                        grad[0] += total
                else:
                    raise NotImplementedError(f"Gradient for optimization type '{opt_type}' not implemented yet.")

            return grad


        else:
            raise NotImplementedError(
                "Gradient currently implemented for 'compliance' and 'displacement' objectives only.")

    def finite_difference_gradient(self, r, eps: float = 1e-6, scheme: str = "central") -> np.ndarray:
        """
        Approximate the gradient of the *normalized* objective w.r.t. the optimizer parameters
        using finite differences, in the same parameter space as SLSQP.

        Parameters
        ----------
        r : array-like
            Current optimization parameters (normalized if enable_normalization=True).
        eps : float
            Finite-difference step (applied in the same space as `r`).
        scheme : {"central","forward","backward"}
            Finite-difference scheme.

        Returns
        -------
        np.ndarray
            Approximate gradient vector (same shape as r).
        """
        r = np.asarray(r, dtype=float).copy()
        n = r.size
        g = np.zeros_like(r)

        # Make sure normalization scale is fixed before perturbations
        if self.enable_normalization and self.initial_value_objective is None:
            _ = self.objective(r)  # sets self.initial_value_objective

        # Helper to clamp to bounds (they are already in the same space as r)
        def _clamp(val, i):
            return float(min(self.bounds.ub[i], max(self.bounds.lb[i], val)))

        # Optionally cache f(r) for 1-sided schemes
        f0 = None
        if scheme in ("forward", "backward"):
            f0 = self.objective(r)

        for i in range(n):
            if scheme == "forward":
                rp = r.copy()
                rp[i] = _clamp(rp[i] + eps, i)
                if rp[i] == r[i]:
                    # cannot move forward -> fallback to backward
                    rm = r.copy()
                    rm[i] = _clamp(rm[i] - eps, i)
                    fm = self.objective(rm)
                    denom = r[i] - rm[i]
                    g[i] = (f0 - fm) / max(denom, 1e-16)
                else:
                    fp = self.objective(rp)
                    denom = rp[i] - r[i]
                    g[i] = (fp - f0) / max(denom, 1e-16)

            elif scheme == "backward":
                rm = r.copy()
                rm[i] = _clamp(rm[i] - eps, i)
                if rm[i] == r[i]:
                    # cannot move backward -> fallback to forward
                    rp = r.copy()
                    rp[i] = _clamp(rp[i] + eps, i)
                    fp = self.objective(rp)
                    denom = rp[i] - r[i]
                    g[i] = (fp - f0) / max(denom, 1e-16)
                else:
                    fm = self.objective(rm)
                    denom = r[i] - rm[i]
                    g[i] = (f0 - fm) / max(denom, 1e-16)

            else:  # central (default)
                rp = r.copy(); rm = r.copy()
                rp[i] = _clamp(rp[i] + eps, i)
                rm[i] = _clamp(rm[i] - eps, i)

                if rp[i] == rm[i]:
                    # both clamped to same value -> try tiny forward step
                    rp2 = r.copy()
                    rp2[i] = _clamp(rp2[i] + eps, i)
                    f0c = self.objective(r)
                    fp2 = self.objective(rp2)
                    denom = rp2[i] - r[i]
                    g[i] = (fp2 - f0c) / max(denom, 1e-16)
                else:
                    fp = self.objective(rp)
                    fm = self.objective(rm)
                    denom = rp[i] - rm[i]
                    g[i] = (fp - fm) / max(denom, 1e-16)

        # Restore the state at r (to keep downstream code consistent)
        self.set_optimization_parameters(list(r))
        if not self._sim_is_current:
            self._simulate_lattice_equilibrium()

        return g


    @timingOpti.timeit
    def load_relative_density_model(self):
        """
        Load the relative density model from a file

        Returns:
        --------
        model: Kriging
            The loaded model
        """
        path_dataset, name_dataset = _find_path_to_data(self)
        path_model = (
            path_dataset.parent
            / "surrogate_model"
            / ("kriging_model_" + name_dataset.replace("RelativeDensities_", ""))
        )

        if not os.path.exists(path_model):
            print(f"Model file not found: {path_model}")
        else:
            kriging_surrogate_dict = joblib.load(path_model)
            self.kriging_model_relative_density = kriging_surrogate_dict["model"]
            self.kriging_model_geometries_types = self.geom_types
            print(f"Loaded relative density model from {path_model}")

    def callback_function(self, r):
        """
        Callback function for the optimization (Printing and plotting)
        """
        self.iteration += 1
        print(f"[Itération {self.iteration}] Objective : {self.actual_objective}")

        if "relative_density" in self.constraints_dict:
            print("Relative density = ", self.get_relative_density())

        if self._convergence_plotting:
            if self.plotter is None:
                self.plotter = OptimizationPlotter(self)
            self.plotter.update(self.actual_objective, r)

        self._record_iteration(list(r))

    def _reset_optimization_history(self) -> None:
        """Clear the in-memory optimization history."""
        self._history = {
            "iteration": [],
            "objective_norm": [],
            "objective": [],
            "relative_density": [],
            "relative_density_error": [],
            "parameters": [],
            "timestamp": [],
        }

    def _record_iteration(self, r: list[float]) -> None:
        """Append current iteration data to the history."""
        try:
            rho = self.get_relative_density()
        except Exception:
            rho = float("nan")
        rho_err = (
            rho - float(self.relative_density_objective)
            if getattr(self, "relative_density_objective", None) is not None
            else float("nan")
        )
        self._history["iteration"].append(int(self.iteration))
        self._history["objective_norm"].append(
            float(self.actual_objective) if self.actual_objective is not None else float("nan")
        )
        self._history["objective"].append(
            float(self.denorm_objective) if self.denorm_objective is not None else float("nan")
        )
        self._history["relative_density"].append(float(rho))
        self._history["relative_density_error"].append(float(rho_err))
        self._history["parameters"].append([float(v) for v in r])
        self._history["timestamp"].append(datetime.now().isoformat() + "Z")

    def save_optimization_json(self, name_file: str) -> None:
        """
        Save optimization data (history + summary + useful metadata) to a JSON file.

        Parameters
        ----------
        name_file : str
            Name of the output JSON file (will be created in data/outputs/optimization_data_files/)
        """
        folder_path = Path(__file__).resolve().parents[2] / "data" / "outputs" / "optimization_data_files"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if name_file is None:
            name_file = "optimization_summary_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".json"
        elif not name_file.lower().endswith(".json"):
            name_file += ".json"
        folder_path /= name_file

        # Constraints metadata (optional fields handled safely)
        rel_dens_cfg = self.constraints_dict.get("relative_density", {}) if hasattr(self, "constraints_dict") else {}
        rel_dens_target = rel_dens_cfg.get("value", None)

        summary = {
            "generated_at": datetime.now().isoformat() + "Z",
            "name_file": getattr(self, "name_file", None) if hasattr(self, "name_file") else None,
            "objective_type": self.objective_type,
            "objective_function": self.objective_function,
            "normalization_enabled": bool(self.enable_normalization),
            "normalization_reference": float(self.initial_value_objective) if self.initial_value_objective else None,
            "simulation_type": self._simulation_type,
            "min_radius": float(self.min_radius),
            "max_radius": float(self.max_radius),
            "relative_density_constraint": {
                "mode": getattr(self, "relative_density_mode", None),
                "target": float(rel_dens_target) if rel_dens_target is not None else None,
                "tolerance": float(getattr(self, "relative_density_tolerance", 0.0)),
            },
            "optimizer": {
                "method": "SLSQP",
                "max_iterations": int(self.optim_max_iteration),
                "ftol": float(self.optim_ftol),
                "eps": float(self.optim_eps),
                "disp": bool(self.optim_disp),
            },
            "solution": {
                "success": bool(getattr(self.solution, "success", False)) if self.solution is not None else False,
                "message": getattr(self.solution, "message", None) if self.solution is not None else None,
                "nit": int(getattr(self.solution, "nit", self.iteration)),
                "final_parameters": [float(v) for v in (self.solution.x if self.solution is not None else [])],
                "final_objective_norm": float(self.actual_objective) if self.actual_objective is not None else None,
                "final_objective": float(self.denorm_objective) if self.denorm_objective is not None else None,
                "final_relative_density": float(self.get_relative_density()) if len(self.cells) > 0 else None,
            },
            "history": self._history,
        }

        with folder_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Optimization summary saved to {folder_path}")




