# =============================================================================
# CLASS: Material
#
# DESCRIPTION:
# This class encapsulates all material properties for a lattice structure,
# including methods to set material parameters, compute mechanical properties,
# and calculate gradients with penalization for topology optimization.
# =============================================================================
import numpy as np
import math

from dolfinx.fem import Constant, Function, functionspace
from ufl import as_vector

from pyLattice.materials import MatProperties
from pyLattice.timing import timing

class Material:
    """Encapsulates all material properties for a lattice structure."""

    def __init__(self, domain):
        self.domain = domain

        # Material properties
        self._E = None
        self._nu = None
        self._G = None
        self._rho = None
        self._g = Constant(self.domain, 9.81)  # Default gravity
        self._kappa = Constant(self.domain, 0.9)  # Shear area coefficient
        self._ES = None
        self._EI1 = None
        self._EI2 = None
        self._GJ = None
        self._GS1 = None
        self._GS2 = None

        # Mechanical properties for beams
        self._initialize_function_spaces()

    @timing.category("materialFEM")
    @timing.timeit
    def _initialize_function_spaces(self):
        """Initialize function spaces for mechanical properties and gradients."""
        V0 = functionspace(self.domain, ("DG", 0))

        # Mechanical properties
        self._ES = Function(V0)
        self._EI1 = Function(V0)
        self._EI2 = Function(V0)
        self._GJ = Function(V0)
        self._GS1 = Function(V0)
        self._GS2 = Function(V0)

        # Gradient properties
        self._ESgrad = Function(V0)
        self._EI1grad = Function(V0)
        self._EI2grad = Function(V0)
        self._GJgrad = Function(V0)
        self._GS1grad = Function(V0)
        self._GS2grad = Function(V0)

        # Neutral radius for gradient computation
        self.radiusNeutral = Function(V0)

    @property
    def E(self):
        if self._E is None:
            raise AttributeError("Elastic modulus E has not been initialized.")
        return self._E

    @property
    def nu(self):
        if self._nu is None:
            raise AttributeError("Poisson's ratio nu has not been initialized.")
        return self._nu

    @property
    def G(self):
        if self._G is None:
            raise AttributeError("Shear modulus G has not been initialized.")
        return self._G

    @property
    def rho(self):
        if self._rho is None:
            raise AttributeError("Density rho has not been initialized.")
        return self._rho

    @property
    def g(self):
        return self._g

    @property
    def properties_for_stress(self):
        """Return mechanical properties as a vector for stress calculations."""
        return as_vector([self._ES, self._GS1, self._GS2, self._GJ, self._EI1, self._EI2])

    @property
    def properties_for_stress_gradient(self):
        """Return gradient of mechanical properties as a vector."""
        return as_vector([self._ESgrad, self._GS1grad, self._GS2grad, self._GJgrad, self._EI1grad, self._EI2grad])

    @timing.category("materialFEM")
    @timing.timeit
    def set_material(self, lattice_material: "MatProperties"):
        """
        Define material properties for linear behavior based on material_name.
        """
        self._E = lattice_material.young_modulus
        self._nu = lattice_material.poisson_ratio
        self._E = Constant(self.domain, self._E)
        self._nu = Constant(self.domain, self._nu)
        G_value = self._E.value / (2 * (1 + self._nu.value))
        self._G = Constant(self.domain, G_value)

    @timing.category("materialFEM")
    @timing.timeit
    def set_density(self, density: float):
        """Define material density."""
        self._rho = Constant(self.domain, density)

    @timing.category("materialFEM")
    @timing.timeit
    def compute_mechanical_properties(self, radius):
        """Compute mechanical properties based on the beam radius."""
        if self._E is None or self._G is None:
            raise RuntimeError("Material properties must be initialized before computing mechanical properties.")

        S = math.pi * radius ** 2
        I = (math.pi * radius ** 4) / 4
        J = 2 * I  # Torsional inertia

        self._ES = self._E * S
        self._EI1 = self._E * I
        self._EI2 = self._E * I
        self._GJ = self._G * J
        self._GS1 = self._G * self._kappa * S
        self._GS2 = self._G * self._kappa * S

    # =============================================================================
    # SECTION: Gradient Computation with Penalized beams
    # =============================================================================
    @timing.category("materialFEM")
    @timing.timeit
    def compute_gradient(self, radius, tagBeam, lattice):
        """
        Compute the gradient of mechanical properties, considering penalization.
        Vectorized version for performance.
        """
        if self._E is None or self._G is None:
            raise RuntimeError("Material properties must be initialized before computing gradients.")

        # Copy radius values
        self.radiusNeutral.x.array[:] = radius.x.array
        radius_array = self.radiusNeutral.x.array  # alias

        # Penalize modified beams (tag == 1)
        tol = 1e-8
        if 1 in tagBeam:
            mod_rads = np.array(list(tagBeam[1]))
            for rad in mod_rads:
                mask = np.abs(radius_array - rad) < tol
                radius_array[mask] /= lattice.penalization_coefficient

        # Initialize tag array
        tag_array = np.full_like(radius_array, -1, dtype=int)
        for tag, rad_set in tagBeam.items():
            for rad in rad_set:
                mask = np.abs(radius_array - rad) < tol
                tag_array[mask] = tag

        if np.any(tag_array == -1):
            print("Warning: Some radius values were not assigned a tag!")

        # Preallocate all arrays
        ES_arr = np.zeros_like(radius_array)
        EI_arr = np.zeros_like(radius_array)
        GJ_arr = np.zeros_like(radius_array)
        GS_arr = np.zeros_like(radius_array)

        # Vectorized computation
        is_mod = tag_array == 1
        is_norm = tag_array == 0

        rad_mod = radius_array[is_mod]
        rad_norm = radius_array[is_norm]
        P = lattice.penalization_coefficient

        # Modified beams
        if np.any(is_mod):
            S_mod = 2 * math.pi * P ** 2 * rad_mod
            I_mod = math.pi * P ** 4 * rad_mod ** 3
            ES_arr[is_mod] = self._E.value * S_mod
            EI_arr[is_mod] = self._E.value * I_mod
            GJ_arr[is_mod] = self._G.value * 2 * I_mod
            GS_arr[is_mod] = self._G.value * self._kappa.value * S_mod

        # Normal beams
        if np.any(is_norm):
            S_norm = 2 * math.pi * rad_norm
            I_norm = math.pi * rad_norm ** 3
            ES_arr[is_norm] = self._E.value * S_norm
            EI_arr[is_norm] = self._E.value * I_norm
            GJ_arr[is_norm] = self._G.value * 2 * I_norm
            GS_arr[is_norm] = self._G.value * self._kappa.value * S_norm

        # Assign values back to dolfinx.Function
        self._ESgrad.x.array[:] = ES_arr
        self._EI1grad.x.array[:] = EI_arr
        self._EI2grad.x.array[:] = EI_arr
        self._GJgrad.x.array[:] = GJ_arr
        self._GS1grad.x.array[:] = GS_arr
        self._GS2grad.x.array[:] = GS_arr


