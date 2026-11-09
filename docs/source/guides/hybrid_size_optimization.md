# Hybrid Size Optimization

The `hybrid_size` optimization adjusts **one beam radius per base geometry and per cell** of a hybrid lattice
(e.g. `["BCC", "Hybrid1", "Hybrid4"]`). It solves the equilibrium of the **full lattice** with the direct FEM solver
(`K u = f`, LU factorization) and computes the **analytic gradient of the compliance**, so each optimization
iteration costs a single linear solve.

It is the FEM counterpart of the `unit_cell` parametrization, which relies on the domain decomposition (DDM) and
surrogate Schur complements (see [Optimization methods](optimization_methods.md)).

---

## Design variables

For a lattice of $N_c$ cells built with $N_g$ base geometries, the design vector is

$$
\mathbf{r} = \left( r_{c,g} \right)_{c = 1..N_c,\; g = 1..N_g}, \qquad N_c \times N_g \text{ parameters}
$$

where $r_{c,g}$ is the radius of all the beams of geometry $g$ in cell $c$. The parameter of index
`cell.index * N_g + g` drives geometry `geom_types[g]` of the cell `cell.index`. The initial value of $r_{c,g}$ is the
radius given for geometry $g$ in `geometry.radii`.

With `enable_parameter_normalization`, the optimizer works on $\theta \in [0, 1]$ with
$r = r_{min} + \theta\,(r_{max} - r_{min})$.

---

## Stiffness dependence on the radii

The beams are modeled with Timoshenko elements. Their stiffness is **linear in the section properties**:

| Section property | Dependence on the radius |
| :--- | :--- |
| Axial stiffness $ES$, shear stiffness $\kappa G S$ | $\propto r^2$ |
| Bending stiffness $EI$, torsional stiffness $GJ$ | $\propto r^4$ |

The stiffness of a beam element $e$ of radius $r_e$ is therefore exactly

$$
K_e(r_e) = r_e^2\, K_e^{(2)} + r_e^4\, K_e^{(4)}
$$

Each beam element belongs to a single (cell, geometry) pair, so the global stiffness is additive over the design
variables:

$$
K(\mathbf{r}) = \sum_{c,g} \; \sum_{e \in (c,g)} \left( r_{c,g}^2\, K_e^{(2)} + r_{c,g}^4\, K_e^{(4)} \right)
$$

Each radius $r_{c,g}$ only affects the elements of its own geometry and cell, with no cross term between geometries.

!!! note "Why not $K_{cell}(\mu) = \mu\, K_{cell}^{ref}$?"
    A single scaling $K = \mu K^{ref}$ holds when $\mu$ multiplies the Young's modulus (SIMP-like interpolation),
    since the elastic bilinear form is linear in $E$. With the radius as design variable, the $r^2$ and $r^4$ terms
    make $K$ a polynomial in $r$ instead of a simple scaling, but the decomposition above is still exact and
    gives an exact gradient.

!!! warning "Node penalization must be disabled"
    The node penalization stiffens the beams close to the nodes: their radius is multiplied by 1.5 over a length that
    depends on the radii and on the angles between beams. The stiffness is then no longer given by the decomposition
    above. The `hybrid_size` optimization therefore requires `"node_penalization": false` in
    `simulation_parameters`.

---

## Analytic gradient of the compliance

The compliance is the work of the applied loads:

$$
C = \mathbf{f}^T \mathbf{u}, \qquad K(\mathbf{r})\, \mathbf{u} = \mathbf{f}
$$

When the loads do not depend on the radii and the imposed displacements are zero, differentiating the equilibrium
gives $K\, \partial \mathbf{u} / \partial r = - (\partial K / \partial r)\, \mathbf{u}$. The problem is self-adjoint,
which gives

$$
\frac{\partial C}{\partial r_{c,g}} = - \mathbf{u}^T \frac{\partial K}{\partial r_{c,g}} \mathbf{u}
= - 2 \sum_{e \in (c,g)} \frac{\partial W}{\partial r_e}\bigg|_{\mathbf{u}},
\qquad W = \frac{1}{2}\, \mathbf{u}^T K \mathbf{u}
$$

No adjoint solve is needed: the gradient comes from the displacement field of the direct solve.

### Implementation

1. The radius is a piecewise-constant field (`DG0`) on the beam mesh, and the section properties are UFL expressions
   of this field. The element sensitivities $\partial W / \partial r_e$ are assembled at once as the derivative of the
   strain energy with respect to the radius field:

    ```python
    strain_energy = 0.5 * ufl.action(ufl.action(k_form, u), u)
    dW_dr = fem.assemble_vector(fem.form(ufl.derivative(strain_energy, radius, ufl.TestFunction(V0))))
    ```

    The derivative is exact. Neither $K^{(2)}$ nor $K^{(4)}$ is assembled.

2. Each beam element is matched to its lattice beam (from the element midpoint), then to its parameter
   $(c, g)$. The element contributions $-2\,\partial W / \partial r_e$ are summed per parameter.

3. The gradient is mapped to the optimizer space (normalization of the radii and of the objective) before being
   returned to SLSQP.

The density constraint is computed with the kriging surrogate model of the relative density of the hybrid cell, with
its exact gradient (same as for `unit_cell`).

---

## Gradient verification

`LatticeOpti.check_gradient(eps)` compares the analytic gradient with a central finite-difference gradient at the
initial parameters, and prints both gradients for every parameter together with their relative error.

The analytic gradient is exact up to the accuracy of the linear solve. The remaining difference comes from the finite
differences, whose error has two terms:

$$
\underbrace{\frac{\varepsilon^2}{6}\, C'''}_{\text{truncation}} \;+\; \underbrace{\mathcal{O}\!\left(\frac{\delta\, |C|}{\varepsilon}\right)}_{\text{round-off}}
$$

where $\delta \approx 10^{-13}$ is the relative accuracy of the compliance given by the LU solve (machine precision
amplified by the conditioning of $K$). On the example below (36 parameters):

| `eps` | Global relative error |
| :--- | :--- |
| 1e-2 | 1.2e-3 |
| 1e-3 | 1.2e-5 |
| 1e-4 | 1.2e-7 |
| **1e-5** | **1.9e-8** |
| 1e-6 | 1.6e-7 |
| 1e-7 | 1.1e-6 |

The error decreases as $\varepsilon^2$ down to $\varepsilon \approx 10^{-5}$ and then increases as $1/\varepsilon$
(round-off). An error in the analytic gradient would instead show up as a plateau that does not depend on
$\varepsilon$.

---

## Usage

Example [`examples/optimization/optimization_hybrid_size.py`](https://github.com/Tcadart/pyLatticeDSO/blob/master/examples/optimization/optimization_hybrid_size.py):
a $3 \times 2 \times 2$ cantilever in bending made of `[BCC, Hybrid1, Hybrid4]` cells, clamped on `Xmin` and loaded on
the `Xmax`/`Zmax` edge.

```python
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeOpti.lattice_opti import LatticeOpti

name_file = "optimization/optimization_hybrid_size"

lattice_object = LatticeOpti(name_file, verbose=0, convergence_plotting=True)

# Gradient check on the first iteration
lattice_object.check_gradient(eps=1e-3)

lattice_object.optimize_lattice()

# Visualization optimized lattice
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_boundary_conditions=True)
```

Input file `data/inputs/preset_lattice/optimization/optimization_hybrid_size.json`:

```json
{
  "geometry": {
    "cell_size": {"x": 1, "y": 1, "z": 1},
    "number_of_cells": {"x": 3, "y": 2, "z": 2},
    "radii": [0.05, 0.05, 0.05],
    "geom_types": ["BCC", "Hybrid1", "Hybrid4"]
  },
  "simulation_parameters": {
    "enable": true,
    "material": "VeroClear",
    "periodicity": false,
    "node_penalization": false
  },
  "boundary_conditions": {
    "Displacement": {
      "Fixed": {
        "Surface": ["Xmin"],
        "DOF": ["X", "Y", "Z", "RX", "RY", "RZ"],
        "Value": [0, 0, 0, 0, 0, 0]
      }
    },
    "Force": {
      "Load": {
        "Surface": ["Xmax", "Zmax"],
        "DOF": ["Z"],
        "Value": [-0.1]
      }
    }
  },
  "optimization_informations": {
    "objective_function": "min",
    "objective_type": "compliance",
    "max_iterations": 30,
    "optimization_parameters": {
      "type": "hybrid_size"
    },
    "constraints": {
      "relative_density": {
        "value": 0.2,
        "compute_gradient": true
      }
    },
    "enable_parameter_normalization": true,
    "simulation_type": "FEM",
    "enable_gradient_computing": true
  }
}
```

On this case, SLSQP converges in 20 iterations (about 30 s), with the compliance reduced to 46 % of its initial
value and the density constraint active ($\bar\rho = 0.200$). The cells close to the loaded edge reach the maximum
radius.

### Requirements

The following conditions are checked when the input file is loaded:

| Setting | Required value | Reason |
| :--- | :--- | :--- |
| `optimization_informations.simulation_type` | `"FEM"` | Direct solve of the full lattice |
| `optimization_informations.objective_type` | `"compliance"` | Only objective with an implemented gradient |
| `simulation_parameters.node_penalization` | `false` | Stiffness depending on the radii through the sections only |
| `boundary_conditions.Displacement.<Name>.Value` | all `0` | Self-adjoint compliance gradient |

A kriging model of the relative density must exist for the combination of geometries
(`data/outputs/relative_densities/surrogate_model/kriging_model_<geometries>`).

### Limitations

* **Beams shared by several cells**: such a beam takes the radius of the last cell that sets it, and its sensitivity
  is given to that cell only. The `BCC`, `Hybrid1` and `Hybrid4` geometries have no shared beams.
* **Objectives**: displacement-based objectives would require an adjoint solve $K \boldsymbol{\lambda} =
  \partial J / \partial \mathbf{u}$, which can reuse the LU factorization of the direct solve. This is not
  implemented yet.
