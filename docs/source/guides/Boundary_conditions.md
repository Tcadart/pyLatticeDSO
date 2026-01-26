# Boundary Conditions — Configuration Guide

This guide explains **how to describe boundary conditions** for a lattice structure using a JSON file, based on the example used in `example/simulation_lattice.py`.

---

## Minimal JSON example

```json
{
  "geometry": {
    "cell_size": { "x": 1, "y": 1, "z": 1 },
    "number_of_cells": { "x": 6, "y": 3, "z": 3 },
    "radii": [0.1],
    "geom_types": ["BCC"]
  },
  "simulation_parameters": {
    "enable": true,
    "material": "VeroClear",
    "periodicity": false
  },
  "boundary_conditions": {
    "Displacement": {
      "Fixed": {
        "Surface": ["Xmin"],
        "DOF": ["X", "Y", "Z", "RX", "RY", "RZ"],
        "Value": [0, 0, 0, 0, 0, 0]
      },
      "Displacement": {
        "Surface": ["Xmax", "Zmax"],
        "DOF": ["Z"],
        "Value": [-0.01]
      }
    },
    "Force": {
      "Force": {
        "Surface": ["Xmax", "Zmin"],
        "DOF": ["Y"],
        "Value": [0.025]
      }
    }
  }
}
```

---

## Structure of the `boundary_conditions` block

* **`Displacement`**: prescribed displacements (Dirichlet BCs).
* **`Force`**: applied loads (Neumann BCs)

Each entry (e.g., `"Fixed"`, `"Displacement"`, `"Force"`) follows the same schema:

* `Surface` *(list\[str], required)*
  Combination of index extrema: `Xmin`, `Xmax`, `Ymin`, `Ymax`, `Zmin`, `Zmax`, `Xmid`, `Ymid`, `Zmid`.
  The order **matters**: filtering is **iterative** (see below).
* `DOF` *(list\[str], required)*
  Degrees of freedom: `X`, `Y`, `Z`, `RX`, `RY`, `RZ`.
* `Value` *(list\[float], required)*
  Same length and order as `DOF`.
  Units: translations → length → mm; rotations → radians.
* `SurfaceCells` *(list\[str], optional — not used in the example)*
  Select the nodes on the surface of the cell, if not indicated the same surface in `Surface` is used.

---

## Surface selection

Selection of cells based on their indices is done iteratively:

* `["Xmin"]`: all cells at the **minimum X** index.
* `["Xmax","Zmax"]`: from `Xmax`, keep only those with **maximum Z**.
* `["Xmax","Zmin"]`: from `Xmax`, keep only those with **minimum Z**.

> **Important:** The order of items in `Surface` defines the sequence of successive filters.

---

## DOF mapping

| DOF | Meaning          | Index |
| --- | ---------------- |-------|
| X   | Translation in X | 0     |
| Y   | Translation in Y | 1     |
| Z   | Translation in Z | 2     |
| RX  | Rotation about X | 3     |
| RY  | Rotation about Y | 4     |
| RZ  | Rotation about Z | 5     |

---

## Reading the example, step by step

### 1) Clamp the `Xmin` side

```json
"Fixed": {
  "Surface": ["Xmin"],
  "DOF": ["X", "Y", "Z", "RX", "RY", "RZ"],
  "Value": [0, 0, 0, 0, 0, 0]
}
```

* Selection: cells with **minimum X** index.
* Effect: all DOFs fixed to 0 → **built-in clamp** on the `Xmin` face.

### 2) Downward displacement on the top front edge

```json
"Displacement": {
  "Surface": ["Xmax", "Zmax"],
  "DOF": ["Z"],
  "Value": [-0.01]
}
```

* Selection: first `Xmax`, then among those `Zmax` → top-most region at `Xmax`.
* Effect: imposed **Z** displacement of **−0.01**.

### 3) Lateral force on the bottom rear edge

```json
"Force": {
  "Force": {
    "Surface": ["Xmax", "Zmin"],
    "DOF": ["Y"],
    "Value": [0.025]
  }
}
```

* Selection: first `Xmax`, then `Zmin` → bottom-most region at `Xmax`.
* Effect: **Y**-direction force (or equivalent load) of **+0.025**.

---

## Best practices

* **Match `DOF` and `Value`**: identical lengths and same ordering.
* **Surface order**: think about surface order (e.g., `Xmax` then `Zmin` may differ from `Zmin` then `Xmax` if 
  lattice design is complex).
* **BC conflicts**: avoid imposing different conditions on the **same node/DOF**.
* **Units & sign**: be consistent with the global coordinate system and solver conventions.


---

## Quick troubleshooting

* **Plotting simulation results**: use plotting at your adavantage to visualize the lattice and boundary conditions.
```python
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(
    enable_system_coordinates=True,
    deformedForm=True,
    enable_boundary_conditions=True
)
```
* **Instability**: ensure at least one region is **properly fixed** (avoid free rigid-body modes).
* **Loads vs. displacements**: don’t impose a displacement and a force **on the same node/DOF** without a clear strategy (can over-constrain).
