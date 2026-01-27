# 🔷 Geometry Definitions for Lattice Structures

The unit cell geometries used to generate lattice structures are defined in **JSON files**, which are stored in the 
`src/geometries/` folder of the project.

---

## 📁 Structure of Geometry Files

Each JSON file includes:

* a `name` (string)
* a `description` (string)
* a list of **beams**
* optionally, a dictionary of **parameters**

Minimal example:

```json
{
  "name": "BCC",
  "description": "Body-Centered Cubic [BCC] lattice structure",
  "beams": [
    [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
  ]
}
```

Each beam entry represents a line between two 3D points:

```text
[x1, y1, z1, x2, y2, z2]
```

---

## ⚙️ Parametric Geometries

A geometry file may include a `"parameters"` block that defines named variables or symbolic expressions:

```json
"parameters": {
  "hgeom": 0.35,
  "angleGeom": 20,
  "valGeom": "hgeom - tan(angleGeom * pi / 180) / 2"
}
```

These parameter names can then be referenced in the beam list:

```json
[0.5, 0.0, "hgeom", 0.5, 0.0, "1 - hgeom"]
```

### ✅ Supported Expressions and Functions

Expressions are evaluated using [SymPy](https://www.sympy.org), allowing the use of safe mathematical functions only:

* Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
* Exponential/logarithmic: `exp`, `log`
* Other: `sqrt`, `pi`

> ⚠️ **Do not use `math.` in expressions.**
> Just write `tan(...)`, not `math.tan(...)`.

---

## 🔁 Random Geometry Option

There is also a **random geometry mode**.
If you set `"Random"` in the `geom_types` parameter when creating a lattice, the system will automatically select 
one geometry from the available `.json` files in the `geometries/` folder.

---

## ➕ Adding a New Geometry

To define your own unit cell geometry:

1. Create a `.json` file inside the `src/geometries/` folder.
2. Use the following structure:

```json
{
  "name": "MyNewCell",
  "description": "A custom parametric lattice unit cell",
  "parameters": {
    "a": 0.25,
    "b": "1 - a"
  },
  "beams": [
    [0.0, 0.0, 0.0, "a", "a", "a"],
    ["a", "a", "a", "b", "b", "b"]
  ]
}
```

### Information:

Geometries are defined in a unit cell of size 1x1x1, so the coordinates in the `beams` list should be normalized
to this unit cell size.
