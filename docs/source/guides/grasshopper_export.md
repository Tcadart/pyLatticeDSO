# pyLattice ↔︎ Rhino/Grasshopper (GhPython)

Visualize and post-process pyLattice results directly in **Grasshopper** using a lightweight GhPython script.  
This page explains installation, the expected JSON format, how to use the component, and how to troubleshoot.

---

## Export lattice from pyLattice to Grasshopper
Generate a lattice in pyLattice and export it to a JSON file:
```python
from pyLattice.lattice import Lattice
from pyLattice.utils import save_JSON_to_Grasshopper

# Create a lattice instance from a JSON parameter file
path = "design/"
name_file = "simple_BCC"
lattice = Lattice(path + name_file)
# Export the lattice to a JSON file
save_JSON_to_Grasshopper(lattice, name_file)
```

## Construct component in Grasshopper and import the script
Use the provided GhPython script to create a Grasshopper component that reads the lattice JSON file and generates a voxelized mesh using the **Dendro** plugin.
**Remember to configure the path to your JSON file in the script.**

> The GhPython script is provided at: `extras/grasshopper/pylattice_gh.py`.

The component has the following structure (see image below):
![Grasshopper script](../grasshopper_script.png)

- **Inputs**: a lattice name (pointing to a JSON export) `nameLattice`, a boolean to optionally cut the mesh `cutCell`.
- **Outputs**:
  - `lines`: preview lines built from lattice nodes,
  - `mesh`: voxelized volume using **Dendro**,
  - `vol`: the volume of the resulting mesh (Rhino units³).

    
---

## Requirements

- **Rhino 7 or 8**, **Grasshopper**
- **Dendro** plugin (for voxelization)
- A lattice JSON file exported by pyLattice

