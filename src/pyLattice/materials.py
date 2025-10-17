# =============================================================================
# CLASS: MatProperties
# =============================================================================
import json
import os
from pathlib import Path


class MatProperties:
    """
    A class to represent the properties of a material loaded from a file.
    """
    project_root = Path(__file__).resolve().parents[2]
    MATERIALS_DIR = project_root / "src" / "pyLattice" / "materials" # Directory containing material files

    def __init__(self, name_material: str):
        """
        Initialize the MatProperties object by loading data from a file.

        Parameters:
        name_material (str): The name of the material file (without extension) to load.
        """
        self.name_material = None
        self.density = None
        self.young_modulus = None
        self.poisson_ratio = None
        self.plastic = None

        material_file = f"{name_material}.json"
        self.file_path = os.path.join(self.MATERIALS_DIR, material_file)

        self.load_material()

    def load_material(self):
        """
        Loads material properties from a JSON file.

        :return: Material name_lattice, density, elastic properties, and plastic properties
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Material file not found: {self.file_path}")

        with open(self.file_path, "r") as file:
            data = json.load(file)

        self.name_material = data.get("name", None)
        self.density = data.get("density", None)
        self.young_modulus = data.get("Young_modulus", None)
        self.poisson_ratio = data.get("Poisson_ratio", None)
        self.plastic = data.get("plastic", None)


