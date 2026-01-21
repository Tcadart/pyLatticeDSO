# =============================================================================
# FUNCTIONS: handle geometry JSON files and evaluate symbolic expressions
# =============================================================================

import json
from pathlib import Path
import random
from typing import Union

from sympy import sin, cos, tan, asin, acos, atan, exp, log, sqrt, pi, sympify

SAFE_FUNCTIONS = {
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "asin": asin,
    "acos": acos,
    "atan": atan,
    "exp": exp,
    "log": log,
    "sqrt": sqrt,
    "pi": pi,
}


def evaluate_symbolic_expression(expr: Union[str, float, int], local_vars: dict) -> float:
    """
    Evaluate a symbolic expression, converting it to a float.
    """
    if isinstance(expr, (int, float)):
        return float(expr)
    try:
        context = {**SAFE_FUNCTIONS, **local_vars}
        result = sympify(expr, locals=context)
        return float(result.evalf()) if hasattr(result, "evalf") else float(result)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expr}': {e}\n"
                         f"Tip: remove 'math.' and use functions like tan(), pi directly.")


def get_beam_structure(lattice_type: str) -> list[list[float]]:
    """
    Get the beam structure based on the lattice type from the geometry data in JSON format,
    evaluating symbolic parameters if present.

    Parameters:
    -----------
    lattice_type : str
        Name of the lattice geometry, matching the filename (without .json extension)

    Returns:
    --------
    list of list of float
        List of beam definitions with all coordinates evaluated.
    """
    project_root = Path(__file__).resolve().parent.parent

    # Handle "Random" case by picking a random JSON file in the directory
    if lattice_type == "Random":
        json_files = list((project_root / "geometries").glob("*.json"))
        if not json_files:
            raise FileNotFoundError("No geometry JSON files found in 'geometries' directory.")
        json_path = random.choice(json_files)
    else:
        json_path = project_root / "geometries" / f"{lattice_type}.json"

    try:
        with open(json_path, 'r') as file:
            geometry = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Error in geometry json file (Wrong formatting) '{json_path}': {e.msg} at line {e.lineno}, "
            f"column {e.colno}") from e
    except FileNotFoundError:
        raise FileNotFoundError(f"Geometry file '{json_path}' not found.")

    param_dict = {}
    if "parameters" in geometry:
        # Evaluate all parameter definitions first
        for key, val in geometry["parameters"].items():
            param_dict[key] = evaluate_symbolic_expression(val, {**param_dict, "pi": pi, "tan": tan})

    # Evaluate beams using resolved parameters
    resolved_beams = []
    for beam in geometry["beams"]:
        resolved_beam = [evaluate_symbolic_expression(coord, {**param_dict, "pi": pi, "tan": tan}) for coord in beam]
        resolved_beams.append(resolved_beam)

    return resolved_beams