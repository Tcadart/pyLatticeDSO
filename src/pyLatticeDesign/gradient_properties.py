"""
Gradient properties module.
"""

import math
import random
from enum import Enum
from .timing import timing



def grad_settings_constant(num_cells_x: int, num_cells_y: int, num_cells_z: int, material_gradient: bool = False) -> (
        list)[list]:
    """
    Generate constant gradient settings (i.e., all values = 1.0).

    Parameters:
    -----------
    num_cells_x : int
        Number of cells in the x-direction.

    num_cells_y : int
        Number of cells in the y-direction.

    num_cells_z : int
        Number of cells in the z-direction.

    material_gradient : bool
        If True, return a 3D list for material gradient; otherwise, return a flat list.

    Returns:
    --------
    list[list[float]]:
        A list of [1.0, 1.0, 1.0] repeated for the total number of cells.
    """
    if material_gradient:
        return [[[1 for _ in range(num_cells_x)] for _ in range(num_cells_y)] for _ in range(num_cells_z)]
    else:
        total_cells = num_cells_x * num_cells_y * num_cells_z
        return [[1.0, 1.0, 1.0] for _ in range(total_cells)]

@timing.category("gradient_properties")
@timing.timeit
def get_grad_settings(num_cells_x, num_cells_y, num_cells_z, grad_properties: list) -> list[list[float]]:
    """
    Generate gradient settings based on the provided rule, direction, and parameters.

    Parameters:
    -----------
    num_cells_x : int
        Number of cells in the x-direction.

    num_cells_y : int
        Number of cells in the y-direction.

    num_cells_z : int
        Number of cells in the z-direction.

    gradProperties: list[Rule, Direction, Parameters]
        All types of properties for gradient definition.

    Return:
    ---------
    gradientData: list[list[float]]
        Generated gradient settings (list of lists).
    """

    class GradientRule(str, Enum):
        """
        Enum for different gradient rules.
        """
        CONSTANT = "constant"
        LINEAR = "linear"
        PARABOLIC = "parabolic"
        SINUSOIDE = "sinusoide"
        EXPONENTIAL = "exponential"

    def compute_gradient_factor(i: int, total_cells: int, param_value: float, rule: GradientRule) -> float:
        """
        Compute the gradient factor for a given rule and cell index.

        Parameters:
            i : int
                Index of the current cell.
            total_cells : int
                Total number of cells in the direction.
            param_value : float
                Strength of the gradient.
            rule : GradientRule
                Type of gradient.

        Returns:
            float
                Computed gradient factor.
        """
        if total_cells <= 0:
            raise ValueError("total_cells must be greater than 0")

        mid = total_cells / 2
        match rule:
            case GradientRule.CONSTANT:
                return 1.0
            case GradientRule.LINEAR:
                return 1.0 + i * param_value
            case GradientRule.PARABOLIC:
                if i < mid:
                    return 1.0 + (i / mid) * param_value
                else:
                    return 1.0 + ((total_cells - i - 1) / mid) * param_value
            case GradientRule.SINUSOIDE:
                return 1.0 + param_value * math.sin((i / total_cells) * math.pi)
            case GradientRule.EXPONENTIAL:
                return 1.0 + math.exp(i * param_value)
            case _:
                raise ValueError(f"Unknown gradient rule: {rule}")

    # Extract gradient properties
    rule, direction, parameters = grad_properties

    # Determine the number of cells in each direction
    number_cells = [num_cells_x, num_cells_y, num_cells_z]

    indices = [0, 0, 0]

    gradientData = []

    for _ in range(max(number_cells)):
        gradientData.append([
            compute_gradient_factor(indices[dim], number_cells[dim], parameters[dim], rule) if direction[
                                                                                                   dim] == 1 else 1.0
            for dim in range(3)
        ])

        for dim in range(3):
            if direction[dim] == 1 and indices[dim] < number_cells[dim] - 1:
                indices[dim] += 1
    return gradientData


@timing.category("gradient_properties")
@timing.timeit
def grad_material_setting(numCellsX, numCellsY, numCellsZ, gradMatProperty: list) -> list:
    """
    Define gradient material settings.

    Parameters:
    ------------
    gradMatProperty: list[Multimat, GradMaterialDirection]
        Set of properties for material gradient.

    Returns:
    --------
    grad_mat: list
        3D list representing the material type_beam in the structure.
    """
    multimat, direction = gradMatProperty
    print(f"Gradient material setting: multimat={multimat}, direction={direction}")

    # Initialize grad_mat based on `multimat` value
    if multimat == -1:  # Random materials
        return [[[random.randint(1, 3) for _ in range(numCellsX)] for _ in range(numCellsY)] for _ in
                range(numCellsZ)]

    elif multimat == 0:  # Single material
        return [[[1 for _ in range(numCellsX)] for _ in range(numCellsY)] for _ in range(numCellsZ)]

    elif multimat == 1:  # Graded materials
        # Generate gradient based on the direction
        return [
            [
                [
                    (X + 1) if direction == 1 else X if direction != 1 else X
                    if False else
                    (Y + 1) if direction == 2 else Y
                    if direction != 2 else Y
                    if False else
                    (Z + 1) if direction == 3 else Z
                    for X in range(numCellsX)
                ]
                for Y in range(numCellsY)
            ]
            for Z in range(numCellsZ)
        ]
    else:
        # Default case: return an empty grad_mat if no valid `multimat` is provided
        return []
