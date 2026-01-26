"""
List of functions to transform the lattice structure in different ways.

These functions can be used to modify the lattice geometry, such as attracting points, curving the lattice, 
applying cylindrical transformations, and fitting to surfaces.
"""
import math
from .point import Point
from .timing import timing

@timing.category("design_transformation")
@timing.timeit
def attractor_lattice(lattice, PointAttractorList: list[float] = None, alpha: float = 0.5,
                      inverse: bool = False) -> None:
    """
    Attract lattice to a specific point

    Parameters:
    -----------
    PointAttractor: list of float in dim 3
        Coordinates of the attractor point (default: None)

    alpha: float
        Coefficient of attraction (default: 0.5)

    inverse: bool
        If True, points farther away are attracted less (default: False)
    """

    def movePointAttracted(point, attractorPoint, alpha_coeff, inverse_bool):
        """
        Move point1 relative from attractorPoint with coefficient alpha

        Parameters:
        -----------
        point: point object
            point to move
        attractorPoint: point object
            Attractor point
        alpha_coeff: float
            Coefficient of attraction
        inverse: bool
            If True, points farther away are attracted less
        """
        Length = point.distance_to(attractorPoint)
        if inverse_bool:
            factor = alpha_coeff / Length if Length != 0 else alpha_coeff
        else:
            factor = alpha_coeff * Length

        DR = [(attractorPoint.x - point.x) * factor, (attractorPoint.y - point.y) * factor,
              (attractorPoint.z - point.z) * factor]

        pointMod = [point.x, point.y, point.z]
        pointMod = [p1 + p2 for p1, p2 in zip(pointMod, DR)]
        point.move_to(pointMod[0], pointMod[1], pointMod[2])

    if PointAttractorList is None:
        pointAttractor = Point(5, 0.5, -2)
    else:
        pointAttractor = Point(PointAttractorList[0], PointAttractorList[1], PointAttractorList[2])

    for cell in lattice.cells:
        for beam in cell.beams_cell:
            movePointAttracted(beam.point1, pointAttractor, alpha, inverse)
            movePointAttracted(beam.point2, pointAttractor, alpha, inverse)
    lattice.define_lattice_dimensions()


@timing.category("design_transformation")
@timing.timeit
def curveLattice(lattice, center_x: float, center_y: float, center_z: float,
                 curvature_strength: float = 0.1) -> None:
    """
    Curve the lattice structure around a given center.

    Parameters:
    -----------
    center_x: float
        The x-coordinate of the center of the curvature.

    center_y: float
        The y-coordinate of the center of the curvature.

    center_z: float
        The z-coordinate of the center of the curvature.

    curvature_strength: float (default: 0.1)
        The strength of the curvature applied to the lattice.
        Positive values curve upwards, negative values curve downwards.
    """
    for cell in lattice.cells:
        for beam in cell.beams_cell:
            for node in [beam.point1, beam.point2]:
                x, y, z = node.x, node.y, node.z
                # Calculate the distance from the center of curvature
                dx = x - center_x
                dy = y - center_y
                dz = z - center_z
                new_z = z - curvature_strength * (dx ** 2 + dy ** 2 + dz ** 2)
                node.move_to(x, y, new_z)
    lattice.define_lattice_dimensions()

@timing.category("design_transformation")
@timing.timeit
def cylindrical_transform(lattice, radius: float) -> None:
    """
    Apply cylindrical transformation to the lattice structure.
    To create stent structures, 1 cell in the X direction is required and you can choose any number of cells in
    the Y and Z direction.

    Parameters:
    -----------
    radii: float
        radii of the cylinder.
    """
    max_y = lattice.size_y
    for cell in lattice.cells:
        for node in cell.points_cell:
            x, y, z = node.x, node.y, node.z
            # Convert Cartesian coordinates (x, y, z) to cylindrical coordinates (r, theta, z)
            theta = (y / max_y) * 2 * math.pi  # theta = (y / total height) * 2 * pi
            new_x = radius * math.cos(theta)
            new_y = radius * math.sin(theta)
            node.move_to(new_x, new_y, z)
    lattice.define_lattice_dimensions()
    lattice.delete_duplicated_beams()


@timing.category("design_transformation")
@timing.timeit
def moveToCylinderForm(lattice, radius: float) -> None:
    """
    Move the lattice to a cylindrical form.

    Parameters:
    -----------
    radii: float
        radii of the cylinder.
    """
    if radius <= lattice.x_max / 2:
        raise ValueError("The radii of the cylinder is too small: minimum value = ", lattice.x_max / 2)

    # Find moving distance
    def formula(x_coords):
        """
        Formula to calculate the new z-coordinate of the node.

        Parameters:
        -----------
        x: float
            x-coordinate of the node.
        """
        return radius - math.sqrt(radius ** 2 - (x_coords - lattice.x_max / 2) ** 2)

    for cell in lattice.cells:
        for node in cell.points_cell:
            x, y, z = node.x, node.y, node.z
            new_z = z - formula(x)
            node.move_to(x, y, new_z)
    lattice.define_lattice_dimensions()


@timing.category("design_transformation")
@timing.timeit
def fitToSurface(lattice, equation: callable, mode: str = "z", params: dict = None):
    """
    Adjust the lattice nodes to follow a surface defined by an equation.

    Parameters:
    -----------
    equation : callable
        Function representing the surface. For example, a lambda function or a normal function.
        Example: lambda x, y: x**2 + y**2 (for a paraboloid).

    mode : str
        Adjustment mode:
        - "z": Adjust nodes on a surface (z = f(x, y)).
        - "z_plan": Adjust nodes on a plan (z = f(x, y)) without changing the z-coordinate.

    params : dict
        Additional parameters for the equation or mode (e.g., radii, angle, etc.).
    """
    if params is None:
        params = {}
    nodeAlreadyChanged = []
    for cell in lattice.cells:
        for node in cell.points_cell:
            x, y, z = node.x, node.y, node.z
            if node not in nodeAlreadyChanged:
                nodeAlreadyChanged.append(node)
                # Adjust for a surface \( z = f(x, y) \)
                if mode == "z":
                    new_z = equation(x, y, **params)
                    new_z = z + new_z
                    node.move_to(x, y, new_z)
                elif mode == "z_plan":
                    new_z = equation(x, y, **params)
                    node.move_to(x, y, new_z)

                # Other modes can be added here (e.g. cylindrical, spherical)
                else:
                    raise ValueError(f"Mode '{mode}' non supporté.")

    # Update lattice limits after adjustment
    lattice.define_lattice_dimensions()
