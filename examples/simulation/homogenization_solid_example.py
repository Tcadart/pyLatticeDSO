"""
Homogenization example of a hybrid cell meshed as a 3D solid with a periodic mesh
"""
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils import create_homogenization_figure
from pyLatticeSim.utils_simulation import get_homogenized_properties_solid
from pyLatticeSim.export_simulation_results import exportSimulationResults


name_file = "simulation/hybrid_cell_simulation"

lattice_object = LatticeSim(name_file)

mat_S_orthotropic, homogenization_analysis = get_homogenized_properties_solid(lattice_object, mesh_refinement=0.5,
                                                                              element_degree=2)
homogenization_analysis.print_orthotropic_form()
print("Relative density:", homogenization_analysis.get_relative_density())

create_homogenization_figure(mat_S_orthotropic, save=True, name_file=name_file + "_solid", plot=False)

# Export simulations to Paraview
exportData = exportSimulationResults(homogenization_analysis, name_file + "_solid")
exportData.export_solid_data_homogenization()
