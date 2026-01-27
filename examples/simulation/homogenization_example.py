"""
Simple homogenization example of a hybrid cell
"""
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils import create_homogenization_figure
from pyLatticeSim.utils_simulation import get_homogenized_properties
from pyLatticeSim.export_simulation_results import exportSimulationResults


name_file = "simulation/hybrid_cell_simulation"

lattice_object = LatticeSim(name_file)

mat_S_orthotropic, homogenization_analysis = get_homogenized_properties(lattice_object)

create_homogenization_figure(mat_S_orthotropic, save=True, name_file=name_file, plot= False)

# Export simulations to Paraview
exportData = exportSimulationResults(homogenization_analysis, name_file)
exportData.export_data_homogenization()