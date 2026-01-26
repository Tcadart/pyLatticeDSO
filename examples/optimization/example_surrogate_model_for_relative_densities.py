"""
Example file for generating a surrogate model to predict relative denities based on cell properties.
"""
from pathlib import Path

from pyLatticeDesign.lattice import Lattice
from pyLatticeOpti.surrogate_model_relative_densities import (compute_relative_densities_dataset, plot_3D_iso_surface,
                                                              plot_3D_scatter, csv_to_dataset,
                                                              evaluate_kriging_from_pickle,
                                                              evaluate_saved_kriging)

name_cell = "hybrid_cell_simulation"
name_file = "simulation/" + name_cell

hybrid_cell = Lattice(name_file, verbose=-1)

compute_relative_densities_dataset(hybrid_cell)

# plot_3D_iso_surface(name_dataset="Test")
# plot_3D_scatter(name_dataset="Test")

name_cell = "BCC_Hybrid1"

evaluate_kriging_from_pickle(name_dataset="RelativeDensities_" + name_cell, dataset_dir=None)

evaluate_saved_kriging(name_dataset="RelativeDensities_" + name_cell, dataset_dir=None, model_name= name_cell,
                       save_parity_path=Path("data"))

