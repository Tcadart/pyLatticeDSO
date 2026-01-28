"""Test module for Materials."""
import pytest
from pyLatticeDesign.materials import MatProperties


class TestMatProperties:
    """Test cases for MatProperties class."""
    
    def test_load_existing_material(self):
        """Test loading an existing material file."""
        # Test with Ti-6Al-4V which should exist
        mat = MatProperties("Ti-6Al-4V")
        
        assert mat.name_material is not None
        assert mat.density is not None
        assert mat.young_modulus is not None
        assert mat.poisson_ratio is not None
        assert isinstance(mat.density, (int, float))
        assert isinstance(mat.young_modulus, (int, float))
        assert isinstance(mat.poisson_ratio, (int, float))
        assert 0 <= mat.poisson_ratio <= 0.5  # Physically reasonable Poisson ratio

    def test_load_nonexistent_material(self):
        """Test loading a non-existent material file."""
        with pytest.raises(FileNotFoundError):
            MatProperties("NonExistentMaterial")

    def test_material_file_path(self):
        """Test that material file path is constructed correctly."""
        mat = MatProperties("Ti-6Al-4V")
        assert "Ti-6Al-4V.json" in mat.file_path
        assert "materials" in mat.file_path

    def test_multiple_materials(self):
        """Test loading multiple different materials."""
        materials = ["Ti-6Al-4V", "VeroClear", "TPU"]
        loaded_materials = []
        
        for mat_name in materials:
            try:
                mat = MatProperties(mat_name)
                loaded_materials.append(mat)
                assert mat.name_material is not None
                assert mat.density is not None
                assert mat.young_modulus is not None
                assert mat.poisson_ratio is not None
            except FileNotFoundError:
                # Skip if material file doesn't exist
                continue
        
        # Should have loaded at least one material
        assert len(loaded_materials) > 0

    def test_material_properties_types(self):
        """Test that material properties have correct types."""
        mat = MatProperties("Ti-6Al-4V")
        
        if mat.density is not None:
            assert isinstance(mat.density, (int, float))
            assert mat.density > 0
        
        if mat.young_modulus is not None:
            assert isinstance(mat.young_modulus, (int, float))
            assert mat.young_modulus > 0
        
        if mat.poisson_ratio is not None:
            assert isinstance(mat.poisson_ratio, (int, float))
            assert 0 <= mat.poisson_ratio <= 0.5

