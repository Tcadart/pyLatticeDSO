"""Test module for Materials and Timing classes."""
import pytest
import os
import tempfile
import json
import time
from src.pyLatticeDesign.materials import MatProperties
from src.pyLatticeDesign.timing import Timing


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


class TestTiming:
    """Test cases for Timing class."""
    
    def test_timing_initialization(self):
        """Test timing class initialization."""
        timing = Timing()
        
        assert timing.timings is not None
        assert timing.call_stack == []
        assert timing.call_graph is not None
        assert timing.call_counts is not None
        assert hasattr(timing, 'start_time85')

    def test_timing_decorator(self):
        """Test timing decorator functionality."""
        timing = Timing()
        
        @timing.timeit
        def test_function():
            time.sleep(0.01)  # Sleep for 10ms
            return "test_result"
        
        result = test_function()
        
        assert result == "test_result"
        assert "test_function" in timing.timings
        assert len(timing.timings["test_function"]) == 1
        assert timing.timings["test_function"][0] >= 0.01  # Should be at least 10ms
        assert timing.call_counts["test_function"] == 1

    def test_timing_multiple_calls(self):
        """Test timing with multiple function calls."""
        timing = Timing()
        
        @timing.timeit
        def fast_function():
            return "fast"
        
        # Call function multiple times
        for _ in range(3):
            fast_function()
        
        assert len(timing.timings["fast_function"]) == 3
        assert timing.call_counts["fast_function"] == 3
        assert all(t >= 0 for t in timing.timings["fast_function"])

    def test_timing_nested_calls(self):
        """Test timing with nested function calls."""
        timing = Timing()
        
        @timing.timeit
        def outer_function():
            inner_function()
            return "outer"
        
        @timing.timeit
        def inner_function():
            time.sleep(0.005)  # 5ms
            return "inner"
        
        result = outer_function()
        
        assert result == "outer"
        assert "outer_function" in timing.timings
        assert "inner_function" in timing.timings
        assert "outer_function" in timing.call_graph
        assert "inner_function" in timing.call_graph["outer_function"]
        assert timing.call_graph["outer_function"]["inner_function"] >= 0.005

    def test_timing_summary(self, capsys):
        """Test timing summary output."""
        timing = Timing()
        
        @timing.timeit
        def test_function():
            time.sleep(0.001)  # 1ms
        
        test_function()
        timing.summary()
        
        captured = capsys.readouterr()
        assert "Function" in captured.out
        assert "test_function" in captured.out
        assert "Total" in captured.out

    def test_timing_call_hierarchy(self):
        """Test call hierarchy tracking."""
        timing = Timing()
        
        @timing.timeit
        def level1():
            level2()
            return "level1"
        
        @timing.timeit
        def level2():
            level3()
            return "level2"
        
        @timing.timeit
        def level3():
            return "level3"
        
        level1()
        
        # Check that call hierarchy is properly tracked
        assert "level1" in timing.call_graph
        assert "level2" in timing.call_graph["level1"]
        assert "level2" in timing.call_graph
        assert "level3" in timing.call_graph["level2"]

    def test_timing_performance_measurement(self):
        """Test that timing measurements are reasonably accurate."""
        timing = Timing()
        
        @timing.timeit
        def timed_sleep(duration):
            time.sleep(duration)
        
        sleep_duration = 0.01  # 10ms
        timed_sleep(sleep_duration)
        
        measured_time = timing.timings["timed_sleep"][0]
        
        # Measured time should be approximately the sleep duration
        # Allow for some tolerance due to system timing variations
        assert measured_time >= sleep_duration
        assert measured_time <= sleep_duration * 2  # Should not be more than 2x

    def test_timing_thread_safety(self):
        """Test timing thread safety with threading.local."""
        timing = Timing()
        
        @timing.timeit
        def thread_function():
            return "thread_result"
        
        result = thread_function()
        
        # Basic test that threading.local is initialized
        assert hasattr(timing, 'local')
        assert result == "thread_result"
        assert "thread_function" in timing.timings