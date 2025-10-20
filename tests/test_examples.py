"""
Test suite for validating example scripts.

This module tests that all example scripts in the examples/ directory
can be parsed correctly and have valid imports.
"""
import ast
import os
import sys
from pathlib import Path
import pytest


# Define the root directory of the project
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def get_example_files(subdirectory):
    """Get all Python example files in a subdirectory."""
    example_path = EXAMPLES_DIR / subdirectory
    if not example_path.exists():
        return []
    return sorted(example_path.glob("*.py"))


def test_design_examples_syntax():
    """Test that all design examples have valid Python syntax."""
    example_files = get_example_files("design")
    assert len(example_files) > 0, "No design examples found"
    
    for example_file in example_files:
        with open(example_file, 'r') as f:
            code = f.read()
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {example_file.name}: {e}")


def test_optimization_examples_syntax():
    """Test that all optimization examples have valid Python syntax."""
    example_files = get_example_files("optimization")
    assert len(example_files) > 0, "No optimization examples found"
    
    for example_file in example_files:
        with open(example_file, 'r') as f:
            code = f.read()
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {example_file.name}: {e}")


def test_simulation_examples_syntax():
    """Test that all simulation examples have valid Python syntax."""
    example_files = get_example_files("simulation")
    assert len(example_files) > 0, "No simulation examples found"
    
    for example_file in example_files:
        with open(example_file, 'r') as f:
            code = f.read()
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {example_file.name}: {e}")


def check_imports(example_file):
    """
    Check if all imports in an example file can be resolved.
    
    This function temporarily adds the example file's directory to sys.path
    and attempts to compile the file, which will fail if imports are broken.
    """
    # Set matplotlib to non-interactive backend to avoid display issues
    os.environ['MPLBACKEND'] = 'Agg'
    
    # Add the project root to sys.path if not already there
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    with open(example_file, 'r') as f:
        code = f.read()
    
    # Try to compile the code - this will check if all imports can be resolved
    try:
        compile(code, str(example_file), 'exec')
        return True, None
    except Exception as e:
        return False, str(e)


def test_design_examples_imports():
    """Test that all design examples have valid imports."""
    example_files = get_example_files("design")
    
    failed_imports = []
    for example_file in example_files:
        success, error = check_imports(example_file)
        if not success:
            failed_imports.append((example_file.name, error))
    
    if failed_imports:
        error_msg = "\n".join([f"{name}: {error}" for name, error in failed_imports])
        pytest.fail(f"Import errors in design examples:\n{error_msg}")


def test_optimization_examples_imports():
    """Test that all optimization examples have valid imports."""
    example_files = get_example_files("optimization")
    
    failed_imports = []
    for example_file in example_files:
        success, error = check_imports(example_file)
        if not success:
            failed_imports.append((example_file.name, error))
    
    if failed_imports:
        error_msg = "\n".join([f"{name}: {error}" for name, error in failed_imports])
        pytest.fail(f"Import errors in optimization examples:\n{error_msg}")


def test_simulation_examples_imports():
    """Test that all simulation examples have valid imports."""
    example_files = get_example_files("simulation")
    
    failed_imports = []
    for example_file in example_files:
        success, error = check_imports(example_file)
        if not success:
            failed_imports.append((example_file.name, error))
    
    if failed_imports:
        error_msg = "\n".join([f"{name}: {error}" for name, error in failed_imports])
        pytest.fail(f"Import errors in simulation examples:\n{error_msg}")


def test_all_examples_have_docstrings():
    """Test that all examples have module-level docstrings."""
    all_subdirs = ["design", "optimization", "simulation"]
    
    missing_docstrings = []
    for subdir in all_subdirs:
        example_files = get_example_files(subdir)
        for example_file in example_files:
            with open(example_file, 'r') as f:
                code = f.read()
            
            tree = ast.parse(code)
            has_docstring = (
                ast.get_docstring(tree) is not None
            )
            
            if not has_docstring:
                missing_docstrings.append(f"{subdir}/{example_file.name}")
    
    if missing_docstrings:
        pytest.fail(
            f"The following examples are missing docstrings:\n" +
            "\n".join(missing_docstrings)
        )
