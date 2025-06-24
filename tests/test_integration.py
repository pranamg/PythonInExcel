"""Integration tests for the notebook converter."""

import os
import pytest
from pathlib import Path
import nbformat
import json
from src.converter.md_converter import NotebookConverter

@pytest.fixture
def test_config():
    return {
        'default_kernel': 'python3',
        'code_cell_metadata': {'tags': ['example']},
        'markdown_cell_metadata': {'tags': ['doc']}
    }

@pytest.fixture
def converter(test_config):
    return NotebookConverter(config=test_config)

@pytest.fixture
def complex_md_file(tmp_path):
    content = """# Test Notebook

This is a test with multiple code blocks and languages.

```python
import pandas as pd
# Python code block 1
df = pd.DataFrame({'a': [1, 2, 3]})
```

Some text between code blocks.

```sql
SELECT * FROM table;
```

```python
# Python code block 2
print("Hello, World!")
```

Final markdown section with **bold** and *italic* text."""

    file_path = tmp_path / "complex_test.md"
    file_path.write_text(content)
    return file_path

def test_config_handling(test_config):
    """Test that configuration is properly handled."""
    converter = NotebookConverter(config=test_config)
    assert converter.config['default_kernel'] == 'python3'
    assert converter.config['code_cell_metadata']['tags'] == ['example']

def test_complex_conversion(converter, complex_md_file, tmp_path):
    """Test conversion of a complex markdown file with multiple code blocks."""
    output_path = tmp_path / "complex_test.ipynb"
    converter.md_to_notebook(str(complex_md_file), str(output_path))
    
    # Verify the notebook was created
    assert output_path.exists()
    
    # Load and verify notebook content
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Should have 5 cells: title, text, python code, sql (as markdown), python code, final text
    assert len(nb.cells) == 6
    
    # Check cell types
    assert nb.cells[0].cell_type == "markdown"  # Title
    assert nb.cells[1].cell_type == "markdown"  # Text
    assert nb.cells[2].cell_type == "code"      # Python code
    assert nb.cells[3].cell_type == "markdown"  # SQL code (as markdown)
    assert nb.cells[4].cell_type == "code"      # Python code
    assert nb.cells[5].cell_type == "markdown"  # Final text
    
    # Verify metadata
    assert nb.cells[0].metadata['tags'] == ['doc']  # markdown cell
    assert nb.cells[2].metadata['tags'] == ['example']  # code cell

def test_directory_conversion(converter, tmp_path):
    """Test converting multiple files in a directory structure."""
    # Create test directory structure
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    target_dir = tmp_path / "target"
    
    # Create test files
    (source_dir / "test1.md").write_text("# Test 1\n```python\nprint(1)\n```")
    (source_dir / "test2.md").write_text("# Test 2\n```python\nprint(2)\n```")
    
    # Create subdirectory with files
    subdir = source_dir / "subdir"
    subdir.mkdir()
    (subdir / "test3.md").write_text("# Test 3\n```python\nprint(3)\n```")
    
    # Convert directory
    converter.convert_directory(str(source_dir), str(target_dir))
    
    # Verify output structure
    assert (target_dir / "test1.ipynb").exists()
    assert (target_dir / "test2.ipynb").exists()
    assert (target_dir / "subdir" / "test3.ipynb").exists()

def test_error_handling(converter, tmp_path):
    """Test error handling for various scenarios."""
    # Test non-existent source file
    with pytest.raises(FileNotFoundError):
        converter.md_to_notebook(str(tmp_path / "nonexistent.md"), str(tmp_path / "out.ipynb"))
    
    # Test invalid target directory (no permissions)
    if os.name != 'nt':  # Skip on Windows
        invalid_dir = tmp_path / "noperm"
        invalid_dir.mkdir()
        invalid_dir.chmod(0o000)
        
        with pytest.raises(PermissionError):
            converter.md_to_notebook(
                str(tmp_path / "test.md"),
                str(invalid_dir / "out.ipynb")
            )
        
        invalid_dir.chmod(0o755)  # Restore permissions for cleanup
