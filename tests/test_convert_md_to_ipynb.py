import os
import pytest
import nbformat
from convert_md_to_ipynb import md_to_notebook

@pytest.fixture
def temp_md_file(tmp_path):
    md_content = """# Test Notebook
    
This is a test markdown file with code blocks.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3]})
```

Some more markdown text.

```sql
SELECT * FROM table;
```

Final markdown text."""
    
    file_path = tmp_path / "test.md"
    with open(file_path, "w") as f:
        f.write(md_content)
    return file_path

def test_md_to_notebook(temp_md_file, tmp_path):
    # Create output path
    output_path = tmp_path / "test.ipynb"
    
    # Convert markdown to notebook
    md_to_notebook(temp_md_file, output_path)
    
    # Verify the notebook was created
    assert output_path.exists()
    
    # Load the notebook
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check number of cells
    assert len(nb.cells) == 5
    
    # Check cell types
    assert nb.cells[0].cell_type == "markdown"  # # Test Notebook
    assert nb.cells[1].cell_type == "markdown"  # This is a test...
    assert nb.cells[2].cell_type == "code"      # Python code block
    assert nb.cells[3].cell_type == "markdown"  # SQL code block (non-python)
    assert nb.cells[4].cell_type == "markdown"  # Final markdown text

def test_empty_markdown(tmp_path):
    # Test with empty markdown file
    md_file = tmp_path / "empty.md"
    md_file.write_text("")
    
    output_path = tmp_path / "empty.ipynb"
    md_to_notebook(md_file, output_path)
    
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    assert len(nb.cells) == 0

def test_markdown_only(tmp_path):
    # Test with markdown-only content
    md_file = tmp_path / "markdown_only.md"
    md_file.write_text("# Title\n\nJust markdown\n\nNo code blocks")
    
    output_path = tmp_path / "markdown_only.ipynb"
    md_to_notebook(md_file, output_path)
    
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == "markdown"

def test_code_only(tmp_path):
    # Test with only code blocks
    md_file = tmp_path / "code_only.md"
    md_file.write_text("```python\nprint('hello')\n```\n\n```python\nx = 1\n```")
    
    output_path = tmp_path / "code_only.ipynb"
    md_to_notebook(md_file, output_path)
    
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    assert len(nb.cells) == 2
    assert all(cell.cell_type == "code" for cell in nb.cells)
