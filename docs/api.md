# API Documentation

## NotebookConverter

The `NotebookConverter` class provides functionality to convert markdown files to Jupyter notebooks, preserving code blocks and markdown content appropriately.

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.converter.md_converter import NotebookConverter

# Create converter instance
converter = NotebookConverter()

# Convert a single file
converter.md_to_notebook("path/to/input.md", "path/to/output.ipynb")

# Convert all markdown files in a directory
converter.convert_directory("source_dir", "target_dir")
```

### Configuration

The converter can be configured using a dictionary with the following options:

```python
config = {
    'default_kernel': 'python3',
    'code_cell_metadata': {'tags': []},
    'markdown_cell_metadata': {'tags': []},
}

converter = NotebookConverter(config=config)
```

### Methods

#### `__init__(config: Optional[Dict] = None)`

Initialize the converter with optional configuration.

**Parameters:**
- `config`: Optional dictionary containing configuration options
  - `default_kernel`: Kernel name for code cells (default: 'python3')
  - `code_cell_metadata`: Metadata for code cells
  - `markdown_cell_metadata`: Metadata for markdown cells

#### `md_to_notebook(md_path: str, nb_path: str) -> None`

Convert a markdown file to a Jupyter notebook.

**Parameters:**
- `md_path`: Path to the source markdown file
- `nb_path`: Path where the notebook should be saved

**Raises:**
- `FileNotFoundError`: If source file doesn't exist
- `PermissionError`: If unable to write to target location

#### `convert_directory(source_dir: str, target_dir: str, pattern: str = "**/*.md") -> None`

Convert all markdown files in a directory to Jupyter notebooks.

**Parameters:**
- `source_dir`: Source directory containing markdown files
- `target_dir`: Target directory for notebooks
- `pattern`: Glob pattern for finding markdown files (default: "**/*.md")

### Command Line Interface

The converter can be used from the command line:

```bash
python -m src.converter.md_converter source_dir target_dir [--pattern "**/*.md"]
```

### Examples

1. Basic conversion:
```python
converter = NotebookConverter()
converter.md_to_notebook("input.md", "output.ipynb")
```

2. With custom configuration:
```python
config = {
    'default_kernel': 'python3',
    'code_cell_metadata': {'tags': ['example']},
    'markdown_cell_metadata': {'tags': ['documentation']}
}
converter = NotebookConverter(config=config)
converter.convert_directory("docs", "notebooks")
```

3. Using pattern matching:
```python
converter.convert_directory(
    "source",
    "target",
    pattern="**/*tutorial*.md"
)
