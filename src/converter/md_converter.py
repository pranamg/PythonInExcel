"""
Markdown to Jupyter Notebook Converter.

This module provides functionality to convert markdown files to Jupyter notebooks,
preserving code blocks and markdown content appropriately.
"""

import os
import re
from typing import List, Dict, Optional, Union, Any
import nbformat
from tqdm import tqdm
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConversionStats:
    """Statistics about the conversion process."""
    total_files: int
    successful: int
    failed: int
    start_time: datetime
    end_time: Optional[datetime] = None
    errors: List[str] = None

    def __post_init__(self):
        self.errors = self.errors or []

    def add_error(self, error: str) -> None:
        """Add an error message to the stats."""
        self.errors.append(error)

    def finish(self) -> None:
        """Mark the conversion as finished."""
        self.end_time = datetime.now()

    @property
    def duration(self) -> float:
        """Get the duration of the conversion in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

class NotebookConverter:
    """Converts markdown files to Jupyter notebooks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the converter with optional configuration.

        Args:
            config: Dictionary containing configuration options
                   Default: None, uses default settings
        """
        self.config = self._load_config(config)
        self._setup_logging()
        self.stats = None

    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load and validate configuration."""
        default_config = {
            'default_kernel': 'python3',
            'code_cell_metadata': {'tags': []},
            'markdown_cell_metadata': {'tags': []},
            'validate_notebooks': True,
            'backup_existing': True,
            'jupyter_version': 4,
            'languages': {
                'python': {'cell_type': 'code'},
                'py': {'cell_type': 'code'},
                'raw': {'cell_type': 'raw'},
            }
        }

        if config:
            default_config.update(config)

        return default_config

    def _setup_logging(self) -> None:
        """Configure logging for the converter."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def validate_notebook(self, nb: nbformat.NotebookNode) -> bool:
        """
        Validate a notebook structure.

        Args:
            nb: The notebook to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            nbformat.validate(nb)
            return True
        except Exception as e:
            self.logger.warning(f"Notebook validation failed: {str(e)}")
            return False

    def backup_notebook(self, path: str) -> None:
        """
        Create a backup of an existing notebook.

        Args:
            path: Path to the notebook to backup
        """
        if os.path.exists(path):
            backup_path = f"{path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")

    def md_to_notebook(self, md_path: str, nb_path: str) -> bool:
        """
        Convert a markdown file to a Jupyter notebook.

        Args:
            md_path: Path to the source markdown file
            nb_path: Path where the notebook should be saved

        Returns:
            bool: True if conversion was successful, False otherwise

        Raises:
            FileNotFoundError: If source file doesn't exist
            PermissionError: If unable to write to target location
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            nb = nbformat.v4.new_notebook()
            cells = []
            in_code_block = False
            code_block_lang = ''
            code_lines: List[str] = []
            md_lines: List[str] = []

            def flush_md() -> None:
                if md_lines:
                    cells.append(
                        nbformat.v4.new_markdown_cell(
                            ''.join(md_lines).strip(),
                            metadata=self.config['markdown_cell_metadata']
                        )
                    )
                    md_lines.clear()

            def flush_code() -> None:
                if code_lines:
                    lang = code_block_lang.lower()
                    lang_config = self.config['languages'].get(lang, {'cell_type': 'markdown'})
                    
                    if lang_config['cell_type'] == 'code':
                        cells.append(
                            nbformat.v4.new_code_cell(
                                ''.join(code_lines).strip(),
                                metadata=self.config['code_cell_metadata']
                            )
                        )
                    elif lang_config['cell_type'] == 'raw':
                        cells.append(
                            nbformat.v4.new_raw_cell(
                                ''.join(code_lines).strip()
                            )
                        )
                    else:
                        # Non-Python code blocks become markdown with code fences
                        code_block = f'```{lang}\n{"".join(code_lines)}```'
                        cells.append(
                            nbformat.v4.new_markdown_cell(
                                code_block.strip(),
                                metadata=self.config['markdown_cell_metadata']
                            )
                        )
                    code_lines.clear()

            code_block_pattern = re.compile(r'^```(\w*)\s*$')
            for line in lines:
                code_block_match = code_block_pattern.match(line)
                if code_block_match:
                    if not in_code_block:
                        flush_md()
                        in_code_block = True
                        code_block_lang = code_block_match.group(1)
                    else:
                        flush_code()
                        in_code_block = False
                        code_block_lang = ''
                else:
                    if in_code_block:
                        code_lines.append(line)
                    else:
                        md_lines.append(line)

            # Flush any remaining content
            if in_code_block:
                flush_code()
            if md_lines:
                flush_md()

            nb.cells = cells
            
            # Validate notebook if configured
            if self.config['validate_notebooks'] and not self.validate_notebook(nb):
                raise ValueError("Generated notebook failed validation")

            # Backup existing notebook if configured
            if self.config['backup_existing']:
                self.backup_notebook(nb_path)

            # Create directory and save notebook
            os.makedirs(os.path.dirname(nb_path), exist_ok=True)
            with open(nb_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            
            self.logger.info(f"Successfully converted {md_path} to {nb_path}")
            return True

        except FileNotFoundError:
            self.logger.error(f"Source file not found: {md_path}")
            raise
        except PermissionError:
            self.logger.error(f"Permission denied when writing to: {nb_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error converting {md_path}: {str(e)}")
            raise

    def convert_directory(self, source_dir: str, target_dir: str, pattern: str = "**/*.md") -> ConversionStats:
        """
        Convert all markdown files in a directory to Jupyter notebooks.

        Args:
            source_dir: Source directory containing markdown files
            target_dir: Target directory for notebooks
            pattern: Glob pattern for finding markdown files (default: "**/*.md")

        Returns:
            ConversionStats: Statistics about the conversion process
        """
        source_path = Path(source_dir)
        target_path = Path(target_dir)

        # Find all markdown files in source directory
        md_files = list(source_path.glob(pattern))
        
        self.stats = ConversionStats(
            total_files=len(md_files),
            successful=0,
            failed=0,
            start_time=datetime.now()
        )
        
        self.logger.info(f"Found {len(md_files)} markdown files to convert")
        
        for md_file in tqdm(md_files, desc="Converting files"):
            # Preserve directory structure
            relative_path = md_file.relative_to(source_path)
            nb_path = target_path / relative_path.with_suffix('.ipynb')
            
            try:
                if self.md_to_notebook(str(md_file), str(nb_path)):
                    self.stats.successful += 1
            except Exception as e:
                self.stats.failed += 1
                self.stats.add_error(f"Failed to convert {md_file}: {str(e)}")
                continue

        self.stats.finish()
        self.logger.info(
            f"Conversion complete. "
            f"Success: {self.stats.successful}/{self.stats.total_files}, "
            f"Failed: {self.stats.failed}, "
            f"Duration: {self.stats.duration:.2f}s"
        )

        return self.stats

def main() -> None:
    """CLI entry point for the converter."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Convert markdown files to Jupyter notebooks")
    parser.add_argument("source_dir", help="Source directory containing markdown files")
    parser.add_argument("target_dir", help="Target directory for notebooks")
    parser.add_argument("--pattern", default="**/*.md", help="Glob pattern for finding markdown files")
    parser.add_argument("--config", help="Path to JSON configuration file")
    parser.add_argument("--no-validate", action="store_true", help="Skip notebook validation")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backups of existing notebooks")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override configuration with command line arguments
    if config is None:
        config = {}
    if args.no_validate:
        config['validate_notebooks'] = False
    if args.no_backup:
        config['backup_existing'] = False
    
    converter = NotebookConverter(config=config)
    stats = converter.convert_directory(args.source_dir, args.target_dir, args.pattern)
    
    if stats.failed > 0:
        print("\nErrors encountered:")
        for error in stats.errors:
            print(f"- {error}")

if __name__ == "__main__":
    main()
