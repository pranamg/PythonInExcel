import os
import nbformat
import re

DOCS_DIR = 'docs'
NOTEBOOKS_DIR = 'notebooks'


def md_to_notebook(md_path, nb_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    nb = nbformat.v4.new_notebook()
    cells = []
    in_code_block = False
    code_block_lang = ''
    code_lines = []
    md_lines = []

    def flush_md():
        if md_lines:
            cells.append(nbformat.v4.new_markdown_cell(''.join(md_lines).strip()))
            md_lines.clear()

    def flush_code():
        if code_lines:
            lang = code_block_lang.lower()
            # Only python/py code blocks become code cells; all others are markdown with code fences
            if lang == 'python' or lang == 'py':
                cells.append(nbformat.v4.new_code_cell(''.join(code_lines).strip()))
            else:
                code_fence_start = f'```{lang}\n' if lang else '```\n'
                code_fence_end = '```\n'
                code_content = code_fence_start + ''.join(code_lines) + code_fence_end
                cells.append(nbformat.v4.new_markdown_cell(code_content.strip()))
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
    os.makedirs(os.path.dirname(nb_path), exist_ok=True)
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)


def convert_all_markdown():
    for root, _, files in os.walk(DOCS_DIR):
        for file in files:
            if file.endswith('.md'):
                md_path = os.path.join(root, file)
                rel_path = os.path.relpath(md_path, DOCS_DIR)
                nb_path = os.path.join(NOTEBOOKS_DIR, os.path.splitext(rel_path)[0] + '.ipynb')
                md_to_notebook(md_path, nb_path)
                print(f'Converted: {md_path} -> {nb_path}')


if __name__ == '__main__':
    convert_all_markdown()
    print('All markdown files have been converted to notebooks.')
