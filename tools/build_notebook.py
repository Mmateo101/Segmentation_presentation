"""Build Segmentation_UNet.ipynb from tools/nb_source.py (cells separated by '# %%' markers).

Usage:  python tools/build_notebook.py
"""
from pathlib import Path
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / 'tools' / 'nb_source.py').read_text(encoding='utf-8')

cells = []
for block in re.split(r'^# %%', src, flags=re.M)[1:]:
    header, _, body = block.partition('\n')
    body = body.strip('\n')
    if header.strip() == '[markdown]':
        text = '\n'.join(re.sub(r'^# ?', '', line) for line in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(body))

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
    'language_info': {'name': 'python'},
    'colab': {'provenance': []},
    'accelerator': 'GPU',
})
out = ROOT / 'Segmentation_UNet.ipynb'
nbformat.write(nb, out)
print(f'{out.name}: {len(cells)} cells')
