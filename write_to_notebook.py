# import IPython.nbformat.current as nbf
# nb = nbf.read(open('movie_rating_prediction.py', 'r'), 'py')
# nbf.write(nb, open('movie_rating_prediction.ipynb', 'w'), 'ipynb')


import nbformat
from nbformat.v4 import new_code_cell,new_notebook

import codecs

sourceFile = "movie_rating_prediction.py"     # <<<< change
destFile = "movie_rating_prediction.ipynb"    # <<<< change


def parsePy(fn):
    """ Generator that parses a .py file exported from a IPython notebook and
extracts code cells (whatever is between occurrences of "In[*]:").
Returns a string containing one or more lines
"""
    with open(fn,"r") as f:
        lines = []
        for l in f:
            l1 = l.strip()
            if l1.startswith('# In[') and l1.endswith(']:') and lines:
                yield "".join(lines)
                lines = []
                continue
            lines.append(l)
        if lines:
            yield "".join(lines)

# Create the code cells by parsing the file in input
cells = []
for c in parsePy(sourceFile):
    cells.append(new_code_cell(source=c))

# This creates a V4 Notebook with the code cells extracted above
nb0 = new_notebook(cells=cells,
                   metadata={'language': 'python',})

with codecs.open(destFile, encoding='utf-8', mode='w') as f:
    nbformat.write(nb0, f, 4)