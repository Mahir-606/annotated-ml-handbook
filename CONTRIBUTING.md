# Contributing

Thank you for helping improve the companion code for *The Annotated ML Handbook*.

## Reporting a problem

- **A notebook or script fails, or gives a clearly different result:** open a
  [code problem](../../issues/new?template=code-problem.yml) issue. Include the chapter, the
  cell, the full error message, and the output of
  `python -c "import sys, numpy, pandas, sklearn, tensorflow; print(sys.version, numpy.__version__, pandas.__version__, sklearn.__version__, tensorflow.__version__)"`.
- **A mistake in the book:** open an [erratum](../../issues/new?template=erratum.yml) issue
  with the page or section number.

Small numeric differences (the fourth decimal of a score, a slightly different training curve)
are normal across machines and library versions, especially for neural networks.

## Fixing the text

The chapters are Markdown files in `book-source/chapters/`. To fix a typo, edit the file on GitHub and open a
pull request. The notebooks are generated from these files, so please change the chapter source rather
than the notebook.

## Pull requests

1. Fork the repository and create a branch.
2. Keep each example identical in behaviour to the book: fixes that make the code run on a newer
   library version are welcome; rewrites of the teaching examples are not.
3. Run `python tools/run_all.py` (or `python tools/run_all.py 21` for one chapter) and make
   sure it passes.
4. Do not commit the `outputs/` folders.
5. Open the pull request and describe what changed and why.

The deliberate error examples (cells tagged `raises-exception`) are meant to fail. Please do not
"fix" them.

By contributing code you agree that it is released under the MIT License of this repository.
