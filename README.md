<div align="center">

<img src="assets/cover-front.jpg" alt="The Annotated ML Handbook, by Mahir Absar Khan" width="300">

# The Annotated ML Handbook

**Python → NumPy → pandas → Matplotlib · Seaborn · Plotly → scikit-learn → TensorFlow & Keras → ML for research → Debugging**
*From your first line of code to machine learning you can publish. Every line explained. Every output real.*

### 📖 The complete book, free to read

**[🌐 Read the web edition](https://mahir-606.github.io/annotated-ml-handbook/)** &nbsp;·&nbsp; **[📄 Download the PDF](https://mahir-606.github.io/annotated-ml-handbook/The-Annotated-ML-Handbook.pdf)** &nbsp;·&nbsp; **[📚 Read it here on GitHub](book/README.md)**

[![Run all notebooks](https://github.com/Mahir-606/annotated-ml-handbook/actions/workflows/run-notebooks.yml/badge.svg)](https://github.com/Mahir-606/annotated-ml-handbook/actions/workflows/run-notebooks.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-1f4fff?style=flat-square)
![Chapters](https://img.shields.io/badge/chapters-34-7b3ff2?style=flat-square)
![Examples](https://img.shields.io/badge/code_examples-441-e0245e?style=flat-square)
![Figures](https://img.shields.io/badge/figures-121-0091b3?style=flat-square)
![Definitions](https://img.shields.io/badge/terms_defined-273-0f9b62?style=flat-square)
[![Code license: MIT](https://img.shields.io/badge/code_license-MIT-ff6a00?style=flat-square)](LICENSE)

</div>

This repository is the **whole book**: every chapter's text, definitions, tips, pitfalls and research advice, every code example with a **line-by-line explanation**, the **real output** each example produced, every **figure**, all the **exercises** and their **worked solutions**, plus the book's own source files.

Learn here without downloading anything: open a chapter below and read it on GitHub, or press **Open in Colab** to run it in your browser while you read.

## Who it is for

- **Beginners** who have never programmed: Part I starts from zero.
- **Students and researchers** who want to use machine learning in a thesis or paper, and do it properly: honest validation, no data leakage, confidence intervals, reproducible code.
- **Practitioners** who want a dense reference for NumPy, pandas, visualization, scikit-learn and TensorFlow, and an **encyclopedia of real error messages** with their fixes.

## Three ways to read

| | Best for | How |
|---|---|---|
| 🌐 **[Web edition](https://mahir-606.github.io/annotated-ml-handbook/)** | Reading straight through, on any device | The designed book with search, interactive Plotly charts and the colour-coded route |
| 📄 **[PDF](https://mahir-606.github.io/annotated-ml-handbook/The-Annotated-ML-Handbook.pdf)** | Offline, printing, tablets | 472 pages with clickable contents |
| 📓 **[Chapters on GitHub](book/README.md)** | Learning by doing | Full-text notebooks: read on GitHub, or open in Colab or Jupyter and run every example |

## Inside each chapter

Every chapter notebook contains the complete text of the chapter, in order:

- **📖 Definitions** boxes for each new term (273 terms in total, all listed in the [index of definitions](book/index-of-definitions.md))
- **Code you can run**, each cell followed by a **"What each line does"** table
- The **real output** and **figures**, produced by running the code
- **💡 Tips, ⚠️ Pitfalls, 🐞 Bug hunts, 🔬 Research practice** and **🔑 Key ideas**
- **✍️ Try it yourself** exercises, linked to their [worked solutions](solutions)
- **48 deliberate error examples** (cells tagged `raises-exception`) showing the exact message you will meet, and why

## Contents

**Front matter:** [Preface](book/preface.md) · [Acknowledgements](book/acknowledgements.md) · [How to use this book](book/how-to-use-this-book.md)

### ![](https://img.shields.io/badge/Part_I-Python-1f4fff?style=flat-square)

| # | Chapter | Read & run | Code only | Colab |
|---:|---|---|---|:---:|
| 1 | **Values, variables and types** | [chapter](notebooks/part-1-python/01-python-basics.ipynb) | [.py](scripts/part-1-python/01-python-basics.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-1-python/01-python-basics.ipynb) |
| 2 | **Data structures: lists, tuples, dictionaries, sets** | [chapter](notebooks/part-1-python/02-python-collections.ipynb) | [.py](scripts/part-1-python/02-python-collections.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-1-python/02-python-collections.ipynb) |
| 3 | **Control flow: decisions and loops** | [chapter](notebooks/part-1-python/03-python-control.ipynb) | [.py](scripts/part-1-python/03-python-control.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-1-python/03-python-control.ipynb) |
| 4 | **Functions: reusable, testable logic** | [chapter](notebooks/part-1-python/04-python-functions.ipynb) | [.py](scripts/part-1-python/04-python-functions.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-1-python/04-python-functions.ipynb) |
| 5 | **Classes and objects** | [chapter](notebooks/part-1-python/05-python-oop.ipynb) | [.py](scripts/part-1-python/05-python-oop.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-1-python/05-python-oop.ipynb) |
| 6 | **Errors, files and modules** | [chapter](notebooks/part-1-python/06-python-errors-files.ipynb) | [.py](scripts/part-1-python/06-python-errors-files.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-1-python/06-python-errors-files.ipynb) |
| 7 | **The Pythonic toolkit** | [chapter](notebooks/part-1-python/07-python-toolkit.ipynb) | [.py](scripts/part-1-python/07-python-toolkit.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-1-python/07-python-toolkit.ipynb) |

### ![](https://img.shields.io/badge/Part_II-NumPy-7b3ff2?style=flat-square)

| # | Chapter | Read & run | Code only | Colab |
|---:|---|---|---|:---:|
| 8 | **NumPy arrays: shape, dtype and indexing** | [chapter](notebooks/part-2-numpy/08-numpy-arrays.ipynb) | [.py](scripts/part-2-numpy/08-numpy-arrays.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-2-numpy/08-numpy-arrays.ipynb) |
| 9 | **Computing with NumPy: vectorization, broadcasting, statistics, linear algebra** | [chapter](notebooks/part-2-numpy/09-numpy-compute.ipynb) | [.py](scripts/part-2-numpy/09-numpy-compute.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-2-numpy/09-numpy-compute.ipynb) |
| 10 | **Machine learning from scratch with NumPy** | [chapter](notebooks/part-2-numpy/10-numpy-ml-scratch.ipynb) | [.py](scripts/part-2-numpy/10-numpy-ml-scratch.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-2-numpy/10-numpy-ml-scratch.ipynb) |

### ![](https://img.shields.io/badge/Part_III-Pandas-e0245e?style=flat-square)

| # | Chapter | Read & run | Code only | Colab |
|---:|---|---|---|:---:|
| 11 | **pandas foundations: Series, DataFrames and selection** | [chapter](notebooks/part-3-pandas/11-pandas-basics.ipynb) | [.py](scripts/part-3-pandas/11-pandas-basics.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-3-pandas/11-pandas-basics.ipynb) |
| 12 | **Cleaning data with pandas** | [chapter](notebooks/part-3-pandas/12-pandas-cleaning.ipynb) | [.py](scripts/part-3-pandas/12-pandas-cleaning.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-3-pandas/12-pandas-cleaning.ipynb) |
| 13 | **Grouping, reshaping, joining and time series** | [chapter](notebooks/part-3-pandas/13-pandas-transform.ipynb) | [.py](scripts/part-3-pandas/13-pandas-transform.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-3-pandas/13-pandas-transform.ipynb) |

### ![](https://img.shields.io/badge/Part_IV-Visualization-0091b3?style=flat-square)

| # | Chapter | Read & run | Code only | Colab |
|---:|---|---|---|:---:|
| 14 | **Matplotlib and Seaborn: from first plot to publication figure** | [chapter](notebooks/part-4-visualization/14-viz-matplotlib.ipynb) | [.py](scripts/part-4-visualization/14-viz-matplotlib.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-4-visualization/14-viz-matplotlib.ipynb) |
| 15 | **The Matplotlib chart gallery** | [chapter](notebooks/part-4-visualization/15-viz-mpl-gallery.ipynb) | [.py](scripts/part-4-visualization/15-viz-mpl-gallery.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-4-visualization/15-viz-mpl-gallery.ipynb) |
| 16 | **The Seaborn chart gallery** | [chapter](notebooks/part-4-visualization/16-viz-seaborn-gallery.ipynb) | [.py](scripts/part-4-visualization/16-viz-seaborn-gallery.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-4-visualization/16-viz-seaborn-gallery.ipynb) |
| 17 | **Interactive charts with Plotly** | [chapter](notebooks/part-4-visualization/17-viz-plotly.ipynb) | [.py](scripts/part-4-visualization/17-viz-plotly.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-4-visualization/17-viz-plotly.ipynb) |
| 18 | **Figures for papers, theses and assignments** | [chapter](notebooks/part-4-visualization/18-viz-export.ipynb) | [.py](scripts/part-4-visualization/18-viz-export.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-4-visualization/18-viz-export.ipynb) |

### ![](https://img.shields.io/badge/Part_V-scikit--learn-0f9b62?style=flat-square)

| # | Chapter | Read & run | Code only | Colab |
|---:|---|---|---|:---:|
| 19 | **Machine-learning concepts and the scikit-learn API** | [chapter](notebooks/part-5-scikit-learn/19-sklearn-concepts.ipynb) | [.py](scripts/part-5-scikit-learn/19-sklearn-concepts.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-5-scikit-learn/19-sklearn-concepts.ipynb) |
| 20 | **Preprocessing, feature engineering and pipelines** | [chapter](notebooks/part-5-scikit-learn/20-sklearn-preprocessing.ipynb) | [.py](scripts/part-5-scikit-learn/20-sklearn-preprocessing.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-5-scikit-learn/20-sklearn-preprocessing.ipynb) |
| 21 | **Regression: predicting numbers** | [chapter](notebooks/part-5-scikit-learn/21-sklearn-regression.ipynb) | [.py](scripts/part-5-scikit-learn/21-sklearn-regression.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-5-scikit-learn/21-sklearn-regression.ipynb) |
| 22 | **Classification: predicting categories** | [chapter](notebooks/part-5-scikit-learn/22-sklearn-classification.ipynb) | [.py](scripts/part-5-scikit-learn/22-sklearn-classification.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-5-scikit-learn/22-sklearn-classification.ipynb) |
| 23 | **Model selection: cross-validation and hyperparameter tuning** | [chapter](notebooks/part-5-scikit-learn/23-sklearn-model-selection.ipynb) | [.py](scripts/part-5-scikit-learn/23-sklearn-model-selection.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-5-scikit-learn/23-sklearn-model-selection.ipynb) |
| 24 | **Unsupervised learning: clustering and dimensionality reduction** | [chapter](notebooks/part-5-scikit-learn/24-sklearn-unsupervised.ipynb) | [.py](scripts/part-5-scikit-learn/24-sklearn-unsupervised.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-5-scikit-learn/24-sklearn-unsupervised.ipynb) |
| 25 | **Interpreting, explaining and saving models** | [chapter](notebooks/part-5-scikit-learn/25-sklearn-interpret.ipynb) | [.py](scripts/part-5-scikit-learn/25-sklearn-interpret.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-5-scikit-learn/25-sklearn-interpret.ipynb) |

### ![](https://img.shields.io/badge/Part_VI-TensorFlow_%26_Keras-ff6a00?style=flat-square)

| # | Chapter | Read & run | Code only | Colab |
|---:|---|---|---|:---:|
| 26 | **TensorFlow fundamentals: tensors, gradients and data pipelines** | [chapter](notebooks/part-6-tensorflow-keras/26-tf-basics.ipynb) | [.py](scripts/part-6-tensorflow-keras/26-tf-basics.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-6-tensorflow-keras/26-tf-basics.ipynb) |
| 27 | **Neural networks with Keras** | [chapter](notebooks/part-6-tensorflow-keras/27-keras-mlp.ipynb) | [.py](scripts/part-6-tensorflow-keras/27-keras-mlp.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-6-tensorflow-keras/27-keras-mlp.ipynb) |
| 28 | **Deep learning architectures: CNNs, sequence models and custom training** | [chapter](notebooks/part-6-tensorflow-keras/28-keras-advanced.ipynb) | [.py](scripts/part-6-tensorflow-keras/28-keras-advanced.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-6-tensorflow-keras/28-keras-advanced.ipynb) |

### ![](https://img.shields.io/badge/Part_VII-ML_for_Research-d99a00?style=flat-square)

| # | Chapter | Read & run | Code only | Colab |
|---:|---|---|---|:---:|
| 29 | **The ML research protocol: a complete worked study** | [chapter](notebooks/part-7-ml-for-research/29-research-protocol.ipynb) | [.py](scripts/part-7-ml-for-research/29-research-protocol.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-7-ml-for-research/29-research-protocol.ipynb) |
| 30 | **Choosing models and metrics, and comparing them rigorously** | [chapter](notebooks/part-7-ml-for-research/30-research-choosing.ipynb) | [.py](scripts/part-7-ml-for-research/30-research-choosing.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-7-ml-for-research/30-research-choosing.ipynb) |
| 31 | **Reproducibility, leakage checklists and writing it up** | [chapter](notebooks/part-7-ml-for-research/31-research-reporting.ipynb) | [.py](scripts/part-7-ml-for-research/31-research-reporting.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-7-ml-for-research/31-research-reporting.ipynb) |

### ![](https://img.shields.io/badge/Part_VIII-Debugging-d42a2a?style=flat-square)

| # | Chapter | Read & run | Code only | Colab |
|---:|---|---|---|:---:|
| 32 | **How to debug: reading tracebacks and a systematic method** | [chapter](notebooks/part-8-debugging/32-debug-strategy.ipynb) | [.py](scripts/part-8-debugging/32-debug-strategy.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-8-debugging/32-debug-strategy.ipynb) |
| 33 | **The error encyclopedia** | [chapter](notebooks/part-8-debugging/33-debug-encyclopedia.ipynb) | [.py](scripts/part-8-debugging/33-debug-encyclopedia.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-8-debugging/33-debug-encyclopedia.ipynb) |
| 34 | **When the code runs but the model is wrong** | [chapter](notebooks/part-8-debugging/34-debug-models.ipynb) | [.py](scripts/part-8-debugging/34-debug-models.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/notebooks/part-8-debugging/34-debug-models.ipynb) |

### ![](https://img.shields.io/badge/Appendices-reference-596075?style=flat-square)

| | |
|---|---|
| **A** | [Quick-reference sheets](book/quick-reference.md) |
| **B** | [Glossary and further reading](book/glossary.md) |
| **C** | [Solutions: Parts I–IV](solutions/solutions-parts-1-4.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/solutions/solutions-parts-1-4.ipynb) |
| **D** | [Solutions: Parts V–VIII](solutions/solutions-parts-5-8.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mahir-606/annotated-ml-handbook/blob/main/solutions/solutions-parts-5-8.ipynb) |
| | [Index of definitions](book/index-of-definitions.md) · [All exercises on one page](exercises/README.md) |

## Run it on your computer

```bash
git clone https://github.com/Mahir-606/annotated-ml-handbook.git
cd annotated-ml-handbook
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                        # then open notebooks/part-1-python/01-python-basics.ipynb
```

With conda: `conda env create -f environment.yml && conda activate annotated-ml`.
Run a chapter's code as a plain script: `python scripts/part-5-scikit-learn/21-sklearn-regression.py`.
Check that everything runs on your machine: `python tools/run_all.py`.

> **Tip:** each notebook starts with a small setup cell. It applies the book's figure style, copies in the practice dataset, and sends any files the examples create into an `outputs/` folder, which git ignores.

## Repository layout

```text
annotated-ml-handbook/
├── book/                 contents page, preface, how to use, quick reference, glossary, index
├── notebooks/            the 34 chapters as full-text notebooks, grouped by part
├── solutions/            worked solutions to every exercise (full text + code)
├── exercises/            every exercise on one page
├── scripts/              each chapter's code as a plain .py file
├── figures/              every figure the code generates, by chapter
├── data/                 insurance.csv, the practice dataset from Chapter 11
├── docs/                 the web edition and PDF (served by GitHub Pages)
├── book-source/          the book's source: chapters in Markdown + the build system
├── tools/                run_all.py (runs every notebook), set_github_user.py
└── assets/               covers and social preview
```

## Versions

Every example was run with these versions, and the outputs shown come from them:

| Python | NumPy | pandas | scikit-learn | Matplotlib | Seaborn | Plotly | TensorFlow |
|---|---|---|---|---|---|---|---|
| 3.13 | 2.5.3 | 3.0.5 | 1.9.1 | 3.11.2 | 0.13.2 | 7.2.0 | 2.21.0 |

Newer versions usually work. GitHub re-runs every notebook each month and after every change (the badge at the top), so breakage from new library releases is caught early.

## Found a problem?

- **Code doesn't run or gives a different result?** [Open a code issue](../../issues/new?template=code-problem.yml).
- **Typo or mistake in the text?** [Report an erratum](../../issues/new?template=erratum.yml).
- Want to contribute? Read [CONTRIBUTING.md](CONTRIBUTING.md). The text lives in [book-source/chapters](book-source/chapters).

## Cite

GitHub's **Cite this repository** button (from [CITATION.cff](CITATION.cff)) gives APA and BibTeX:

```bibtex
@book{khan2026annotatedml,
  author    = {Mahir Absar Khan},
  title     = {The Annotated ML Handbook: Python, NumPy, Pandas, Matplotlib, Seaborn, Plotly, scikit-learn and TensorFlow},
  year      = {2026},
  edition   = {First},
  url       = {https://github.com/Mahir-606/annotated-ml-handbook}
}
```

## Author

**Mahir Absar Khan** is an industrial and production engineer and a Research Assistant at the Computational Intelligence and Operations Laboratory. He researches machine learning and LLMs in supply chains, manufacturing, and healthcare, and has co-authored papers at ACM, ICML, IJCAI, and AAAI venues.

If the book helps you, ⭐ star the repository so others can find it.

## License

- **Code** (all code cells, scripts, solutions, tools, build system): [MIT License](LICENSE). Use it freely in your own programs, coursework and research.
- **Book text, figures and cover art**: © 2026 Mahir Absar Khan. Free to read, download and use for personal study; please do not republish or sell it. See [LICENSE-CONTENT.md](LICENSE-CONTENT.md).
