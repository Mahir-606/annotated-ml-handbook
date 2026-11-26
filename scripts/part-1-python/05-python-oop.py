"""The Annotated ML Handbook, Chapter 5: Classes and objects

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-1-python/05-python-oop.ipynb.
Cells are marked "# %%" so you can run them one at a time in VS Code, PyCharm or Spyder.
"""

# %%
# Repository setup (not part of the book's examples): run this cell first.
import os, sys
from pathlib import Path

# Files the examples create (CSV, models, figures) go into ./outputs next to the notebook.
OUT = Path.cwd() if Path.cwd().name == "outputs" else Path.cwd() / "outputs"
OUT.mkdir(exist_ok=True)
os.chdir(OUT)

# insurance.csv is the practice dataset created in Chapter 11 and reused by later chapters.
# If it is not here yet, copy it from the repository's data/ folder (or download it in Colab).
if not Path("insurance.csv").exists():
    import shutil, urllib.request
    for base in OUT.parents:
        if (base / "data" / "insurance.csv").exists():
            shutil.copy(base / "data" / "insurance.csv", "insurance.csv")
            break
    else:
        try:
            urllib.request.urlretrieve("https://raw.githubusercontent.com/Mahir-606/"
                                       "annotated-ml-handbook/main/data/insurance.csv", "insurance.csv")
        except Exception as e:
            print("Could not fetch insurance.csv; run Chapter 11 first.", e)

# The figure style used in the book.
import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams.update({
    "figure.figsize": (7.2, 4.0), "figure.dpi": 100,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler(color=["#1f4fff", "#e0245e", "#0f9b62", "#ff6a00",
                                     "#7b3ff2", "#0091b3", "#d99a00", "#d42a2a"]),
    "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlelocation": "left",
    "font.size": 10.5, "legend.frameon": False, "image.cmap": "viridis",
})

# Plotly: interactive in Jupyter/Colab; tools/run_all.py stores static PNGs so GitHub can show them.
try:
    import plotly.io as pio
    pio.renderers.default = os.environ.get("PLOTLY_RENDERER") or (
        "colab" if "google.colab" in sys.modules else "notebook_connected")
except ImportError:
    pass


# %%
class Dataset:
    """A tiny container for numeric samples."""

    def __init__(self, name, values):
        self.name = name
        self.values = list(values)

    def mean(self):
        return sum(self.values) / len(self.values)

    def add(self, x):
        self.values.append(x)
        return self

ds = Dataset("heights", [1.62, 1.75, 1.80])
print(ds.name, ds.values)
print(round(ds.mean(), 3))
ds.add(1.70).add(1.68)
print(len(ds.values), round(ds.mean(), 3))


# %%
class Vector2:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __mul__(self, k):
        return Vector2(self.x * k, self.y * k)

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

v, w = Vector2(3, 4), Vector2(1, 1)
print(v + w, v * 2, abs(v), v == Vector2(3, 4))


# %%
class Model:
    def __init__(self, name):
        self.name = name
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        raise NotImplementedError("subclasses must implement predict")

class MeanRegressor(Model):
    def fit(self, X, y):
        self.mean_ = sum(y) / len(y)
        return super().fit(X, y)

    def predict(self, X):
        return [self.mean_ for _ in X]

class MajorityClassifier(Model):
    def fit(self, X, y):
        self.majority_ = max(set(y), key=y.count)
        return super().fit(X, y)

    def predict(self, X):
        return [self.majority_] * len(X)

reg = MeanRegressor("baseline").fit([[1], [2], [3]], [10, 20, 60])
clf = MajorityClassifier("baseline").fit([[0]] * 5, ["a", "b", "b", "c", "b"])
print(reg.predict([[4], [5]]), reg.fitted)
print(clf.predict([[0], [0]]))
print(isinstance(reg, Model), isinstance(reg, MajorityClassifier))


# %%
class Experiment:
    count = 0

    def __init__(self, lr):
        self.lr = lr
        Experiment.count += 1

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        if value <= 0:
            raise ValueError("learning rate must be positive")
        self._lr = value

    @staticmethod
    def default_seeds():
        return [0, 1, 2]

    @classmethod
    def fast(cls):
        return cls(lr=0.1)

e = Experiment(0.01)
f = Experiment.fast()
print(e.lr, f.lr, Experiment.count, Experiment.default_seeds())
try:
    e.lr = -1
except ValueError as err:
    print("Rejected:", err)


# %%
from dataclasses import dataclass, field, asdict

@dataclass
class TrainConfig:
    model: str
    lr: float = 1e-3
    epochs: int = 20
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])

cfg = TrainConfig("cnn", epochs=50)
print(cfg)
print(asdict(cfg))
print(cfg == TrainConfig("cnn", epochs=50))
