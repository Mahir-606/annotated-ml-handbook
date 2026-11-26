"""The Annotated ML Handbook, Chapter 33: The error encyclopedia

Companion script by Mahir Absar Khan. Cells are marked with "# %%" so you can run them one
at a time in VS Code, PyCharm or Spyder, or run the whole file: python 33-debug-encyclopedia.py
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


# %% [markdown]
# ## 33.1 Python errors · *Beginner*
#
# ### 33.1.1 `SyntaxError` and `IndentationError`
#
# **Missing colon**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
code = "if x > 1\n    print(x)"
compile(code, "<example>", "exec")
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.2 `NameError: name '…' is not defined`
#
# **Typo or never defined**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
learning_rate = 0.01
print(learnig_rate)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.3 `TypeError: unsupported operand type(s)` / `can only concatenate str`
#
# **Mixing types**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
total = "5" + 3
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.4 `TypeError: 'X' object is not callable`
#
# **A built-in shadowed by a variable**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
list = [1, 2, 3]
list((4, 5))
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.5 `TypeError: … takes N positional arguments but M were given`
#
# **Forgot self, or passed too many arguments**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
class Scaler:
    def fit(X):
        return X
Scaler().fit([1, 2])
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.6 `IndexError: list index out of range`
#
# **Index past the end**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
scores = [0.8, 0.9, 0.85]
scores[3]
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.7 `KeyError`
#
# **Missing dictionary key**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
config = {"lr": 0.01, "epochs": 10}
config["batch_size"]
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.8 `AttributeError: '…' object has no attribute '…'`
#
# **Method on the wrong type, or on None**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
data = [3, 1, 2]
result = data.sort()
result.append(4)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.9 `ValueError: invalid literal for int() with base 10`
#
# **Converting a non-numeric string**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
int("3.7")
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.10 `ZeroDivisionError`
#
# **Division by zero**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
correct, total = 0, 0
accuracy = correct / total
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.11 `ModuleNotFoundError: No module named '…'`
#
# **Library not installed in this environment**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
import sklearn_extra_magic
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.12 `FileNotFoundError: [Errno 2] No such file or directory`
#
# **Wrong path or working directory**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
open("data/raw/survey.csv")
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.1.13 `RecursionError: maximum recursion depth exceeded`
#
# **Recursion without a base case**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
def countdown(n):
    return countdown(n - 1)
countdown(5)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ## 33.2 NumPy errors · *Beginner*
#
# ### 33.2.1 `ValueError: operands could not be broadcast together with shapes`
#
# **Incompatible shapes**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
import numpy as np
X = np.ones((100, 3))
weights = np.ones(100)
X * weights
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.2.2 `ValueError: matmul: Input operand 1 has a mismatch in its core dimension`
#
# **Matrix product with incompatible inner dimensions**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
A = np.ones((4, 3))
B = np.ones((4, 2))
A @ B
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.2.3 `ValueError: cannot reshape array of size N into shape …`
#
# **Total size mismatch**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
np.arange(10).reshape(4, 3)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.2.4 `ValueError: The truth value of an array with more than one element is ambiguous`
#
# **Using `if` or `and` with an array**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
a = np.array([1, 5, 3])
if a > 2:
    print("big")
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.2.5 `IndexError: too many indices for array`
#
# **Indexing a 1-D array as if it were 2-D**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
v = np.array([1, 2, 3])
v[:, 0]
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.2.6 `numpy.exceptions.AxisError: axis 1 is out of bounds`
#
# **Axis beyond ndim**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
np.array([1, 2, 3]).sum(axis=1)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ## 33.3 pandas errors · *Beginner*
#
# ### 33.3.1 `KeyError` on a column
#
# **Column name mismatch**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
import pandas as pd
df = pd.DataFrame({"Age ": [30, 40], "Income": [50, 60]})
df["Age"]
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.3.2 `ValueError: Length of values does not match length of index`
#
# **Assigning a list of the wrong length**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
df = pd.DataFrame({"a": [1, 2, 3]})
df["b"] = [10, 20]
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.3.3 `ChainedAssignmentError` warning: the assignment has no effect
#
# **Chained indexing with assignment**


# %%
df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 0, 0]})
df[df["a"] > 1]["b"] = 99
print(df)


# %% [markdown]
# ### 33.3.4 `TypeError: Cannot perform reduction 'mean' with string dtype`
#
# **Numbers stored as text**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
df = pd.DataFrame({"price": ["1,200", "950", "1,050"]})
df["price"].mean()
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.3.5 `MergeError` / unexpected row counts after `merge`
#
# **validate catches a many-to-many join**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
left = pd.DataFrame({"id": [1, 1, 2], "x": [10, 11, 20]})
right = pd.DataFrame({"id": [1, 1, 2], "y": ["a", "b", "c"]})
pd.merge(left, right, on="id", validate="one_to_one")
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.3.6 `ValueError: Index contains duplicate entries, cannot reshape`
#
# **pivot with duplicates**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
long = pd.DataFrame({"subject": ["s1", "s1"], "week": ["w1", "w1"], "score": [5, 6]})
long.pivot(index="subject", columns="week", values="score")
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.3.7 `UnicodeDecodeError` when reading a CSV
#
# **Wrong text encoding**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
from pathlib import Path
Path("latin.csv").write_bytes("name\nJosé\n".encode("latin-1"))
pd.read_csv("latin.csv")
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ## 33.4 scikit-learn errors · *Intermediate*
#
# ### 33.4.1 `ValueError: Expected a 2-dimensional container but got <class 'pandas.Series'> instead`
#
# **One feature passed as a Series**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
from sklearn.linear_model import LinearRegression
df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.1, 5.9]})
LinearRegression().fit(df["x"], df["y"])
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.4.2 `ValueError: Input X contains NaN`
#
# **Missing values reach an estimator that can't handle them**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
from sklearn.linear_model import LogisticRegression
X = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 0.5]])
LogisticRegression().fit(X, [0, 1, 0])
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.4.3 `ValueError: could not convert string to float`
#
# **Text categories passed to a numeric model**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
X = pd.DataFrame({"age": [30, 40, 50], "city": ["Oslo", "Lima", "Oslo"]})
LogisticRegression().fit(X, [0, 1, 0])
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.4.4 `ValueError: Found input variables with inconsistent numbers of samples`
#
# **X and y lengths differ**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
X = np.random.default_rng(0).normal(size=(100, 3))
y = np.zeros(98)
LinearRegression().fit(X, y)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.4.5 `NotFittedError: This … instance is not fitted yet`
#
# **predict before fit**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
from sklearn.preprocessing import StandardScaler
StandardScaler().transform([[1.0], [2.0]])
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.4.6 `ValueError: Found unknown categories … during transform`
#
# **A category in the test data that training never saw**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder().fit(pd.DataFrame({"city": ["Oslo", "Lima"]}))
enc.transform(pd.DataFrame({"city": ["Pune"]}))
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.4.7 `ValueError: The feature names should match those that were passed during fit`
#
# **Columns renamed or reordered after fitting**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
model = LinearRegression().fit(pd.DataFrame({"a": [1.0, 2, 3], "b": [1.0, 0, 1]}), [1.0, 2, 3])
model.predict(pd.DataFrame({"b": [1.0], "a": [2.0]}))
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.4.8 `ValueError: Invalid parameter … for estimator`
#
# **Wrong parameter name in a grid search**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
pipe = make_pipeline(StandardScaler(), LogisticRegression())
GridSearchCV(pipe, {"C": [0.1, 1]}).fit(np.random.default_rng(0).normal(size=(20, 2)), [0, 1] * 10)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.4.9 `ValueError: Target is multiclass but average='binary'`
#
# **Binary metric on a multiclass problem**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
from sklearn.metrics import f1_score
f1_score([0, 1, 2, 2], [0, 2, 1, 2])
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.4.10 `ValueError: The least populated classes in y have only 1 member`
#
# **Stratifying with a class that has a single example**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
from sklearn.model_selection import train_test_split
train_test_split(np.arange(6), [0, 0, 0, 1, 1, 2], stratify=[0, 0, 0, 1, 1, 2], test_size=0.5)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.4.11 `ConvergenceWarning: lbfgs failed to converge`
#
# **The optimiser hit max_iter**


# %%
from sklearn.datasets import load_breast_cancer
Xb, yb = load_breast_cancer(return_X_y=True)
LogisticRegression(max_iter=20).fit(Xb, yb)


# %% [markdown]
# ## 33.5 TensorFlow / Keras errors · *Intermediate*
#
# ### 33.5.1 `Input 0 … is incompatible with the layer: expected axis -1 of input shape to have value …`
#
# **Data has a different number of features than the model expects**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([keras.Input(shape=(4,)), layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
model.fit(np.zeros((10, 5), dtype="float32"), np.zeros(10), verbose=0)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.5.2 `ValueError: Arguments target and output must have the same rank`: labels don't match the loss
#
# **One-hot loss with integer labels**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
clf = keras.Sequential([keras.Input(shape=(3,)), layers.Dense(4, activation="softmax")])
clf.compile(optimizer="adam", loss="categorical_crossentropy")
clf.fit(np.zeros((8, 3), dtype="float32"), np.array([0, 1, 2, 3, 0, 1, 2, 3]), verbose=0)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.5.3 `InvalidArgumentError: … expected to be a float tensor but is a double tensor`
#
# **Mixing NumPy float64 with float32 tensors**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
w = tf.Variable(tf.ones((2,)))
x64 = np.array([1.0, 2.0])
tf.add(w, tf.constant(x64))
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 33.5.4 Loss becomes `nan`
#
# **Detecting NaN during training**


# %%
bad_X = np.array([[1.0], [np.nan], [3.0]], dtype="float32")
m = keras.Sequential([keras.Input(shape=(1,)), layers.Dense(1)])
m.compile(optimizer="adam", loss="mse")
h = m.fit(bad_X, np.array([1.0, 2.0, 3.0]), epochs=2, verbose=0,
          callbacks=[keras.callbacks.TerminateOnNaN()])
print("losses:", h.history["loss"])


# %% [markdown]
# ### 33.5.5 Out of memory (`ResourceExhaustedError: OOM when allocating tensor`)
#
# ## 33.6 Notebook and environment gotchas · *Beginner*
#
# ## Try it yourself
#
# Pick any three errors from this chapter, reproduce them in your own environment, then fix each one. Being able to *cause* an error on purpose is the fastest way to recognise it in the wild.
#
# Worked solutions: [solutions-parts-5-8.ipynb](../../solutions/solutions-parts-5-8.ipynb), section *Chapter 33*.
