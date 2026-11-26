"""The Annotated ML Handbook, Chapter 33: The error encyclopedia

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-8-debugging/33-debug-encyclopedia.ipynb.
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


# %%
df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 0, 0]})
df[df["a"] > 1]["b"] = 99
print(df)


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


# %%
from sklearn.datasets import load_breast_cancer
Xb, yb = load_breast_cancer(return_X_y=True)
LogisticRegression(max_iter=20).fit(Xb, yb)


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


# %%
bad_X = np.array([[1.0], [np.nan], [3.0]], dtype="float32")
m = keras.Sequential([keras.Input(shape=(1,)), layers.Dense(1)])
m.compile(optimizer="adam", loss="mse")
h = m.fit(bad_X, np.array([1.0, 2.0, 3.0]), epochs=2, verbose=0,
          callbacks=[keras.callbacks.TerminateOnNaN()])
print("losses:", h.history["loss"])
