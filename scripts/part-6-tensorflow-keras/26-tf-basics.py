"""The Annotated ML Handbook, Chapter 26: TensorFlow fundamentals: tensors, gradients and data pipelines

Companion script by Mahir Absar Khan. Cells are marked with "# %%" so you can run them one
at a time in VS Code, PyCharm or Spyder, or run the whole file: python 26-tf-basics.py
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
# ## 26.1 Tensors · *Beginner*
#
# **Creating and inspecting tensors**


# %%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

print("TensorFlow", tf.__version__)
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(a)
print(a.shape, a.dtype, a.ndim)
b = tf.zeros((2, 3)); c = tf.ones((3,)); r = tf.random.normal((2, 2), seed=1)
print(b.shape, c.numpy(), r.dtype)
x = tf.constant(np.arange(6).reshape(2, 3))
print(x.dtype, tf.cast(x, tf.float32).dtype)


# %% [markdown]
# ## 26.2 Operations · *Beginner*
#
# **Maths, reductions, reshaping, indexing**


# %%
m = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(m + 10, m * m, sep="\n")
print(tf.matmul(m, tf.transpose(m)))
print(tf.reduce_sum(m), tf.reduce_mean(m, axis=0).numpy())
print(tf.reshape(m, (3, 2)).shape, tf.expand_dims(m, 0).shape)
print(m[:, 1].numpy(), tf.argmax(m, axis=1).numpy())
print(tf.nn.softmax(tf.constant([2.0, 1.0, 0.1])).numpy().round(3))


# %% [markdown]
# ## 26.3 Variables and automatic differentiation · *Intermediate*
#
# **Gradients with GradientTape**


# %%
w = tf.Variable(3.0)
with tf.GradientTape() as tape:
    loss = w ** 2 + 2 * w
grad = tape.gradient(loss, w)
print(float(loss), float(grad))

x = tf.constant([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.reduce_sum(tf.sin(x))
print(tape.gradient(y, x).numpy().round(4), np.cos([1, 2, 3]).round(4))


# %% [markdown]
# ### 26.3.1 Training a model by hand · *Intermediate*
#
# **Linear regression with a manual training loop**


# %%
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (200, 1)).astype("float32")
y = (3 * X[:, 0] + 2 + rng.normal(0, 0.2, 200)).astype("float32")

W = tf.Variable(tf.zeros((1,)))
b = tf.Variable(0.0)
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
trace = []
for step in range(100):
    with tf.GradientTape() as tape:
        pred = X[:, 0] * W + b
        loss = tf.reduce_mean((pred - y) ** 2)
    grads = tape.gradient(loss, [W, b])
    opt.apply_gradients(zip(grads, [W, b]))
    trace.append((float(W[0]), float(b)))
print(f"learned slope={W.numpy()[0]:.3f}, intercept={b.numpy():.3f}, final loss={loss.numpy():.4f}")
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot([t[0] for t in trace], label="slope W"); ax.plot([t[1] for t in trace], label="intercept b")
ax.axhline(3, ls=":", color="grey"); ax.axhline(2, ls=":", color="grey")
ax.set(xlabel="training step", ylabel="value", title="Parameters during gradient descent"); ax.legend()
plt.show()


# %% [markdown]
# ## 26.4 Speed: tf.function · *Advanced*
#
# **Graph compilation**


# %%
@tf.function
def train_step(X, y):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((X[:, 0] * W + b - y) ** 2)
    grads = tape.gradient(loss, [W, b])
    opt.apply_gradients(zip(grads, [W, b]))
    return loss

print(float(train_step(X, y)))


# %% [markdown]
# ## 26.5 Input pipelines with tf.data · *Intermediate*
#
# **Building a dataset pipeline**


# %%
Xd = rng.normal(size=(1000, 4)).astype("float32")
yd = (Xd.sum(axis=1) > 0).astype("int32")
ds = tf.data.Dataset.from_tensor_slices((Xd, yd))
ds = (ds.shuffle(buffer_size=1000, seed=0)
        .map(lambda x, t: (x * 2.0, t), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE))
for xb, tb in ds.take(1):
    print(xb.shape, tb.shape, tb[:8].numpy())
print("batches per epoch:", len(list(ds)))


# %% [markdown]
# ## 26.6 Reproducibility · *Beginner*
#
# **One call seeds Python, NumPy and TensorFlow**


# %%
tf.keras.utils.set_random_seed(42)
print(tf.random.uniform((3,)).numpy().round(4))
tf.keras.utils.set_random_seed(42)
print(tf.random.uniform((3,)).numpy().round(4))


# %% [markdown]
# ## Try it yourself
#
# 1. Use `GradientTape` to find the minimum of f(x) = (x − 4)² + 1 by gradient descent, starting from x = 0.
# 2. Extend the manual training loop to two input features.
# 3. Build a `tf.data` pipeline from the breast-cancer data (`load_breast_cancer`) with batches of 64.
#
# Worked solutions: [solutions-parts-5-8.ipynb](../../solutions/solutions-parts-5-8.ipynb), section *Chapter 26*.
