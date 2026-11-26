"""The Annotated ML Handbook, Chapter 10: Machine learning from scratch with NumPy

Companion script by Mahir Absar Khan. Cells are marked with "# %%" so you can run them one
at a time in VS Code, PyCharm or Spyder, or run the whole file: python 10-numpy-ml-scratch.py
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
# ## 10.1 The core idea of supervised learning · *Beginner*
#
# ## 10.2 Making a dataset and splitting it · *Beginner*
#
# **Synthetic data with a known answer**


# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
n = 200
X = rng.uniform(0, 10, size=(n, 1))
true_w, true_b = 2.5, -4.0
y = true_w * X[:, 0] + true_b + rng.normal(0, 2, n)

idx = rng.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = idx[:split], idx[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(X_train.shape, X_test.shape, y_train.shape)


# %% [markdown]
# ## 10.3 Linear regression, the closed-form way · *Intermediate*
#
# **Least squares in three lines**


# %%
def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])

Xb = add_bias(X_train)
w, *_ = np.linalg.lstsq(Xb, y_train, rcond=None)
print(f"intercept={w[0]:.3f}  slope={w[1]:.3f}")

y_pred = add_bias(X_test) @ w
mse = np.mean((y_test - y_pred) ** 2)
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
print(f"test MSE={mse:.3f}  RMSE={np.sqrt(mse):.3f}  R2={r2:.3f}")


# %% [markdown]
# ## 10.4 Linear regression by gradient descent · *Intermediate*
#
# **Gradient descent, step by step**


# %%
def gradient_descent(X, y, lr=0.01, epochs=500):
    Xb = add_bias(X)
    w = np.zeros(Xb.shape[1])
    history = []
    for epoch in range(epochs):
        error = Xb @ w - y
        loss = np.mean(error ** 2)
        grad = 2 / len(y) * Xb.T @ error
        w -= lr * grad
        history.append(loss)
    return w, history

w_gd, hist = gradient_descent(X_train, y_train, lr=0.02, epochs=2000)
print(f"GD intercept={w_gd[0]:.3f} slope={w_gd[1]:.3f} final loss={hist[-1]:.3f}")


# %% [markdown]
# **Visualise the fit and the learning curve**


# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.scatter(X_train, y_train, s=14, alpha=0.6, label="train")
ax1.scatter(X_test, y_test, s=14, alpha=0.8, label="test")
xs = np.linspace(0, 10, 100).reshape(-1, 1)
ax1.plot(xs, add_bias(xs) @ w_gd, color="black", lw=2, label="fitted line")
ax1.set(xlabel="x", ylabel="y", title="Linear regression fit")
ax1.legend()
ax2.plot(hist)
ax2.set(xlabel="epoch", ylabel="MSE (log scale)", yscale="log", title="Training loss")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 10.4.1 The learning rate matters · *Intermediate*
#
# **Too small, just right, too large**


# %%
fig, ax = plt.subplots(figsize=(7, 3.6))
for lr in [0.001, 0.02, 0.0405]:
    _, h = gradient_descent(X_train, y_train, lr=lr, epochs=150)
    ax.plot(h, label=f"lr = {lr}")
ax.set(yscale="log", xlabel="epoch", ylabel="training MSE", title="Effect of the learning rate")
ax.set_ylim(1, 1e4)
ax.legend()
plt.show()


# %% [markdown]
# ## 10.5 Why feature scaling matters · *Intermediate*
#
# **Standardisation, fitted on training data only**


# %%
X2 = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1000, n)])
mu = X2[train_idx].mean(axis=0)
sd = X2[train_idx].std(axis=0)
X2_train = (X2[train_idx] - mu) / sd
X2_test = (X2[test_idx] - mu) / sd
print(X2_train.mean(axis=0).round(3), X2_train.std(axis=0).round(3))
print(X2_test.mean(axis=0).round(3))


# %% [markdown]
# ## 10.6 Logistic regression from scratch · *Advanced*
#
# **A classifier trained by gradient descent**


# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

m = 300
Xc = np.vstack([rng.normal([-1.5, -1.5], 1.0, size=(m // 2, 2)),
                rng.normal([1.5, 1.5], 1.0, size=(m // 2, 2))])
yc = np.r_[np.zeros(m // 2), np.ones(m // 2)]

Xcb = add_bias(Xc)
w = np.zeros(3)
for _ in range(1000):
    p = sigmoid(Xcb @ w)
    grad = Xcb.T @ (p - yc) / m
    w -= 0.5 * grad
p = sigmoid(Xcb @ w)
eps = 1e-12
loss = -np.mean(yc * np.log(p + eps) + (1 - yc) * np.log(1 - p + eps))
acc = np.mean((p >= 0.5) == yc)
print(f"weights={w.round(3)}  log-loss={loss:.3f}  accuracy={acc:.3f}")


# %% [markdown]
# **Plot the decision boundary**


# %%
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])
probs = sigmoid(add_bias(grid) @ w).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(6, 4.6))
cs = ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.8)
ax.contour(xx, yy, probs, levels=[0.5], colors="white", linewidths=2)
ax.scatter(Xc[:, 0], Xc[:, 1], c=yc, cmap="RdBu_r", edgecolor="k", s=18)
fig.colorbar(cs, label="P(class 1)")
ax.set(title="Logistic regression decision boundary", xlabel="feature 1", ylabel="feature 2")
ax.grid(False)
plt.show()


# %% [markdown]
# ## 10.7 Softmax and one-hot encoding · *Advanced*
#
# **Numerically stable softmax and one-hot labels**


# %%
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

scores = np.array([[2.0, 1.0, 0.1],
                   [1000.0, 1001.0, 999.0]])
print(softmax(scores).round(3))
labels = np.array([0, 2, 1, 2])
one_hot = np.eye(3)[labels]
print(one_hot)
print(one_hot.argmax(axis=1))


# %% [markdown]
# ## 10.8 k-nearest neighbours in five lines · *Advanced*
#
# **kNN classifier using broadcasting**


# %%
def knn_predict(X_train, y_train, X_new, k=5):
    d = np.sqrt(((X_new[:, None, :] - X_train[None, :, :]) ** 2).sum(-1))
    nearest = np.argsort(d, axis=1)[:, :k]
    votes = y_train[nearest]
    return (votes.mean(axis=1) >= 0.5).astype(int)

perm = rng.permutation(m)
tr, te = perm[:240], perm[240:]
pred = knn_predict(Xc[tr], yc[tr], Xc[te], k=7)
print("kNN test accuracy:", np.mean(pred == yc[te]))


# %% [markdown]
# ## Try it yourself
#
# 1. Add L2 regularisation to `gradient_descent`: add `2 * alpha * w` to the gradient (excluding the intercept) and watch the slope shrink as `alpha` grows.
# 2. Change the logistic regression's class centres to (−0.5, −0.5) and (0.5, 0.5). What happens to accuracy, and why?
# 3. Make `knn_predict` work for more than two classes using `np.bincount` per row.
#
# Worked solutions: [solutions-parts-1-4.ipynb](../../solutions/solutions-parts-1-4.ipynb), section *Chapter 10*.
