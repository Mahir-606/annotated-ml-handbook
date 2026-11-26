# Machine learning from scratch with NumPy

Libraries hide the maths. Building a few models by hand, once, makes everything in Parts V and VI less magical: you will know what `fit` is doing, why scaling matters, and where a learning rate comes in.

## The core idea of supervised learning {B}

:::def
**Model**: a mathematical function with adjustable parameters that maps inputs (features) to outputs (predictions).

**Parameter**: a number inside a model that is learned from data, such as a regression coefficient or a neural-network weight.

**Loss function**: a measure of how wrong a model's predictions are, which training tries to minimise. Examples: mean squared error, cross-entropy.
:::


You have inputs **X** (shape `(n_samples, n_features)`) and targets **y** (shape `(n_samples,)`). A model is a function with adjustable **parameters**. *Training* means searching for parameter values that make predictions close to `y`, measured by a **loss function**. Everything else is detail.

## Making a dataset and splitting it {B}

```py title="Synthetic data with a known answer"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.

rng = np.random.default_rng(7)  #> Seeded random generator for reproducible data.
n = 200  #> Number of samples.
X = rng.uniform(0, 10, size=(n, 1))                    #> One feature, 200 samples, shape (200, 1).
true_w, true_b = 2.5, -4.0                             #> The "truth" we hope to recover.
y = true_w * X[:, 0] + true_b + rng.normal(0, 2, n)    #> Linear signal + Gaussian noise. X[:, 0] makes it 1-D.

idx = rng.permutation(n)                               #> Shuffle the row indices.
split = int(0.8 * n)                                   #> 80% train, 20% test.
train_idx, test_idx = idx[:split], idx[split:]  #> First 160 shuffled indices for training, the remaining 40 for testing.
X_train, X_test = X[train_idx], X[test_idx]            #> Fancy indexing selects the rows.
y_train, y_test = y[train_idx], y[test_idx]  #> Split the targets with the same indices.
print(X_train.shape, X_test.shape, y_train.shape)  #> (160, 1) (40, 1) (160,).
```

## Linear regression, the closed-form way {I}

For a linear model ŷ = Xw, the weights that minimise squared error solve the *normal equations*. We add a column of ones so the model can learn an intercept.

```py title="Least squares in three lines"
def add_bias(X):  #> Define a function `add_bias` that takes `X`.
    return np.column_stack([np.ones(len(X)), X])      #> Prepend a column of 1s: its weight becomes the intercept b.

Xb = add_bias(X_train)                                 #> Shape (160, 2).
w, *_ = np.linalg.lstsq(Xb, y_train, rcond=None)       #> Solves min ||Xb w - y||^2 stably (better than inv(X.T X) X.T y).
print(f"intercept={w[0]:.3f}  slope={w[1]:.3f}")       #> Close to the true -4.0 and 2.5.

y_pred = add_bias(X_test) @ w                          #> Predictions on unseen data.
mse = np.mean((y_test - y_pred) ** 2)                  #> Mean squared error.
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)   #> R²: share of variance explained.
print(f"test MSE={mse:.3f}  RMSE={np.sqrt(mse):.3f}  R2={r2:.3f}")  #> RMSE is the square root of MSE, in the units of y.
```

## Linear regression by gradient descent {I}

:::def
**Gradient**: the vector of partial derivatives of the loss with respect to each parameter. It points in the direction of steepest increase.

**Gradient descent**: an algorithm that repeatedly moves the parameters a small step against the gradient, so the loss decreases.

**Learning rate**: the size of each gradient-descent step. Too small is slow; too large overshoots and can diverge.

**Epoch**: one full pass through the training data.
:::


Most ML models (and all neural networks) have no closed-form solution. Instead we start with a guess and repeatedly step **downhill** on the loss. For MSE loss L = mean((Xw − y)²), the gradient is ∇L = (2/n)·Xᵀ(Xw − y).

```py title="Gradient descent, step by step"
def gradient_descent(X, y, lr=0.01, epochs=500):  #> Define a function `gradient_descent` that takes `X`, `y`, `lr` and `epochs`.
    Xb = add_bias(X)  #> Add the column of ones.
    w = np.zeros(Xb.shape[1])                  #> Start from all-zero weights.
    history = []  #> Will record the loss at each epoch.
    for epoch in range(epochs):  #> Loop over `range(epochs)`, naming each item `epoch`.
        error = Xb @ w - y                     #> Residuals for every sample at once (vectorized).
        loss = np.mean(error ** 2)  #> Current mean squared error.
        grad = 2 / len(y) * Xb.T @ error       #> The gradient: direction of steepest INCREASE in loss.
        w -= lr * grad                         #> Step the opposite way, scaled by the learning rate.
        history.append(loss)  #> Store it for the learning-curve plot.
    return w, history  #> Send `w, history` back to the caller.

w_gd, hist = gradient_descent(X_train, y_train, lr=0.02, epochs=2000)  #> Train for 2000 epochs with learning rate 0.02.
print(f"GD intercept={w_gd[0]:.3f} slope={w_gd[1]:.3f} final loss={hist[-1]:.3f}")  #> Should match the closed-form solution closely.
```

```py title="Visualise the fit and the learning curve" caption="Left: the fitted line recovers the true relationship. Right: the training loss drops quickly, then flattens as gradient descent converges."
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))       #> One figure, two side-by-side panels (axes).
ax1.scatter(X_train, y_train, s=14, alpha=0.6, label="train")  #> s = marker size, alpha = transparency.
ax1.scatter(X_test, y_test, s=14, alpha=0.8, label="test")  #> Test points in the second colour.
xs = np.linspace(0, 10, 100).reshape(-1, 1)                    #> A smooth grid of x values for drawing the line.
ax1.plot(xs, add_bias(xs) @ w_gd, color="black", lw=2, label="fitted line")  #> Predictions along the grid draw the fitted line.
ax1.set(xlabel="x", ylabel="y", title="Linear regression fit")  #> Set several properties at once.
ax1.legend()  #> Show a legend built from each plotted series' `label`.
ax2.plot(hist)                                                 #> Loss after each epoch.
ax2.set(xlabel="epoch", ylabel="MSE (log scale)", yscale="log", title="Training loss")  #> A log y-axis shows both the fast early drop and the slow tail.
plt.tight_layout()                                             #> Prevent overlapping labels.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

### The learning rate matters {I}

```py title="Too small, just right, too large" caption="A tiny learning rate crawls; a good one converges; one that is too large overshoots and the loss explodes."
fig, ax = plt.subplots(figsize=(7, 3.6))  #> One figure with one panel.
for lr in [0.001, 0.02, 0.0405]:                         #> Three learning rates to compare.
    _, h = gradient_descent(X_train, y_train, lr=lr, epochs=150)  #> Train; keep only the loss history (`_` discards the weights).
    ax.plot(h, label=f"lr = {lr}")  #> One curve per learning rate.
ax.set(yscale="log", xlabel="epoch", ylabel="training MSE", title="Effect of the learning rate")  #> Log scale, labels and title.
ax.set_ylim(1, 1e4)                                      #> Cap the y-axis so the exploding curve doesn't squash the others.
ax.legend()  #> Show a legend built from each plotted series' `label`.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

:::bug Loss goes up, or becomes `inf` / `nan`
The learning rate is too large (or features are on very different scales, which has the same effect). Lower the learning rate by 10x, and standardise your features.
:::

## Why feature scaling matters {I}

```py title="Standardisation, fitted on training data only"
X2 = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1000, n)])   #> Two features on wildly different scales.
mu = X2[train_idx].mean(axis=0)            #> Compute mean and SD from the TRAINING rows only...
sd = X2[train_idx].std(axis=0)  #> Column standard deviations from training rows only.
X2_train = (X2[train_idx] - mu) / sd       #> ...and apply the same numbers to train...
X2_test = (X2[test_idx] - mu) / sd         #> ...and test. Never recompute them on the test set.
print(X2_train.mean(axis=0).round(3), X2_train.std(axis=0).round(3))   #> Train: mean ~0, SD ~1.
print(X2_test.mean(axis=0).round(3))       #> Test: close to 0 but not exactly, as it should be.
```

:::research Data leakage starts here
If you compute scaling statistics on the *whole* dataset before splitting, information about the test set leaks into training and your test score becomes optimistic. scikit-learn's `Pipeline` ([[ch:sklearn-preprocessing]]) makes the correct procedure automatic.
:::

## Logistic regression from scratch {A}

:::def
**Sigmoid function**: σ(z) = 1 / (1 + e⁻ᶻ), which squashes any number into the range 0–1 so it can be read as a probability.

**Cross-entropy (log loss)**: the standard loss for classification. It heavily penalises confident wrong predictions.

**Decision boundary**: the line or surface in feature space where a classifier switches from predicting one class to another.
:::


For binary classification we pass the linear score through the **sigmoid** σ(z) = 1/(1+e⁻ᶻ) to get a probability, and minimise **binary cross-entropy** (log loss).

```py title="A classifier trained by gradient descent"
def sigmoid(z):  #> Define a function `sigmoid` that takes `z`.
    return 1 / (1 + np.exp(-z))  #> Send `1 / (1 + np.exp(-z))` back to the caller.

m = 300  #> Number of samples in the classification example.
Xc = np.vstack([rng.normal([-1.5, -1.5], 1.0, size=(m // 2, 2)),     #> Class 0: a cloud centred at (-1.5, -1.5).
                rng.normal([1.5, 1.5], 1.0, size=(m // 2, 2))])       #> Class 1: a cloud centred at (1.5, 1.5).
yc = np.r_[np.zeros(m // 2), np.ones(m // 2)]                         #> np.r_ concatenates: 150 zeros then 150 ones.

Xcb = add_bias(Xc)  #> Add the intercept column.
w = np.zeros(3)  #> Intercept + 2 feature weights, all starting at 0.
for _ in range(1000):  #> Loop over `range(1000)`, naming each item `_`.
    p = sigmoid(Xcb @ w)                          #> Predicted probabilities of class 1.
    grad = Xcb.T @ (p - yc) / m                   #> Gradient of the mean log loss (a famously simple formula).
    w -= 0.5 * grad  #> Gradient step with learning rate 0.5.
p = sigmoid(Xcb @ w)  #> Final predicted probabilities.
eps = 1e-12                                       #> Avoid log(0).
loss = -np.mean(yc * np.log(p + eps) + (1 - yc) * np.log(1 - p + eps))  #> Binary cross-entropy (log loss) over all samples.
acc = np.mean((p >= 0.5) == yc)                   #> Threshold probabilities at 0.5 to get class labels.
print(f"weights={w.round(3)}  log-loss={loss:.3f}  accuracy={acc:.3f}")  #> Learned weights, final loss and accuracy.
```

```py title="Plot the decision boundary" caption="The background shows the predicted probability of class 1; the white line is the 0.5 decision boundary learned from scratch."
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))   #> A 200x200 grid covering the plot.
grid = np.column_stack([xx.ravel(), yy.ravel()])                         #> Flatten into a list of (x, y) points.
probs = sigmoid(add_bias(grid) @ w).reshape(xx.shape)                    #> Predict every grid point, then reshape back.
fig, ax = plt.subplots(figsize=(6, 4.6))  #> One panel, slightly taller than wide.
cs = ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.8)     #> Filled contours of probability.
ax.contour(xx, yy, probs, levels=[0.5], colors="white", linewidths=2)    #> The boundary where p = 0.5.
ax.scatter(Xc[:, 0], Xc[:, 1], c=yc, cmap="RdBu_r", edgecolor="k", s=18)  #> The data points, coloured by true class.
fig.colorbar(cs, label="P(class 1)")  #> A colour bar explains the background shading.
ax.set(title="Logistic regression decision boundary", xlabel="feature 1", ylabel="feature 2")  #> Title and axis labels.
ax.grid(False)  #> Turn off the grid lines on this panel.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Softmax and one-hot encoding {A}

:::def
**Softmax**: a function turning a vector of scores into probabilities that are positive and sum to 1. Used for multiclass outputs.

**One-hot encoding**: representing a category as a vector of 0s with a single 1 in the position of that category, such as class 2 of 4 → `[0, 0, 1, 0]`.
:::


For more than two classes, **softmax** turns a vector of scores into probabilities that sum to 1. Labels are often **one-hot encoded** to match.

```py title="Numerically stable softmax and one-hot labels"
def softmax(z):  #> Define a function `softmax` that takes `z`.
    z = z - z.max(axis=1, keepdims=True)       #> Subtracting the row max prevents overflow in exp() without changing the result.
    e = np.exp(z)  #> Exponentiate the shifted scores.
    return e / e.sum(axis=1, keepdims=True)    #> Normalise each row so it sums to 1.

scores = np.array([[2.0, 1.0, 0.1],  #> Raw scores for two samples and three classes: first row...
                   [1000.0, 1001.0, 999.0]])    #> The second row would overflow a naive exp().
print(softmax(scores).round(3))  #> Each row now sums to 1, with no overflow.
labels = np.array([0, 2, 1, 2])  #> Integer class labels.
one_hot = np.eye(3)[labels]                    #> Index rows of the identity matrix: a neat one-hot trick.
print(one_hot)  #> Display the value of `one_hot`.
print(one_hot.argmax(axis=1))                  #> argmax reverses it.
```

## k-nearest neighbours in five lines {A}

```py title="kNN classifier using broadcasting"
def knn_predict(X_train, y_train, X_new, k=5):  #> Define a function `knn_predict` that takes `X_train`, `y_train`, `X_new` and `k`.
    d = np.sqrt(((X_new[:, None, :] - X_train[None, :, :]) ** 2).sum(-1))   #> (n_new, n_train) distance matrix.
    nearest = np.argsort(d, axis=1)[:, :k]                                 #> Indices of the k closest training points.
    votes = y_train[nearest]                                               #> Their labels, shape (n_new, k).
    return (votes.mean(axis=1) >= 0.5).astype(int)                         #> Majority vote (binary labels 0/1).

perm = rng.permutation(m)  #> Shuffle the 300 indices.
tr, te = perm[:240], perm[240:]  #> 240 training and 60 test indices.
pred = knn_predict(Xc[tr], yc[tr], Xc[te], k=7)  #> Predict the test points from their 7 nearest training points.
print("kNN test accuracy:", np.mean(pred == yc[te]))  #> Share of correct predictions.
```

:::key What you just built
Train/test split, scaling, a regression model (two ways), gradient descent, a classifier, a decision boundary, softmax and kNN. scikit-learn implements all of these, faster and with many more safeguards, but the ideas are exactly the ones above.
:::

:::ex Try it yourself
1. Add L2 regularisation to `gradient_descent`: add `2 * alpha * w` to the gradient (excluding the intercept) and watch the slope shrink as `alpha` grows.
2. Change the logistic regression's class centres to (−0.5, −0.5) and (0.5, 0.5). What happens to accuracy, and why?
3. Make `knn_predict` work for more than two classes using `np.bincount` per row.
:::
