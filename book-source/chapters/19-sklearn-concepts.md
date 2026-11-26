@part 5

# Machine-learning concepts and the scikit-learn API

Before training any model you need a clear mental map: which kind of problem you have, how models learn, why we split data, and how to tell a model that *generalises* from one that has merely memorised. This chapter builds that map, then introduces the small, consistent API shared by every scikit-learn model.

## The kinds of machine learning {B}

:::def
**Machine learning**: building models that learn patterns from data in order to make predictions or decisions, rather than following hand-written rules.

**Supervised learning**: learning a mapping from features to a known target using labelled examples. It covers regression and classification.

**Unsupervised learning**: finding structure, such as groups or lower-dimensional summaries, in data without a target variable.

**Regression**: supervised learning where the target is a number (price, temperature).

**Classification**: supervised learning where the target is a category (spam/not spam, species).

**Feature**: an input variable used to make predictions; a column of X. Also called a predictor or covariate.

**Target (label)**: the variable a supervised model predicts; y.

**Sample (observation)**: one row of data: one patient, one image, one transaction.

**Hyperparameter**: a setting chosen before training that controls how a model learns, such as tree depth or the regularisation strength `C`. It is not learned from the data.
:::


| Type | You have | Goal | Examples | scikit-learn |
|---|---|---|---|---|
| **Supervised: regression** | features X and a numeric target y | predict a number | price, blood pressure, crop yield | `LinearRegression`, `RandomForestRegressor` |
| **Supervised: classification** | X and a categorical y | predict a class (and its probability) | disease yes/no, species, spam | `LogisticRegression`, `RandomForestClassifier` |
| **Unsupervised: clustering** | X only | find groups | customer segments, cell types | `KMeans`, `DBSCAN` |
| **Unsupervised: dimensionality reduction** | X only | compress or visualise | PCA of gene expression | `PCA`, `TSNE` |
| **Deep learning** | large X (images, text, signals) | any of the above, with learned features | image classification, forecasting | TensorFlow / Keras (Part VI) |

Vocabulary you will see everywhere: a **sample** (row, observation, instance), a **feature** (column, variable, predictor), the **target** (label, response, outcome), **training** (fitting parameters to data), **inference** (predicting on new data), **hyperparameters** (settings you choose *before* training, such as tree depth), and **parameters** (values the model learns, such as regression coefficients).

## The workflow at a glance {B}

```py figonly caption="The supervised learning workflow. The test set is locked away until the very end; all choices (preprocessing, model, hyperparameters) are made using the training data, usually with cross-validation."
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
steps = [("Question &\ndata", "#1f4fff"), ("Explore &\nclean", "#e0245e"), ("Split:\ntrain | test", "#7b3ff2"),
         ("Preprocess\n(pipeline)", "#0091b3"), ("Train &\ncross-validate", "#0f9b62"), ("Tune\nhyperparameters", "#ff6a00"),
         ("Final test\n(once)", "#d42a2a"), ("Interpret &\nreport", "#d99a00")]
fig, ax = plt.subplots(figsize=(11, 2.6)); ax.set_xlim(0, 8); ax.set_ylim(-0.9, 1.1); ax.axis("off")
for i, (txt, c) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((i + 0.06, 0.05), 0.86, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08", color=c))
    ax.text(i + 0.49, 0.5, txt, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    if i < 7: ax.annotate("", xy=(i + 1.06, 0.5), xytext=(i + 0.93, 0.5), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))
ax.annotate("", xy=(3.5, -0.05), xytext=(5.5, -0.05), arrowprops=dict(arrowstyle="-|>", color="#0f9b62", lw=1.6, connectionstyle="arc3,rad=-0.35"))
ax.text(4.5, -0.72, "iterate on training data only", ha="center", color="#0f9b62", fontsize=9.5, fontweight="bold")
plt.show()
```

## The estimator API: fit, predict, transform {B}

:::def
**Estimator**: any scikit-learn object that learns from data with `fit()`. Models (`predict`) and transformers (`transform`) are both estimators.

**Training (fitting)**: adjusting a model's parameters to the training data.

**Inference (prediction)**: using a trained model to produce outputs for new inputs.
:::


Every scikit-learn object follows the same small set of conventions:

| Method | Available on | What it does |
|---|---|---|
| `est = Model(**hyperparams)` | everything | create the object; hyperparameters go here |
| `est.fit(X, y)` | everything | learn from data; returns `est` itself |
| `est.predict(X)` | models | predict targets for new samples |
| `est.predict_proba(X)` | most classifiers | class probabilities, one column per class |
| `est.score(X, y)` | models | default metric: accuracy (classifiers) or R² (regressors) |
| `est.transform(X)` | transformers (scalers, encoders, PCA) | return a transformed version of X |
| `est.fit_transform(X)` | transformers | fit and transform in one step (training data only!) |
| `est.attr_` | fitted objects | learned values end with `_`, e.g. `coef_`, `classes_` |

Data conventions: `X` is 2-D, shape `(n_samples, n_features)` (a DataFrame or array); `y` is 1-D, shape `(n_samples,)`.

```py title="Your first model, end to end"
from sklearn.datasets import load_iris                     #> Small classic datasets ship with scikit-learn: no download needed.
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.
from sklearn.neighbors import KNeighborsClassifier  #> Import `KNeighborsClassifier` from scikit-learn's `neighbors` module.
from sklearn.metrics import accuracy_score  #> Import `accuracy_score` from scikit-learn's `metrics` module.

X, y = load_iris(return_X_y=True, as_frame=True)           #> X: DataFrame of 4 flower measurements; y: species coded 0, 1, 2.
print(X.shape, y.shape)  #> (150, 4) features and (150,) labels.
print(X.head(3))  #> The first three flowers.

X_train, X_test, y_train, y_test = train_test_split(  #> Split features and labels into training and test parts...
    X, y, test_size=0.25, random_state=42, stratify=y)     #> Hold out 25% for testing; stratify keeps class proportions equal in both parts.

model = KNeighborsClassifier(n_neighbors=5)                #> 1. Create, choosing hyperparameters.
model.fit(X_train, y_train)                                #> 2. Fit on TRAINING data only.
y_pred = model.predict(X_test)                             #> 3. Predict on unseen test data.
print("accuracy:", accuracy_score(y_test, y_pred))         #> 4. Compare predictions with the truth.
print("same via .score:", model.score(X_test, y_test))  #> `.score` computes accuracy for classifiers: the same number.
print(model.predict_proba(X_test.iloc[:3]).round(2))       #> Probability of each of the 3 species for the first 3 test flowers.
```

:::key Swap one line, keep the rest
Because every model shares this API, trying a different algorithm usually means changing only the line that creates it. That consistency is scikit-learn's greatest strength.
:::

```py title="Five classifiers, one loop"
from sklearn.linear_model import LogisticRegression  #> Import `LogisticRegression` from scikit-learn's `linear_model` module.
from sklearn.tree import DecisionTreeClassifier  #> Import `DecisionTreeClassifier` from scikit-learn's `tree` module.
from sklearn.ensemble import RandomForestClassifier  #> Import `RandomForestClassifier` from scikit-learn's `ensemble` module.
from sklearn.svm import SVC  #> Import `SVC` from scikit-learn's `svm` module.
from sklearn.naive_bayes import GaussianNB  #> Import `GaussianNB` from scikit-learn's `naive_bayes` module.

models = {  #> A dict mapping a display name to an (unfitted) model.
    "LogisticRegression": LogisticRegression(max_iter=1000),  #> A linear classifier; max_iter gives the solver room to converge.
    "DecisionTree": DecisionTreeClassifier(random_state=0),  #> A single tree of if/else splits.
    "RandomForest": RandomForestClassifier(random_state=0),  #> An ensemble of 100 trees (the default).
    "SVC": SVC(),  #> A support vector machine with the RBF kernel.
    "GaussianNB": GaussianNB(),  #> Naive Bayes assuming normally distributed features.
}  #> Close the dictionary opened above.
for name, m in models.items():  #> Loop over `models.items()`, naming each item `name, m`.
    m.fit(X_train, y_train)                                #> Identical calls for every model.
    print(f"{name:<20} test accuracy = {m.score(X_test, y_test):.3f}")  #> Name left-aligned in 20 characters, accuracy to 3 decimals.
```

## Baselines: is the model actually learning anything? {B}

:::def
**Baseline model**: a deliberately trivial model, such as always predicting the mean or the most frequent class, that any useful model must beat.
:::


Always compare against a trivial baseline. A model that is "90% accurate" is useless if 90% of samples belong to one class.

```py title="DummyClassifier and DummyRegressor"
from sklearn.dummy import DummyClassifier, DummyRegressor  #> Import `DummyClassifier` and `DummyRegressor` from scikit-learn's `dummy` module.
from sklearn.datasets import load_diabetes  #> Import `load_diabetes` from scikit-learn's `datasets` module.

base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)    #> Always predicts the most common class.
print("baseline accuracy:", round(base.score(X_test, y_test), 3))         #> About 1/3 for three balanced classes.

Xd, yd = load_diabetes(return_X_y=True)                                   #> A regression dataset: disease progression.
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, random_state=0)  #> Default split: 75% train, 25% test.
print("mean-predictor R2:", round(DummyRegressor(strategy="mean").fit(Xd_tr, yd_tr).score(Xd_te, yd_te), 3))   #> About 0 (slightly negative on new data).
```

## Overfitting, underfitting and the bias–variance trade-off {B}

:::def
**Overfitting**: when a model learns noise in the training data, performing well on it but poorly on new data.

**Underfitting**: when a model is too simple to capture the real pattern, performing poorly on training and new data alike.

**Bias–variance trade-off**: simple models err systematically (high bias); flexible models vary a lot with the training sample (high variance). Good models balance the two.

**Generalisation**: how well a model performs on new, unseen data, which is the goal of machine learning.
:::


A model **underfits** when it is too simple to capture the pattern (high *bias*): it is poor on training and test data alike. It **overfits** when it is so flexible that it memorises noise (high *variance*): excellent on training data, poor on new data. The goal is the sweet spot in between.

```py title="Polynomial degree controls flexibility" caption="Degree 1 underfits (misses the curve), degree 4 fits well, degree 15 overfits (chases noise). Training error keeps falling as flexibility grows, but test error turns back up: the signature of overfitting."
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
from sklearn.pipeline import make_pipeline  #> Import `make_pipeline` from scikit-learn's `pipeline` module.
from sklearn.preprocessing import PolynomialFeatures  #> Import `PolynomialFeatures` from scikit-learn's `preprocessing` module.
from sklearn.linear_model import LinearRegression  #> Import `LinearRegression` from scikit-learn's `linear_model` module.
from sklearn.metrics import mean_squared_error  #> Import `mean_squared_error` from scikit-learn's `metrics` module.

rng = np.random.default_rng(1)  #> Seeded generator.
x = np.sort(rng.uniform(0, 1, 40))  #> 40 sorted x values between 0 and 1.
y_true = np.sin(2 * np.pi * x)  #> The true underlying curve.
y = y_true + rng.normal(0, 0.25, x.size)                       #> A noisy sine wave.
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.4, random_state=3)  #> 60% train, 40% test.
grid = np.linspace(0, 1, 300)  #> Fine grid for drawing each fitted curve.

fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))  #> Four panels in a row.
for ax, deg in zip(axes[:3], [1, 4, 15]):  #> Loop over `zip(axes[:3], [1, 4, 15])`, naming each item `ax, deg`.
    m = make_pipeline(PolynomialFeatures(deg), LinearRegression())   #> Expand x into x, x², ..., x^deg, then fit a linear model.
    m.fit(x_tr.reshape(-1, 1), y_tr)                                 #> reshape(-1, 1): one feature as a 2-D column.
    ax.scatter(x_tr, y_tr, s=14, label="train"); ax.scatter(x_te, y_te, s=14, label="test")  #> Training and test points in two colours.
    ax.plot(grid, m.predict(grid.reshape(-1, 1)), color="black")  #> The fitted curve.
    ax.set(ylim=(-1.8, 1.8), title=f"degree {deg}")  #> Same y-range in every panel, so they compare fairly.
axes[0].legend(loc="lower left")  #> Legend in the first panel only.

degrees = range(1, 16)  #> Degrees 1 to 15.
tr_err, te_err = [], []  #> Lists for the training and test errors.
for deg in degrees:  #> Loop over `degrees`, naming each item `deg`.
    m = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(x_tr.reshape(-1, 1), y_tr)  #> Fit a polynomial of this degree.
    tr_err.append(mean_squared_error(y_tr, m.predict(x_tr.reshape(-1, 1))))  #> Error on the data it was trained on.
    te_err.append(mean_squared_error(y_te, m.predict(x_te.reshape(-1, 1))))  #> Error on unseen data.
axes[3].plot(degrees, tr_err, marker="o", ms=3, label="train MSE")  #> Training error curve.
axes[3].plot(degrees, te_err, marker="o", ms=3, label="test MSE")  #> Test error curve.
axes[3].set(yscale="log", xlabel="polynomial degree", title="error vs flexibility")  #> Log scale makes both curves visible.
axes[3].legend()  #> Legend for the two curves.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

| Symptom | Diagnosis | Remedies |
|---|---|---|
| train score low, test score low | underfitting (high bias) | more flexible model, better features, less regularisation, train longer |
| train score high, test score much lower | overfitting (high variance) | more data, simpler model, more regularisation, feature selection, early stopping, ensembles |
| train and test both high and close | good fit | confirm with cross-validation, then test once |

## Why the test set is sacred {B}

:::def
**Training set**: the data used to fit the model (and, via cross-validation, to choose among models).

**Validation set**: data held out from training to tune hyperparameters and compare models.

**Test set**: data locked away until the end and used once to estimate final performance on unseen data.
:::


The test set estimates how the model will perform on data it has never seen. **Every time you use test results to make a decision** (choose a model, tweak a feature, pick a threshold) you leak information from it into your modelling, and the final number becomes optimistic. Make all decisions with the training data, typically via **cross-validation** ([[ch:sklearn-model-selection]]), and evaluate on the test set once, at the end.

```py figonly caption="The three-way split. Cross-validation reuses the training data as rotating validation folds, so no separate validation set is wasted."
fig, ax = plt.subplots(figsize=(10, 2.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 3), 8, 0.8, color="#0f9b62")); ax.text(4, 3.4, "training data (80%)", ha="center", va="center", color="white", fontweight="bold")
ax.add_patch(plt.Rectangle((8, 3), 2, 0.8, color="#d42a2a")); ax.text(9, 3.4, "test (20%)", ha="center", va="center", color="white", fontweight="bold")
for k in range(5):
    y0 = 2.2 - k * 0.48
    for j in range(5):
        c = "#ff6a00" if j == k else "#9fd8bf"
        ax.add_patch(plt.Rectangle((j * 1.6, y0), 1.56, 0.4, color=c))
    ax.text(8.2, y0 + 0.2, f"fold {k + 1}", va="center", fontsize=9)
ax.text(9.0, 1.0, "orange = validation\ngreen = fit", fontsize=9, color="#444", va="center")
plt.show()
```

:::research Report what the test set was used for
In a paper, state how data were split (random, stratified, grouped by subject, by time), the split sizes, the random seed, and that the test set was used only once for the final evaluation. If you had to look at it more than once, say so.
:::

## Reproducibility knobs {B}

Many estimators and splitters are random. Pass `random_state=<int>` to make them deterministic: `train_test_split(..., random_state=42)`, `RandomForestClassifier(random_state=0)`, `KFold(shuffle=True, random_state=0)`. Use a fixed integer; avoid relying on a global NumPy seed.

:::ex Try it yourself
1. Load `load_wine()` and compare `DummyClassifier`, `KNeighborsClassifier` and `RandomForestClassifier` on a stratified split.
2. In the polynomial demo, replace `LinearRegression` with `Ridge(alpha=1e-3)` for degree 15. What happens to the wiggles?
3. Re-run the iris example with five different `random_state` values. How much does test accuracy vary? (This is why we cross-validate.)
:::
