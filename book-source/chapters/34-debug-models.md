# When the code runs but the model is wrong

The hardest bugs raise no error at all: the model trains, prints a number, and the number is wrong, too good, or useless. This chapter is a symptom-to-cause guide plus the sanity checks that catch silent failures.

## Reading training curves {B}

```py figonly caption="Schematic training curves (illustrative shapes, not real runs). Learn to recognise these four patterns at a glance."
import numpy as np
import matplotlib.pyplot as plt
e = np.arange(1, 61)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.1), sharey=True)
pats = {
    "Good fit": (0.9 * np.exp(-e / 9) + 0.10, 0.9 * np.exp(-e / 9) + 0.14),
    "Underfitting": (0.6 * np.exp(-e / 30) + 0.45, 0.6 * np.exp(-e / 30) + 0.48),
    "Overfitting": (0.9 * np.exp(-e / 8) + 0.03, 0.9 * np.exp(-e / 8) + 0.18 + 0.006 * np.clip(e - 15, 0, None)),
    "LR too high": (0.5 + 0.25 * np.abs(np.sin(e / 2.3)) + 0.004 * e, 0.55 + 0.3 * np.abs(np.sin(e / 2.1)) + 0.005 * e),
}
for ax, (title, (tr, va)) in zip(axes, pats.items()):
    ax.plot(e, tr, label="training loss", lw=2); ax.plot(e, va, label="validation loss", lw=2)
    ax.set(title=title, xlabel="epoch"); ax.set_ylim(0, 1.2)
axes[0].set_ylabel("loss"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
```

## Symptom → likely cause → what to try {B}

| Symptom | Likely causes | What to try |
|---|---|---|
| Accuracy equals the majority-class share | model predicts one class only: imbalance, no signal, bad features, too much regularisation | check `np.unique(pred, return_counts=True)`; class weights; better metrics; less regularisation |
| Performance suspiciously high (≥ 0.99) | **leakage**: target-derived feature, duplicates across splits, preprocessing on all data, subject overlap | label-permutation test; grouped splits; inspect the top features; check duplicates |
| Great CV score, poor on new data | leakage in CV; dataset shift; tuned on the test set; CV not mimicking deployment | Pipelines; grouped/temporal splits; external validation |
| Training and validation both poor | underfitting; uninformative features; wrong target; labels misaligned with rows | more flexible model; feature engineering; verify `X`/`y` alignment |
| Training great, validation poor | overfitting | regularisation; more data; simpler model; early stopping |
| Loss is `nan` or explodes | learning rate too high; NaN/inf in data; unscaled features or targets; `log(0)` | scale data; lower LR; `TerminateOnNaN`; check `np.isnan`, `np.isinf` |
| Loss doesn't move at all | learning rate too low; dead ReLUs; frozen layers; wrong loss for the output; labels all the same | raise LR; check `trainable`; check output/loss table; check label counts |
| Results change wildly between runs | small data; unseeded randomness; unstable model | seeds; repeated CV; report mean ± SD; more data |
| Regression predicts nearly constant values | target not scaled for a neural net; strong regularisation; features unrelated | scale target; reduce alpha; check feature–target plots |
| Probabilities all near 0.5 or overconfident | poor calibration | `CalibratedClassifierCV`; calibration curve; Brier score |
| Validation better than training | dropout/regularisation active only in training; easier validation set; leakage | normal with dropout early on; check split composition |

## Sanity checks that catch silent bugs {I}

### 1. Can the model overfit a tiny subset?

A correctly wired network should memorise a handful of examples almost perfectly. If it can't, something in the pipeline is broken: labels, loss, shapes or learning rate.

```py title="Overfit 32 samples on purpose"
import os  #> Import the standard-library `os` module (operating-system tools).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #> Silence TensorFlow's start-up logs (set before importing).
from tensorflow import keras  #> Import `keras` from `tensorflow`.
from tensorflow.keras import layers  #> Import `layers` from `tensorflow.keras`.
from sklearn.datasets import load_digits  #> Import `load_digits` from scikit-learn's `datasets` module.

keras.utils.set_random_seed(0)  #> Reproducible.
Xd, yd = load_digits(return_X_y=True)  #> Digits as flat 64-pixel rows.
X_small, y_small = (Xd[:32] / 16).astype("float32"), yd[:32]       #> Just 32 images.
net = keras.Sequential([keras.Input(shape=(64,)), layers.Dense(128, activation="relu"), layers.Dense(10, activation="softmax")])  #> A small dense network.
net.compile(optimizer=keras.optimizers.Adam(1e-2), loss="sparse_categorical_crossentropy", metrics=["accuracy"])  #> A fairly high learning rate to memorise quickly.
h = net.fit(X_small, y_small, epochs=150, verbose=0)  #> Train only on the 32 samples.
print(f"final training accuracy on 32 samples: {h.history['accuracy'][-1]:.3f} (should be ~1.0)")  #> Near-perfect memorisation means the pipeline is wired correctly.
```

### 2. Are features and labels still aligned?

```py title="Shuffling rows separately destroys the relationship"
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
from sklearn.model_selection import cross_val_score  #> Import `cross_val_score` from scikit-learn's `model_selection` module.
from sklearn.linear_model import LogisticRegression  #> Import `LogisticRegression` from scikit-learn's `linear_model` module.
from sklearn.pipeline import make_pipeline  #> Import `make_pipeline` from scikit-learn's `pipeline` module.
from sklearn.preprocessing import StandardScaler  #> Import `StandardScaler` from scikit-learn's `preprocessing` module.

X = pd.DataFrame(Xd / 16)  #> Features as a DataFrame.
y = pd.Series(yd)  #> Labels as a Series.
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))  #> A standard classifier.
good = cross_val_score(pipe, X, y, cv=3).mean()  #> Correctly aligned data.
y_misaligned = y.sample(frac=1, random_state=0).reset_index(drop=True)   #> Simulates a bug: labels re-ordered, features not.
bad = cross_val_score(pipe, X, y_misaligned, cv=3).mean()  #> Same features, shuffled labels.
print(f"aligned: {good:.3f} | misaligned labels: {bad:.3f} (chance = 0.10)")  #> Misalignment drops performance to chance.
```

:::bug Silent misalignment
Sorting, filtering, dropping NaNs or merging `X` but not `y` (or vice versa) silently pairs features with the wrong labels. Keep `X` and `y` together in one DataFrame until the last moment, then split them with `X = df.drop(columns="target"); y = df["target"]`.
:::

### 3. What is the model actually predicting?

```py title="Look at the distribution of predictions"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
from sklearn.datasets import make_classification  #> Import `make_classification` from scikit-learn's `datasets` module.
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.
Xi, yi = make_classification(n_samples=2000, weights=[0.95], random_state=0)  #> 95% of samples in class 0.
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, stratify=yi, random_state=0)  #> Stratified split.
pred = LogisticRegression(C=1e-4, max_iter=1000).fit(Xtr, ytr).predict(Xte)   #> Over-regularised on purpose.
print("predicted class counts [0, 1]:", np.bincount(pred, minlength=2))   #> How often each class is predicted.
print("true class counts      [0, 1]:", np.bincount(yte, minlength=2))  #> What the counts should look like.
print("accuracy:", round(np.mean(pred == yte), 3), "<- looks fine, yet the model never predicts class 1")  #> High accuracy can hide a useless model.
```

### 4. Does the pipeline see what you think it sees?

```py title="Print the transformed feature matrix once"
from sklearn.compose import ColumnTransformer  #> Import `ColumnTransformer` from scikit-learn's `compose` module.
from sklearn.preprocessing import OneHotEncoder  #> Import `OneHotEncoder` from scikit-learn's `preprocessing` module.
from sklearn.impute import SimpleImputer  #> Import `SimpleImputer` from scikit-learn's `impute` module.
ins = pd.read_csv("insurance.csv")  #> Load the practice dataset.
pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), ["age", "bmi"]),  #> Impute two numeric columns...
                         ("cat", OneHotEncoder(drop="if_binary", sparse_output=False), ["smoker", "region"])])  #> ...and one-hot encode two categorical columns as a dense array.
Xt = pre.fit_transform(ins)  #> Transform the table.
print(pd.DataFrame(Xt, columns=pre.get_feature_names_out()).head(3).to_string())   #> Are the columns, values and scales what you expect? to_string() shows every column.
print("any NaN left:", bool(np.isnan(Xt).any()), "| shape:", Xt.shape)  #> Check that imputation worked and the width is as expected.
```

## A pre-submission checklist {B}

Before you trust a result enough to put it in a paper, tick every box:

- [ ] A dummy baseline was run, and the model clearly beats it.
- [ ] All preprocessing lives inside a Pipeline; the test set was used once.
- [ ] The split matches the data's structure (groups, time).
- [ ] The label-permutation test gives chance-level scores.
- [ ] Results are stable across seeds (mean ± SD reported).
- [ ] Metrics fit the problem (imbalance, costs) and come with confidence intervals.
- [ ] Top features make domain sense, and none could encode the answer.
- [ ] Training curves look healthy; the model can overfit a tiny subset.
- [ ] The full analysis reruns from scratch with one command and gives the same numbers.

:::ex Try it yourself
1. Break the "overfit 32 samples" check on purpose (learning rate 1e-6, or the wrong loss) and watch it fail.
2. Introduce a leak into the insurance pipeline (add a feature equal to `charges` plus noise) and see how the permutation importance and test score react.
3. Apply the pre-submission checklist to your own current project.
:::
