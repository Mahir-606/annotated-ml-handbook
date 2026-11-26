# Classification: predicting categories

Classification models predict which class a sample belongs to, and usually how confident they are. Choosing the right **metric** matters more here than anywhere else in ML, because "accuracy" can be dangerously misleading. This chapter uses the Wisconsin breast-cancer dataset that ships with scikit-learn.

## The data and a first model {B}

```py title="Load, split, fit"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.
from sklearn.datasets import load_breast_cancer  #> Import `load_breast_cancer` from scikit-learn's `datasets` module.
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.
from sklearn.pipeline import make_pipeline  #> Import `make_pipeline` from scikit-learn's `pipeline` module.
from sklearn.preprocessing import StandardScaler  #> Import `StandardScaler` from scikit-learn's `preprocessing` module.
from sklearn.linear_model import LogisticRegression  #> Import `LogisticRegression` from scikit-learn's `linear_model` module.

data = load_breast_cancer(as_frame=True)  #> Load the dataset as pandas objects.
X = data.data                                              #> 30 numeric features computed from cell-nucleus images.
y = 1 - data.target                                        #> Original coding: 0 = malignant. We flip it so 1 = malignant, the class we want to DETECT ("positive").
print(X.shape, "| malignant share:", round(y.mean(), 3))  #> 569 samples, 30 features; about 37% malignant.

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)  #> Stratified 75/25 split.
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))  #> Scale, then logistic regression.
clf.fit(X_tr, y_tr)  #> Train on the training set.
y_pred = clf.predict(X_te)                                 #> Hard labels (0/1), using a 0.5 probability threshold.
y_prob = clf.predict_proba(X_te)[:, 1]                     #> Column 1 = probability of class 1 (malignant).
print("accuracy:", round(clf.score(X_te, y_te), 4))  #> Share of correct test predictions.
print("first five probabilities:", y_prob[:5].round(3))  #> Most are close to 0 or 1: a confident model.
```

## The confusion matrix and the metrics built from it {B}

:::def
**Confusion matrix**: a table of predicted vs actual classes. For two classes it holds the true positives, false positives, false negatives and true negatives.

**Positive class**: the class you want to detect (disease, fraud), coded 1. "Positive" means "the thing of interest", not "good".

**Accuracy**: the share of all predictions that are correct.

**Precision**: of the cases predicted positive, the share that really are positive: TP / (TP + FP).

**Recall (sensitivity)**: of the truly positive cases, the share the model detects: TP / (TP + FN).

**Specificity**: of the truly negative cases, the share correctly identified: TN / (TN + FP).

**F1 score**: the harmonic mean of precision and recall; high only when both are high.

**ROC curve and AUC**: the ROC curve plots recall against the false-positive rate at every threshold; the area under it (AUC) is the probability that a random positive is ranked above a random negative.
:::


For a binary problem with a chosen "positive" class:

| | Predicted negative | Predicted positive |
|---|---|---|
| **Actually negative** | True Negative (TN) | False Positive (FP), "false alarm" |
| **Actually positive** | False Negative (FN), "missed case" | True Positive (TP) |

| Metric | Formula | Question it answers |
|---|---|---|
| Accuracy | (TP+TN) / all | What fraction is correct overall? Misleading with imbalanced classes. |
| **Precision** | TP / (TP+FP) | When the model says positive, how often is it right? |
| **Recall** (sensitivity, TPR) | TP / (TP+FN) | Of all real positives, how many did we catch? |
| Specificity (TNR) | TN / (TN+FP) | Of all real negatives, how many did we correctly clear? |
| **F1** | harmonic mean of precision and recall | A single balance of the two. |
| ROC AUC | area under the ROC curve | How well does the model *rank* positives above negatives (threshold-free)? |
| PR AUC / average precision | area under precision–recall curve | Like ROC AUC, but more informative when positives are rare. |
| Balanced accuracy | mean of recall per class | Accuracy that is fair under imbalance. |
| MCC | correlation between prediction and truth | A robust single number for imbalanced binary problems (−1 to 1). |

```py title="Compute them all"
from sklearn.metrics import (confusion_matrix, classification_report, precision_score, recall_score,  #> Import several names from scikit-learn's `metrics` module (the list continues on the next lines).
                             f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score,  #> (continued list of metric functions)
                             matthews_corrcoef)  #> (end of the import list)

tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()        #> For binary labels, ravel() returns TN, FP, FN, TP in that order.
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")  #> The four counts.
print(f"precision={precision_score(y_te, y_pred):.3f}  recall={recall_score(y_te, y_pred):.3f}  "  #> Precision and recall...
      f"F1={f1_score(y_te, y_pred):.3f}  specificity={tn / (tn + fp):.3f}")  #> ...F1, and specificity computed by hand from the counts.
print(f"ROC AUC={roc_auc_score(y_te, y_prob):.4f}  PR AUC={average_precision_score(y_te, y_prob):.4f}")   #> Threshold-free metrics need PROBABILITIES, not labels.
print(f"balanced acc={balanced_accuracy_score(y_te, y_pred):.3f}  MCC={matthews_corrcoef(y_te, y_pred):.3f}")  #> Two imbalance-robust summaries.
print(classification_report(y_te, y_pred, target_names=["benign", "malignant"], digits=3))   #> Per-class precision, recall, F1 and support (count).
```

:::warn Passing labels where probabilities are needed
`roc_auc_score(y_te, y_pred)` with hard 0/1 labels runs without error but gives a degraded, misleading AUC. Always pass `predict_proba(X)[:, 1]` (or `decision_function(X)`) to ROC AUC, average precision and log loss.
:::

## Visual evaluation: confusion matrix, ROC and PR curves {I}

```py title="The three standard plots" caption="Left: counts of each outcome. Middle: ROC curve, where the diagonal is random guessing. Right: precision–recall curve, where the dashed line is the positive-class share (a random model)."
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay  #> Import `ConfusionMatrixDisplay`, `RocCurveDisplay` and `PrecisionRecallDisplay` from scikit-learn's `metrics` module.

fig, axes = plt.subplots(1, 3, figsize=(13, 4))  #> Three panels.
ConfusionMatrixDisplay.from_predictions(y_te, y_pred, display_labels=["benign", "malignant"],  #> Draw the confusion matrix with readable labels...
                                        cmap="Greens", colorbar=False, ax=axes[0])   #> Rows = truth, columns = prediction.
axes[0].set_title("Confusion matrix"); axes[0].grid(False)  #> Title; no grid over the cells.
RocCurveDisplay.from_predictions(y_te, y_prob, name="LogReg", ax=axes[1], plot_chance_level=True)  #> ROC curve from probabilities, with the random-guess diagonal.
axes[1].set_title("ROC curve")  #> Title.
PrecisionRecallDisplay.from_predictions(y_te, y_prob, name="LogReg", ax=axes[2], plot_chance_level=True)  #> Precision-recall curve, with the prevalence baseline.
axes[2].set_title("Precision–recall curve")  #> Title.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Choosing the decision threshold {I}

:::def
**Decision threshold**: the probability above which a classifier predicts the positive class (0.5 by default). Moving it trades precision against recall.
:::


`predict()` uses a probability threshold of 0.5, but nothing makes 0.5 special. In cancer screening a missed malignancy (FN) is far worse than a false alarm (FP), so we may accept more false alarms to raise recall.

```py title="Precision and recall at different thresholds" caption="Lowering the threshold catches more malignant cases (recall rises) at the cost of more false alarms (precision falls). The right threshold depends on the real-world costs of each error."
from sklearn.metrics import precision_recall_curve  #> Import `precision_recall_curve` from scikit-learn's `metrics` module.
prec, rec, thr = precision_recall_curve(y_te, y_prob)     #> Precision and recall at every possible threshold.
fig, ax = plt.subplots(figsize=(7.5, 3.8))  #> One panel.
ax.plot(thr, prec[:-1], label="precision")                #> prec/rec have one more element than thr.
ax.plot(thr, rec[:-1], label="recall")  #> Recall at each threshold.
ax.axvline(0.5, color="grey", ls="--", lw=1); ax.text(0.51, 0.62, "default 0.5", color="grey")  #> Mark the default threshold.
ax.set(xlabel="decision threshold", ylabel="score", title="Threshold trade-off", ylim=(0.6, 1.02))  #> Zoom the y-axis on the interesting range.
ax.legend()  #> Show a legend built from each plotted series' `label`.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
for t in [0.5, 0.3, 0.1]:  #> Loop over `[0.5, 0.3, 0.1]`, naming each item `t`.
    p = (y_prob >= t).astype(int)                          #> Apply a custom threshold by hand.
    print(f"threshold {t}: recall={recall_score(y_te, p):.3f} precision={precision_score(y_te, p):.3f} FN={((p == 0) & (y_te == 1)).sum()}")  #> Recall, precision and the number of missed malignancies at this threshold.
```

```py title="Let cross-validation choose the threshold"
from sklearn.model_selection import TunedThresholdClassifierCV  #> Import `TunedThresholdClassifierCV` from scikit-learn's `model_selection` module.

tuned = TunedThresholdClassifierCV(clf, scoring="balanced_accuracy", cv=5)   #> Tunes the threshold on TRAINING folds (scikit-learn >= 1.5).
tuned.fit(X_tr, y_tr)  #> Fits the model and searches thresholds with internal CV.
print("chosen threshold:", round(tuned.best_threshold_, 3))  #> The threshold that maximised balanced accuracy in CV.
print("test balanced accuracy:", round(balanced_accuracy_score(y_te, tuned.predict(X_te)), 3))  #> Evaluate once on the test set.
```

:::research Choose the threshold on training/validation data
Picking the threshold that looks best on the test set is another form of leakage. Tune it with cross-validation on the training data (as above), decide it from domain costs before looking at test results, or report threshold-free metrics (ROC AUC, PR AUC) alongside.
:::

## Comparing classifiers {I}

```py title="Eight classifiers under the same protocol"
from sklearn.neighbors import KNeighborsClassifier  #> Import `KNeighborsClassifier` from scikit-learn's `neighbors` module.
from sklearn.svm import SVC  #> Import `SVC` from scikit-learn's `svm` module.
from sklearn.tree import DecisionTreeClassifier  #> Import `DecisionTreeClassifier` from scikit-learn's `tree` module.
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier  #> Import `RandomForestClassifier`, `HistGradientBoostingClassifier` and `ExtraTreesClassifier` from scikit-learn's `ensemble` module.
from sklearn.naive_bayes import GaussianNB  #> Import `GaussianNB` from scikit-learn's `naive_bayes` module.
from sklearn.dummy import DummyClassifier  #> Import `DummyClassifier` from scikit-learn's `dummy` module.
from sklearn.calibration import CalibratedClassifierCV  #> Import `CalibratedClassifierCV` from scikit-learn's `calibration` module.

candidates = {  #> Name -> model.
    "Dummy (prior)": DummyClassifier(strategy="prior"),  #> Baseline: predicts class frequencies.
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),  #> Scaled logistic regression.
    "kNN (k=7)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),  #> Scaled kNN with 7 neighbours.
    "SVC (RBF)": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(), ensemble=False)),   #> SVC has no predict_proba of its own; calibration adds it. (`SVC(probability=True)` is deprecated since scikit-learn 1.9.)
    "GaussianNB": GaussianNB(),  #> Naive Bayes.
    "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=0),  #> A shallow tree.
    "RandomForest": RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),  #> 400-tree random forest using all CPU cores.
    "ExtraTrees": ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=-1),  #> Extremely randomised trees: like a forest with random split points.
    "HistGradBoost": HistGradientBoostingClassifier(random_state=0),  #> Histogram gradient boosting.
}  #> Close the dictionary opened above.
rows = []  #> One dict of results per model.
for name, model in candidates.items():  #> Loop over `candidates.items()`, naming each item `name, model`.
    model.fit(X_tr, y_tr)  #> Train.
    prob = model.predict_proba(X_te)[:, 1]  #> Probability of class 1.
    pred = model.predict(X_te)  #> Hard labels.
    rows.append({"model": name, "accuracy": model.score(X_te, y_te), "recall": recall_score(y_te, pred),  #> Record the metrics...
                 "precision": precision_score(y_te, pred, zero_division=0), "ROC_AUC": roc_auc_score(y_te, prob)})   #> zero_division=0 avoids a warning for the dummy, which never predicts positive.
print(pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).round(3).to_string(index=False))  #> Table sorted by ROC AUC.
```

:::warn One split is a noisy judge
With 143 test samples, one misclassification changes accuracy by 0.7 percentage points. Differences in the third decimal place are noise. Use cross-validation ([[ch:sklearn-model-selection]]) and confidence intervals ([[ch:research-choosing]]) before declaring a winner.
:::

| Model | Strengths | Weaknesses | Scale features? |
|---|---|---|---|
| Logistic regression | fast, interpretable coefficients, well-calibrated probabilities | linear boundary unless you engineer features | yes |
| k-nearest neighbours | no training, flexible boundary | slow on big data, struggles in high dimensions | yes |
| SVM (RBF kernel) | strong on small-to-medium data, flexible | slow beyond ~10⁵ samples, needs C/gamma tuning | yes |
| Naive Bayes | extremely fast, good for text | strong independence assumption, poor probabilities | no |
| Decision tree | readable rules | unstable, overfits | no |
| Random forest / Extra trees | robust defaults, little tuning, feature importance | larger models, less interpretable | no |
| Gradient boosting | often best on tabular data, handles NaN (Hist) | more hyperparameters, can overfit | no |
| Neural network (Part VI) | images, text, signals, huge data | data-hungry, many design choices | yes |

## Class imbalance {I}

:::def
**Class imbalance**: when one class is much rarer than the others, which makes accuracy misleading and models biased toward the majority class.
:::


When one class is rare (fraud, rare disease), a model can score high accuracy by ignoring it.

```py title="An imbalanced problem: accuracy lies, other metrics don't"
from sklearn.datasets import make_classification  #> Import `make_classification` from scikit-learn's `datasets` module.

Xi, yi = make_classification(n_samples=4000, n_features=20, n_informative=5,  #> Synthetic data: 4000 samples, 5 informative features...
                             weights=[0.97, 0.03], flip_y=0.01, random_state=1)   #> Only ~3% positives.
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, stratify=yi, test_size=0.3, random_state=1)  #> Stratified split keeps ~3% positives in both parts.
print("positive share in test:", round(yi_te.mean(), 3))  #> Check the imbalance.
for name, m in [("dummy", DummyClassifier(strategy="most_frequent")),  #> Loop over three models: an always-negative baseline...
                ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),  #> ...plain logistic regression...
                ("logreg balanced", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")))]:   #> class_weight="balanced" up-weights errors on the rare class.
    m.fit(Xi_tr, yi_tr)  #> Train.
    p = m.predict(Xi_te)  #> Predict.
    print(f"{name:<16} acc={m.score(Xi_te, yi_te):.3f}  recall={recall_score(yi_te, p):.3f}  "  #> Accuracy and recall...
          f"precision={precision_score(yi_te, p, zero_division=0):.3f}  bal_acc={balanced_accuracy_score(yi_te, p):.3f}")  #> ...precision and balanced accuracy.
```

:::tip Imbalance toolbox
1. Use the right metrics: recall, precision, F1, PR AUC, balanced accuracy, MCC. 2. Stratify every split (`stratify=y`, `StratifiedKFold`). 3. Use `class_weight="balanced"` where available. 4. Tune the decision threshold. 5. Resampling (SMOTE, undersampling) is available in the `imbalanced-learn` package; resample **only inside training folds**, using its `imblearn.pipeline.Pipeline`.
:::

## Probability calibration {A}

:::def
**Calibration**: how well predicted probabilities match observed frequencies: among cases given 80%, about 80% should be positive.
:::


A model is **calibrated** if, among samples given probability 0.8, about 80% are truly positive. This matters whenever probabilities feed decisions or risk scores.

```py title="Reliability diagram" caption="Points on the diagonal mean perfectly calibrated probabilities. Naive Bayes is typically over-confident (an S-shaped curve), logistic regression is usually well calibrated."
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV  #> Import `CalibrationDisplay` and `CalibratedClassifierCV` from scikit-learn's `calibration` module.
from sklearn.metrics import brier_score_loss  #> Import `brier_score_loss` from scikit-learn's `metrics` module.

Xc, yc = make_classification(n_samples=6000, n_features=20, n_informative=6, random_state=0)  #> A balanced synthetic dataset.
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.5, random_state=0)  #> Half for training, half for evaluating calibration.
fig, ax = plt.subplots(figsize=(6.4, 5))  #> One panel.
for name, m in [("LogReg", LogisticRegression(max_iter=1000)), ("GaussianNB", GaussianNB()),  #> Compare three models...
                ("GaussianNB + isotonic", CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5))]:   #> Re-map scores to calibrated probabilities using CV.
    m.fit(Xc_tr, yc_tr)  #> Train.
    prob = m.predict_proba(Xc_te)[:, 1]  #> Predicted probabilities on held-out data.
    CalibrationDisplay.from_predictions(yc_te, prob, n_bins=10, name=f"{name} (Brier {brier_score_loss(yc_te, prob):.3f})", ax=ax)   #> Brier score: mean squared error of probabilities (lower is better).
ax.set_title("Calibration curves")  #> Title.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Multiclass problems {I}

```py title="Three wine cultivars"
from sklearn.datasets import load_wine  #> Import `load_wine` from scikit-learn's `datasets` module.
Xw, yw = load_wine(return_X_y=True)  #> 178 wines, 13 features, 3 classes.
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(Xw, yw, stratify=yw, random_state=0)  #> Stratified split.
mc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xw_tr, yw_tr)   #> Multiclass is handled automatically (multinomial).
pw = mc.predict(Xw_te)  #> Predicted classes.
print(confusion_matrix(yw_te, pw))                                                  #> 3x3: rows truth, columns prediction.
print("macro F1:", round(f1_score(yw_te, pw, average="macro"), 3),                  #> macro = unweighted mean over classes (treats classes equally).
      "| weighted F1:", round(f1_score(yw_te, pw, average="weighted"), 3))         #> weighted = mean weighted by class frequency.
print("one-vs-rest ROC AUC:", round(roc_auc_score(yw_te, mc.predict_proba(Xw_te), multi_class="ovr"), 4))  #> Average AUC of each class against the rest.
```

:::bug `ValueError: Target is multiclass but average='binary'`
`precision_score`, `recall_score` and `f1_score` default to `average="binary"`. For three or more classes pass `average="macro"`, `"weighted"` or `"micro"`. For string labels in a binary problem, pass `pos_label="yes"` (or whichever is positive).
:::

:::ex Try it yourself
1. Find the threshold that gives at least 98% recall on the training data via cross-validated probabilities (`cross_val_predict(..., method="predict_proba")`) and report test precision at that threshold.
2. Repeat the imbalance experiment with `HistGradientBoostingClassifier(class_weight="balanced")`.
3. Plot ROC curves of three models on one axes by passing the same `ax` to each `RocCurveDisplay`.
:::
