# Choosing models and metrics, and comparing them rigorously

Which algorithm should you try? Which metric belongs in the paper? Is model A *really* better than model B, or is the difference noise? This chapter gives practical decision guides and the statistical tools to answer those questions defensibly.

## A model-selection map {B}

```py figonly caption="A practical starting map. Always begin with a baseline and a simple model, then move right only if the data and the question justify it."
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(12, 6.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
def box(x, y, w, h, text, color, tc="white", fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", color=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold", linespacing=1.3)
def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))
    if label: ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2 + 0.08, label, fontsize=8.5, color="#333")
box(0.2, 2.9, 1.9, 0.9, "What is the\ndata?", "#14161c")
box(2.9, 5.2, 2.2, 0.9, "Tables\n(rows x columns)", "#0f9b62")
box(2.9, 2.9, 2.2, 0.9, "Images / audio /\ntext / signals", "#ff6a00")
box(2.9, 0.6, 2.2, 0.9, "No target\n(explore / group)", "#7b3ff2")
arrow(2.1, 3.6, 2.9, 5.6); arrow(2.1, 3.35, 2.9, 3.35); arrow(2.1, 3.1, 2.9, 1.1)
box(5.8, 5.55, 2.6, 0.75, "Predict a number", "#e9f7ef", "#0b5d3b"); box(5.8, 4.6, 2.6, 0.75, "Predict a class", "#e9f7ef", "#0b5d3b")
arrow(5.1, 5.75, 5.8, 5.9); arrow(5.1, 5.5, 5.8, 4.95)
box(9.0, 5.55, 2.85, 0.75, "Ridge/Lasso -> RF -> HistGB", "#0f9b62", fs=9)
box(9.0, 4.6, 2.85, 0.75, "LogReg -> RF -> HistGB (SVM if small)", "#0f9b62", fs=8.6)
arrow(8.4, 5.92, 9.0, 5.92); arrow(8.4, 4.97, 9.0, 4.97)
box(5.8, 3.4, 2.6, 0.75, "Little labelled data", "#fff1e6", "#8a3a00"); box(5.8, 2.45, 2.6, 0.75, "Lots of labelled data", "#fff1e6", "#8a3a00")
arrow(5.1, 3.5, 5.8, 3.75); arrow(5.1, 3.2, 5.8, 2.8)
box(9.0, 3.4, 2.85, 0.75, "Transfer learning / pretrained", "#ff6a00", fs=9); box(9.0, 2.45, 2.85, 0.75, "CNN / Transformer / LSTM", "#ff6a00", fs=9)
arrow(8.4, 3.77, 9.0, 3.77); arrow(8.4, 2.82, 9.0, 2.82)
box(5.8, 1.15, 2.6, 0.75, "Find groups", "#f1ecff", "#4b23a8"); box(5.8, 0.2, 2.6, 0.75, "Reduce / visualise", "#f1ecff", "#4b23a8")
arrow(5.1, 1.2, 5.8, 1.5); arrow(5.1, 0.9, 5.8, 0.6)
box(9.0, 1.15, 2.85, 0.75, "KMeans / GMM / HDBSCAN", "#7b3ff2", fs=9); box(9.0, 0.2, 2.85, 0.75, "PCA -> t-SNE / UMAP", "#7b3ff2", fs=9)
arrow(8.4, 1.52, 9.0, 1.52); arrow(8.4, 0.57, 9.0, 0.57)
ax.text(0.2, 6.35, "Always start with: DummyClassifier/DummyRegressor + one simple, interpretable model", fontsize=10, color="#d42a2a", fontweight="bold")
plt.show()
```

### Rules of thumb {B}

- **Small tabular data (hundreds of rows):** regularised linear/logistic regression, SVM, random forest. Use repeated or nested CV; avoid deep learning.
- **Medium to large tabular data:** gradient boosting (`HistGradientBoosting*`, LightGBM, XGBoost, CatBoost) is usually the strongest; random forest is the robust runner-up.
- **Interpretability required:** logistic/linear regression with sensible features, shallow trees, or GAM-style models; explain complex models with permutation importance, PDP or SHAP, and say clearly that you are explaining *the model*.
- **Images, audio, text:** pretrained deep networks plus fine-tuning; train from scratch only with lots of data.
- **More features than samples (omics):** strong regularisation (Lasso, ElasticNet), feature selection *inside* CV, and nested CV.

## Choosing the metric {B}

| Situation | Primary metric | Also report |
|---|---|---|
| Balanced classes, all errors cost the same | accuracy, ROC AUC | confusion matrix |
| Imbalanced classes | PR AUC (average precision), balanced accuracy, MCC | per-class precision/recall, prevalence |
| Missing a positive is costly (screening) | sensitivity at a fixed specificity (or vice versa) | the threshold and how it was chosen |
| Probabilities used for decisions/risk | log loss, Brier score | calibration curve |
| Ranking / triage | ROC AUC, precision@k | lift or gain curves |
| Regression, errors in natural units | MAE, RMSE | R², residual plots |
| Regression with outliers | MAE, median absolute error | RMSE for comparison |
| Relative errors matter | MAPE (if no zeros), RMSLE | MAE |
| Clustering with ground truth | ARI, NMI | silhouette |
| Forecasting | MAE/RMSE per horizon, MASE vs naive | baseline errors |

:::research State the primary metric in advance
Choose one primary metric when you plan the study, and report secondary metrics as such. Picking the metric that makes your model look best after the fact is a subtle form of p-hacking.
:::

## Is model A really better than model B? {I}

:::def
**p-value**: the probability of seeing a difference at least as large as the one observed if there were really no difference. Small values (often < 0.05) count as evidence of a real difference.

**Statistical significance**: a result unlikely to be due to chance alone at a chosen threshold. It is not the same as practical importance.

**Paired comparison**: comparing two models on the same splits or the same test cases, so their differences can be analysed case by case.
:::


Two models' CV scores always differ a little. The question is whether the difference is larger than the noise. Three tools cover most cases.

### 1. Corrected resampled t-test on CV scores {A}

The CV folds share training data, so the scores are not independent, and a naive t-test is far too optimistic. The **Nadeau–Bengio correction** inflates the variance to account for this overlap.

```py title="Comparing two models on identical repeated-CV splits"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
from scipy import stats  #> Import `stats` from `scipy`.
from sklearn.datasets import load_breast_cancer  #> Import `load_breast_cancer` from scikit-learn's `datasets` module.
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score  #> Import `RepeatedStratifiedKFold` and `cross_val_score` from scikit-learn's `model_selection` module.
from sklearn.pipeline import make_pipeline  #> Import `make_pipeline` from scikit-learn's `pipeline` module.
from sklearn.preprocessing import StandardScaler  #> Import `StandardScaler` from scikit-learn's `preprocessing` module.
from sklearn.linear_model import LogisticRegression  #> Import `LogisticRegression` from scikit-learn's `linear_model` module.
from sklearn.ensemble import RandomForestClassifier  #> Import `RandomForestClassifier` from scikit-learn's `ensemble` module.

X, y = load_breast_cancer(return_X_y=True)  #> Features and labels.
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)       #> 100 splits, SAME for both models (paired comparison).
a = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)  #> 100 AUCs for logistic regression.
b = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)  #> 100 AUCs for the random forest, on the same splits.

def corrected_ttest(a, b, n_train, n_test):  #> Define a function `corrected_ttest` that takes `a`, `b`, `n_train` and `n_test`.
    d = a - b                                                   #> Paired differences, split by split.
    k = len(d)  #> Number of splits.
    var = d.var(ddof=1) * (1 / k + n_test / n_train)            #> Nadeau–Bengio corrected variance.
    t = d.mean() / np.sqrt(var)  #> The t statistic.
    p = 2 * stats.t.sf(abs(t), df=k - 1)                        #> Two-sided p-value.
    return d.mean(), t, p  #> Send `d.mean(), t, p` back to the caller.

n = len(y)  #> Total samples.
diff, t, p = corrected_ttest(a, b, n_train=int(n * 0.9), n_test=n - int(n * 0.9))  #> 10-fold CV: 90% train, 10% test in each split.
naive_p = stats.ttest_rel(a, b).pvalue                          #> The WRONG, uncorrected paired t-test, for comparison.
print(f"mean AUC: LogReg={a.mean():.4f}  RF={b.mean():.4f}  difference={diff:+.4f}")  #> Means and their difference (+ sign shown).
print(f"corrected t={t:.2f}, p={p:.4f}   |   naive paired t-test p={naive_p:.2e} (overconfident)")  #> Compare the honest and the naive p-values.
```

The corrected p-value is the honest one. The naive test treats 100 overlapping splits as 100 independent experiments and reports a far smaller p-value than the evidence supports.

### 2. McNemar's test on a single test set {I}

When both models are evaluated on the same test set, compare *which cases each gets wrong*.

```py title="Exact McNemar test"
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.
from sklearn.svm import SVC  #> Import `SVC` from scikit-learn's `svm` module.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)  #> One train/test split.
pa = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X_tr, y_tr).predict(X_te)  #> Model A's test predictions.
pb = make_pipeline(StandardScaler(), SVC()).fit(X_tr, y_tr).predict(X_te)  #> Model B's test predictions.
a_right_b_wrong = int(((pa == y_te) & (pb != y_te)).sum())     #> Discordant pairs are the only ones that carry information.
a_wrong_b_right = int(((pa != y_te) & (pb == y_te)).sum())  #> Cases only B gets right.
res = stats.binomtest(a_right_b_wrong, a_right_b_wrong + a_wrong_b_right, p=0.5)   #> Exact test: are discordant cases split 50/50?
print(f"LogReg acc={np.mean(pa == y_te):.3f}  SVC acc={np.mean(pb == y_te):.3f}")  #> Both accuracies.
print(f"discordant: LogReg-only correct={a_right_b_wrong}, SVC-only correct={a_wrong_b_right}, exact McNemar p={res.pvalue:.3f}")  #> The discordant counts and the p-value.
```

### 3. Paired bootstrap for any metric {I}

```py title="95% CI for the difference in test AUC"
from sklearn.metrics import roc_auc_score  #> Import `roc_auc_score` from scikit-learn's `metrics` module.
prob_a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]  #> Model A's probabilities.
prob_b = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]  #> Model B's probabilities.
rng = np.random.default_rng(0)  #> Seeded generator.
diffs = []  #> Bootstrap differences.
for _ in range(3000):  #> Loop over `range(3000)`, naming each item `_`.
    idx = rng.integers(0, len(y_te), len(y_te))                 #> Resample test cases; use the SAME cases for both models (paired).
    if y_te[idx].min() == y_te[idx].max():  #> Run the indented block only if `y_te[idx].min() == y_te[idx].max()` is true.
        continue  #> Skip to the next iteration of the loop.
    diffs.append(roc_auc_score(y_te[idx], prob_a[idx]) - roc_auc_score(y_te[idx], prob_b[idx]))  #> AUC difference on this resample.
lo, hi = np.percentile(diffs, [2.5, 97.5])  #> 95% interval.
print(f"AUC difference (LogReg - RF) = {roc_auc_score(y_te, prob_a) - roc_auc_score(y_te, prob_b):+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")  #> Observed difference and its interval.
print("CI includes 0 -> no evidence of a difference" if lo < 0 < hi else "CI excludes 0 -> evidence of a difference")  #> Interpret the interval.
```

:::warn Many models, many tests
Comparing ten models pairwise is 45 tests; some will be "significant" by chance. Correct for multiple comparisons, with `statsmodels.stats.multitest.multipletests(pvals, method="holm")` for Holm or `scipy.stats.false_discovery_control(pvals)` for the Benjamini–Hochberg false discovery rate, or compare every model against one pre-specified reference. And remember: a statistically significant AUC gain of 0.002 may be practically meaningless.
:::

## How much data do I need? {I}

There is no universal answer, but a **learning curve** on your own data tells you whether more data would help, and the width of your confidence intervals tells you how precisely you can estimate performance.

```py title="Precision of an accuracy estimate vs test-set size"
for n_test in [50, 100, 200, 500, 1000, 5000]:  #> Loop over `[50, 100, 200, 500, 1000, 5000]`, naming each item `n_test`.
    p_hat = 0.85                                                 #> Suppose the true accuracy is about 85%.
    half_width = 1.96 * np.sqrt(p_hat * (1 - p_hat) / n_test)    #> Normal-approximation 95% CI half-width for a proportion.
    print(f"n_test={n_test:>5}: accuracy 0.85 ± {half_width:.3f}")  #> The margin of error for this test-set size.
```

With 100 test cases your accuracy estimate is uncertain by about ±7 percentage points, which is larger than the gap between most competing models. For small datasets, prefer repeated or nested CV over a single small test set, and always show intervals.

:::research Report negative and null results
"Gradient boosting did not outperform logistic regression (difference in AUC −0.003, 95% CI −0.012 to 0.006)" is a valuable finding. It tells readers the simpler model suffices.
:::

:::ex Try it yourself
1. Use the corrected t-test to compare `SVC` and `LogisticRegression` on the wine dataset.
2. Run the paired bootstrap with `average_precision_score` instead of ROC AUC.
3. Simulate: draw two models with identical true accuracy and count how often the naive t-test on CV scores declares p < 0.05.
:::
