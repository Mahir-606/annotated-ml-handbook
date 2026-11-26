@part 7

# The ML research protocol: a complete worked study

This chapter strings everything together into a protocol you can follow for a thesis chapter or a paper, and runs it end to end on a real dataset. Each step says **what to do**, **why**, and **what to report**. Copy the structure, swap in your data.

**The study.** Can routinely measured cell-nucleus features from fine-needle aspirate images classify breast masses as malignant or benign, and which features drive the prediction? (Wisconsin Diagnostic Breast Cancer data, 569 cases, bundled with scikit-learn.)

## The protocol at a glance {B}

:::def
**Research protocol**: a written, step-by-step plan for a study, fixed before analysis, covering the question, data, methods and how results will be judged.

**Primary metric**: the single metric chosen in advance to judge success. Others are reported as secondary.
:::


| Step | Do | Report |
|---|---|---|
| 1. Question | Define the target, the unit of analysis, the population, the primary metric, and what "useful" means | Objective, outcome definition, primary metric |
| 2. Data | Document the source, variables, missingness and exclusions | Data source, sample size, a table of characteristics |
| 3. Explore | Distributions, class balance, correlations, oddities, using training data for decisions | Key descriptive statistics, any data issues found |
| 4. Split | Lock a test set away (stratified, grouped or temporal as appropriate) | Split method, sizes, seed |
| 5. Baselines | Dummy model and a simple, interpretable model | Baseline performance |
| 6. Compare | Candidate pipelines under repeated cross-validation | Mean ± SD for every candidate, not just the winner |
| 7. Tune | Hyperparameter search inside CV; nested CV if there is no test set | Search space, method, chosen values |
| 8. Test | Evaluate the final model on the test set once, with confidence intervals | Test metrics with 95% CIs, confusion matrix, curves |
| 9. Interpret | Permutation importance, PDP/SHAP, error analysis, checked for stability | Top features, their direction, caveats |
| 10. Robustness | Seeds, calibration, learning curve, subgroups, sensitivity analyses | Variability and limitations |
| 11. Reproduce | Save code, environment, seeds, model; write it up | Software versions, code availability |

## Step 1: define the question before touching the data {B}

Write these down first. They stop you from moving the goalposts after seeing results.

- **Target**: malignant (1) vs benign (0), from histopathology.
- **Unit of analysis**: one mass per patient (so a random split is valid; with repeated measures you would group by patient).
- **Primary metric**: ROC AUC (threshold-free ranking quality). **Secondary**: sensitivity (recall for malignant) at a threshold chosen on training data, specificity, balanced accuracy, Brier score.
- **What counts as useful**: high sensitivity, because a missed cancer is costlier than a false alarm.

:::research Pre-specify
Writing the analysis plan (metric, split, candidate models, tuning budget) before modelling, ideally with a timestamp (OSF, a dated Git commit), is the ML equivalent of pre-registration. It is your best defence against accidental p-hacking ("trying things until something works").
:::

## Step 2: load and document the data {B}

```py title="Data and a characteristics table"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.
import seaborn as sns  #> Import Seaborn (statistical plots built on Matplotlib) under the short name `sns`.
from sklearn.datasets import load_breast_cancer  #> Import `load_breast_cancer` from scikit-learn's `datasets` module.

RANDOM_STATE = 2026                                       #> ONE seed constant, used everywhere, reported in the paper.
raw = load_breast_cancer(as_frame=True)  #> Load the data as pandas objects.
df = raw.frame.copy()  #> One DataFrame with features and target together.
df["malignant"] = 1 - df.pop("target")                    #> pop removes "target"; we recode so 1 = malignant (the positive class).
print(df.shape)  #> Display the value of `df.shape`.
print("missing values:", int(df.isna().sum().sum()), "| duplicate rows:", int(df.duplicated().sum()))  #> Data-quality checks: both are 0 here.
print(df["malignant"].value_counts().rename({0: "benign", 1: "malignant"}))  #> Class counts with readable labels.
table1 = (df.groupby("malignant")[["mean radius", "mean texture", "mean concave points", "worst area"]]  #> Group by class and pick four key variables...
            .agg(["mean", "std"]).round(2).T)              #> A compact "Table 1": mean (SD) of key variables by class.
table1.columns = ["benign", "malignant"]  #> Readable column names.
print(table1)  #> Display the value of `table1`.
```

## Step 3: exploratory analysis {B}

```py title="Distributions and correlations" caption="Left: several features separate the classes strongly. Right: many features are highly correlated (radius, perimeter and area measure nearly the same thing), which matters for interpretation and for linear models."
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))  #> Two panels.
long = df.melt(id_vars="malignant", value_vars=["mean radius", "mean texture", "mean concave points", "mean smoothness"])  #> Reshape four features to long format for Seaborn.
long["z"] = long.groupby("variable")["value"].transform(lambda s: (s - s.mean()) / s.std())   #> Standardise so features share one axis.
sns.boxplot(data=long, x="variable", y="z", hue="malignant", ax=axes[0], fliersize=2)  #> One pair of boxes (benign vs malignant) per feature.
axes[0].set(xlabel="", ylabel="standardised value", title="Feature distributions by class")  #> Labels.
axes[0].tick_params(axis="x", rotation=15)  #> Tilt the feature names.
corr = df.drop(columns="malignant").iloc[:, :10].corr()  #> Correlations among the first 10 features.
sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[1], cbar_kws={"shrink": 0.8},  #> Heatmap on a fixed -1..1 scale...
            xticklabels=[c.replace("mean ", "") for c in corr.columns], yticklabels=[c.replace("mean ", "") for c in corr.columns])  #> ...with shorter tick labels.
axes[1].set_title("Correlation of the 10 'mean' features"); axes[1].grid(False)  #> Title, no grid.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

:::warn Explore with care
Looking at the *whole* dataset to understand variables and spot errors is fine. Using it to **choose** features or thresholds is not: those choices must be made on training data. A safe habit is to split first and run detailed exploration on the training set only.
:::

## Step 4: split and lock away the test set {B}

:::def
**Hold-out set**: data set aside from all model development and used only for final evaluation (the test set).
:::


```py title="Stratified hold-out split"
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.
X = df.drop(columns="malignant")  #> Features.
y = df["malignant"]  #> Target.
X_train, X_test, y_train, y_test = train_test_split(  #> Split into training and test sets...
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)   #> 25% test, class balance preserved.
print(f"train n={len(y_train)} ({y_train.mean():.1%} malignant) | test n={len(y_test)} ({y_test.mean():.1%} malignant)")  #> Sizes and class balance of each part.
```

## Steps 5–6: baselines and candidate models under repeated CV {I}

```py title="Compare candidates with identical, repeated CV"
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate  #> Import `RepeatedStratifiedKFold` and `cross_validate` from scikit-learn's `model_selection` module.
from sklearn.pipeline import make_pipeline  #> Import `make_pipeline` from scikit-learn's `pipeline` module.
from sklearn.preprocessing import StandardScaler  #> Import `StandardScaler` from scikit-learn's `preprocessing` module.
from sklearn.dummy import DummyClassifier  #> Import `DummyClassifier` from scikit-learn's `dummy` module.
from sklearn.linear_model import LogisticRegression  #> Import `LogisticRegression` from scikit-learn's `linear_model` module.
from sklearn.neighbors import KNeighborsClassifier  #> Import `KNeighborsClassifier` from scikit-learn's `neighbors` module.
from sklearn.svm import SVC  #> Import `SVC` from scikit-learn's `svm` module.
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier  #> Import `RandomForestClassifier` and `HistGradientBoostingClassifier` from scikit-learn's `ensemble` module.

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)   #> 25 train/validation splits, IDENTICAL for every model.
candidates = {  #> The candidate models:
    "Dummy": DummyClassifier(strategy="prior"),  #> Baseline.
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),  #> Scaled logistic regression.
    "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9)),  #> Scaled kNN, 9 neighbours.
    "SVM-RBF": make_pipeline(StandardScaler(), SVC()),  #> Scaled RBF support vector machine.
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),  #> Random forest.
    "HistGB": HistGradientBoostingClassifier(random_state=RANDOM_STATE),  #> Gradient boosting.
}  #> Close the dictionary opened above.
scoring = {"auc": "roc_auc", "bal_acc": "balanced_accuracy", "recall": "recall"}  #> Short name -> scikit-learn scorer.
cv_scores, rows = {}, []  #> Per-model AUC arrays, and summary rows.
for name, est in candidates.items():  #> Loop over `candidates.items()`, naming each item `name, est`.
    r = cross_validate(est, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)  #> Cross-validate this candidate on the 25 splits.
    cv_scores[name] = r["test_auc"]                                  #> Keep all 25 AUCs for plotting and statistical comparison.
    rows.append({"model": name, **{f"{k} mean": r[f"test_{k}"].mean() for k in scoring},  #> Mean of each metric...
                 "auc sd": r["test_auc"].std()})  #> ...plus the spread of AUC.
summary = pd.DataFrame(rows).sort_values("auc mean", ascending=False).round(4)  #> Summary table, best AUC first.
print(summary.to_string(index=False))  #> Print it.
```

```py title="Show the spread, not just the mean" caption="ROC AUC over 25 CV splits per model. The overlap between the top models is large: on this evidence they are practically equivalent, and the simplest one is a defensible choice."
fig, ax = plt.subplots(figsize=(8, 3.8))  #> One panel.
order = summary["model"].tolist()[::-1]  #> Worst first, so the best is drawn at the top.
order = [m for m in order if m != "Dummy"]                          #> The dummy (AUC 0.5) would squash the axis.
ax.boxplot([cv_scores[m] for m in order], orientation="horizontal", tick_labels=order, widths=0.55)  #> One horizontal box per model.
for i, m in enumerate(order, start=1):  #> Loop over `enumerate(order, start=1)`, naming each item `i, m`.
    ax.scatter(cv_scores[m], np.random.default_rng(i).normal(i, 0.06, len(cv_scores[m])), s=9, alpha=0.6, color="#d99a00")   #> Jittered raw points.
ax.set(xlabel="ROC AUC (25 CV splits)", title="Cross-validated comparison (training data only)")  #> Labels.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Step 7: tune the chosen model {I}

Logistic regression is near the top, fast, and interpretable, so it is our primary model. We tune its regularisation strength `C` (and penalty type) inside CV.

```py title="Grid search on training data"
from sklearn.model_selection import GridSearchCV, StratifiedKFold  #> Import `GridSearchCV` and `StratifiedKFold` from scikit-learn's `model_selection` module.
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear"))   #> liblinear handles both L1 and L2 penalties and is fast on small data.
grid = {"logisticregression__C": np.logspace(-3, 2, 11),  #> 11 values of C from 0.001 to 100...
        "logisticregression__l1_ratio": [0.0, 1.0]}                 #> l1_ratio=0 -> L2 (ridge-like), 1 -> L1 (lasso-like, sparse).
search = GridSearchCV(pipe, grid, scoring="roc_auc",  #> Grid search maximising ROC AUC...
                      cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1)  #> ...with stratified 5-fold CV.
search.fit(X_train, y_train)  #> Run the search on training data only.
best_C = search.best_params_["logisticregression__C"]  #> The chosen C.
print("best:", {"C": round(float(best_C), 4), "l1_ratio": search.best_params_["logisticregression__l1_ratio"]},  #> Report the chosen values...
      "| CV AUC:", round(search.best_score_, 4))  #> ...and their CV score.
final_model = search.best_estimator_                                #> Already refitted on the full training set.
```

### Choose the operating threshold on training data {I}

```py title="A threshold for ≥ 95% sensitivity, from out-of-fold predictions"
from sklearn.model_selection import cross_val_predict  #> Import `cross_val_predict` from scikit-learn's `model_selection` module.
oof = cross_val_predict(final_model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),  #> Out-of-fold probabilities for the training cases...
                        method="predict_proba")[:, 1]               #> Each training case predicted by a model that did not see it.
thresholds = np.linspace(0.01, 0.99, 99)  #> Candidate thresholds 0.01, 0.02, ..., 0.99.
sens = np.array([((oof >= t) & (y_train == 1)).sum() / (y_train == 1).sum() for t in thresholds])  #> Sensitivity at each threshold.
threshold = float(thresholds[sens >= 0.95].max())                   #> The highest threshold that still gives >= 95% sensitivity.
print(f"chosen threshold = {threshold:.2f}")  #> The threshold we will use on the test set.
```

## Step 8: the one-time test evaluation {I}

```py title="Test metrics with bootstrap 95% confidence intervals"
from sklearn.metrics import roc_auc_score, recall_score, balanced_accuracy_score, brier_score_loss, confusion_matrix  #> Import `roc_auc_score`, `recall_score`, `balanced_accuracy_score`, `brier_score_loss` and `confusion_matrix` from scikit-learn's `metrics` module.

p_test = final_model.predict_proba(X_test)[:, 1]  #> Test-set probabilities (first and only look).
yhat = (p_test >= threshold).astype(int)                            #> Apply the PRE-CHOSEN threshold.
tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()  #> The four outcome counts.
metrics = {  #> Each metric as a function of (truth, probability, label):
    "ROC AUC": lambda yt, p, yh: roc_auc_score(yt, p),  #> Uses probabilities.
    "Sensitivity": lambda yt, p, yh: recall_score(yt, yh),  #> Uses labels: recall of malignant cases.
    "Specificity": lambda yt, p, yh: recall_score(yt, yh, pos_label=0),   #> Specificity = recall of the negative class.
    "Balanced acc.": lambda yt, p, yh: balanced_accuracy_score(yt, yh),  #> Mean of sensitivity and specificity.
    "Brier score": lambda yt, p, yh: brier_score_loss(yt, p),  #> Mean squared error of the probabilities.
}  #> Close the dictionary opened above.
rng = np.random.default_rng(RANDOM_STATE)  #> Seeded generator for the bootstrap.
yt_arr = y_test.to_numpy()  #> Plain array for fast indexing.
boot = {k: [] for k in metrics}  #> One list of bootstrap values per metric.
for _ in range(2000):  #> Loop over `range(2000)`, naming each item `_`.
    idx = rng.integers(0, len(yt_arr), len(yt_arr))                 #> Resample test cases with replacement.
    if yt_arr[idx].min() == yt_arr[idx].max():  #> Run the indented block only if `yt_arr[idx].min() == yt_arr[idx].max()` is true.
        continue                                                     #> Skip the (rare) resamples containing a single class.
    for k, f in metrics.items():  #> Loop over `metrics.items()`, naming each item `k, f`.
        boot[k].append(f(yt_arr[idx], p_test[idx], yhat[idx]))  #> Compute the metric on this resample.
results = pd.DataFrame({k: {"estimate": f(yt_arr, p_test, yhat),  #> A table: point estimate...
                            "95% CI low": np.percentile(boot[k], 2.5),  #> ...2.5th percentile...
                            "95% CI high": np.percentile(boot[k], 97.5)} for k, f in metrics.items()}).T.round(3)  #> ...97.5th percentile; .T puts metrics in rows.
print(results)  #> Display the value of `results`.
print(f"confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")  #> Raw counts for the paper.
```

```py title="Final figures for the paper" caption="Test-set performance of the tuned logistic regression: ROC curve, and calibration (reliability) curve. Both computed once, on data never used for any decision."
from sklearn.metrics import RocCurveDisplay  #> Import `RocCurveDisplay` from scikit-learn's `metrics` module.
from sklearn.calibration import CalibrationDisplay  #> Import `CalibrationDisplay` from scikit-learn's `calibration` module.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))  #> Two panels.
RocCurveDisplay.from_predictions(y_test, p_test, ax=axes[0], name=f"LogReg (AUC {roc_auc_score(y_test, p_test):.3f})",  #> Test ROC curve, with the AUC in the legend...
                                 plot_chance_level=True)  #> ...and the chance diagonal.
axes[0].scatter(fp / (fp + tn), tp / (tp + fn), s=60, color="#d42a2a", zorder=5, label=f"threshold {threshold:.2f}")   #> Mark the operating point.
axes[0].legend(loc="lower right"); axes[0].set_title("ROC curve (test set)")  #> Legend and title.
CalibrationDisplay.from_predictions(y_test, p_test, n_bins=8, ax=axes[1], name="LogReg")  #> Reliability curve with 8 bins.
axes[1].set_title("Calibration (test set)")  #> Title.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Step 9: interpretation, with stability {I}

```py title="Permutation importance on the test set, and coefficient stability across CV folds"
from sklearn.inspection import permutation_importance  #> Import `permutation_importance` from scikit-learn's `inspection` module.
pi = permutation_importance(final_model, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1)  #> Shuffle each feature 30 times on the test set.
imp = pd.DataFrame({"drop in AUC": pi.importances_mean, "sd": pi.importances_std}, index=X.columns)  #> Mean and SD of the AUC drop per feature.
print(imp.sort_values("drop in AUC", ascending=False).head(6).round(4))  #> Top 6 features.

cvr = cross_validate(final_model, X_train, y_train, cv=cv, return_estimator=True, n_jobs=-1)   #> Refit on 25 resamples...
coefs = pd.DataFrame([e.named_steps["logisticregression"].coef_[0] for e in cvr["estimator"]], columns=X.columns)  #> One row of coefficients per resample.
stab = pd.DataFrame({"median coef": coefs.median(), "% non-zero": (coefs != 0).mean() * 100,  #> Median size, how often kept...
                     "% positive": (coefs > 0).mean() * 100})        #> ...and check how consistent each coefficient's size and sign are.
print(stab.reindex(coefs.median().abs().sort_values(ascending=False).index).head(8).round(2))  #> The 8 features with the largest median coefficients.
```

:::note Reading these results
The L1 penalty keeps *worst radius* in every resample, while near-duplicates such as *worst area* are included only some of the time (see "% non-zero"). Shuffling worst radius therefore hurts a lot, while shuffling a duplicate changes little. That does not make area irrelevant: the model only needed one member of that correlated group. Interpret importance at the level of **groups of correlated features**. In the stability table, features with the same sign in all 25 resamples are robust findings; features that appear in only some resamples (low "% non-zero") are not.
:::

:::warn The test set can disagree with your plan
The threshold was chosen to give at least 95% sensitivity on out-of-fold *training* predictions. On the test set, sensitivity came out lower (see the table above). The confidence interval is wide because there are only about 53 malignant test cases. This is exactly why you report CIs, and why you never quietly move the threshold after seeing test results. Report the shortfall honestly; if the target is critical, collect more data or validate the threshold on an independent cohort.
:::

## Step 10: robustness checks {I}

```py title="Seed sensitivity and a learning curve"
from sklearn.model_selection import learning_curve  #> Import `learning_curve` from scikit-learn's `model_selection` module.
seed_aucs = []  #> Test AUC for each split seed.
for seed in range(10):                                               #> Repeat the whole split-tune-test pipeline with different seeds.
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)  #> A new split.
    m = GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1).fit(Xa, ya)  #> Re-tune on the new training part.
    seed_aucs.append(roc_auc_score(yb, m.predict_proba(Xb)[:, 1]))  #> Score on the new test part.
print(f"test AUC over 10 different splits: mean={np.mean(seed_aucs):.4f}, SD={np.std(seed_aucs):.4f}, "  #> Mean and SD across splits...
      f"range={min(seed_aucs):.4f}–{max(seed_aucs):.4f}")  #> ...and the full range.
sizes, tr_s, va_s = learning_curve(final_model, X_train, y_train, cv=5, scoring="roc_auc",  #> Training sizes and train/validation scores...
                                   train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1)  #> ...at 20%, 40%, ..., 100% of the training data.
print("learning curve (validation AUC):", dict(zip(sizes.tolist(), va_s.mean(axis=1).round(4).tolist())))  #> Training size -> mean validation AUC.
```

:::research A results paragraph you could adapt
"Six candidate pipelines were compared using 5 × 5 repeated stratified cross-validation on the training set (n = 426). L1-regularised logistic regression performed as well as or better than the non-linear models (CV ROC AUC 0.998; Table 2) and was selected for interpretability. After tuning by grid search, the final model achieved a ROC AUC of 0.977 (95% bootstrap CI 0.944–0.999) on the held-out test set (n = 143). At a threshold pre-specified on training data to target ≥ 95% sensitivity, test sensitivity was 0.906 (0.816–0.980) and specificity 0.989 (0.965–1.000), so the sensitivity target was not met on the test set. Across ten different random splits the test AUC ranged from 0.974 to 0.999." Every number comes from the outputs above. Copy them exactly; never round in your favour, and report targets that were missed.
:::

## Step 11: make it reproducible {B}

```py title="Save the model, the configuration and the environment"
import json, joblib, sys, sklearn, platform  #> Import the standard-library `json` module and joblib, used to save and load fitted models and the `sys` module (interpreter information) and scikit-learn and the `platform` module (system and version info).
joblib.dump(final_model, "final_model.joblib")  #> Save the final fitted pipeline.
run_record = {  #> Everything needed to reproduce and report the run:
    "random_state": RANDOM_STATE, "threshold": threshold,  #> The seed and the operating threshold.
    "best_params": {k: float(v) for k, v in search.best_params_.items()},  #> Chosen hyperparameters.
    "test_metrics": results["estimate"].to_dict(),  #> Test metrics.
    "versions": {"python": platform.python_version(), "numpy": np.__version__,  #> Software versions...
                 "pandas": pd.__version__, "scikit-learn": sklearn.__version__},  #> ...continued.
}  #> Close the dictionary opened above.
with open("run_record.json", "w") as f:  #> Open the output file.
    json.dump(run_record, f, indent=2)  #> Write it as readable JSON.
print(json.dumps(run_record, indent=2)[:600])  #> Show the start of the record.
```

:::ex Try it yourself
1. Re-run the protocol with `HistGradientBoostingClassifier` as the primary model. Is the conclusion different?
2. Replace the hold-out test set by nested cross-validation ([[ch:sklearn-model-selection]]) and compare the estimates.
3. Write the Methods paragraph for this study using the checklist in [[ch:research-reporting]].
:::
