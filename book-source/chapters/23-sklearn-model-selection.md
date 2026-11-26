# Model selection: cross-validation and hyperparameter tuning

A single train/test split gives one noisy number. **Cross-validation** gives a more reliable estimate and lets you compare models and tune hyperparameters without touching the test set. This chapter covers the CV strategies you will need in research, including grouped and time-ordered data, then grid search, random search, learning curves and nested cross-validation.

## k-fold cross-validation {B}

:::def
**Cross-validation (CV)**: repeatedly splitting the training data into fitting and validation parts and averaging the scores, for a more reliable performance estimate.

**Fold**: one of the k parts the data is divided into in k-fold cross-validation.

**Stratification**: splitting data so that each part keeps the same class proportions as the whole.
:::


The training data is split into *k* folds. The model is trained *k* times, each time holding out a different fold for validation. You get *k* scores, so you can report a mean and a spread.

```py title="cross_val_score and cross_validate"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.
from sklearn.datasets import load_breast_cancer  #> Import `load_breast_cancer` from scikit-learn's `datasets` module.
from sklearn.model_selection import (train_test_split, cross_val_score, cross_validate,  #> Import several names from scikit-learn's `model_selection` module (the list continues on the next lines).
                                     KFold, StratifiedKFold, RepeatedStratifiedKFold)  #> (end of the import list)
from sklearn.pipeline import make_pipeline  #> Import `make_pipeline` from scikit-learn's `pipeline` module.
from sklearn.preprocessing import StandardScaler  #> Import `StandardScaler` from scikit-learn's `preprocessing` module.
from sklearn.linear_model import LogisticRegression  #> Import `LogisticRegression` from scikit-learn's `linear_model` module.

X, y = load_breast_cancer(return_X_y=True)  #> Features and labels as arrays.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)  #> Keep a test set aside.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))  #> The model to evaluate.

scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc")     #> 5 folds; one ROC AUC per fold. Pipeline = no leakage.
print(scores.round(4), f"mean={scores.mean():.4f} sd={scores.std():.4f}")  #> Five scores, their mean and spread.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)            #> Explicit splitter: stratified, shuffled, reproducible.
res = cross_validate(model, X_tr, y_tr, cv=cv,  #> cross_validate returns a dict of score arrays...
                     scoring=["accuracy", "roc_auc", "f1"],               #> Several metrics at once.
                     return_train_score=True)                             #> Also score the training folds to spot overfitting.
print(pd.DataFrame(res).mean().round(4))  #> Average of each column: fit time, score time, train and test scores.
```

:::tip Scoring names
`scoring=` accepts strings such as `"accuracy"`, `"balanced_accuracy"`, `"f1"`, `"f1_macro"`, `"roc_auc"`, `"average_precision"`, `"neg_log_loss"`, `"r2"`, `"neg_mean_absolute_error"`, `"neg_root_mean_squared_error"`. Error metrics are **negated** so that "higher is better" always holds; flip the sign when reporting. `sklearn.metrics.get_scorer_names()` lists them all.
:::

```py title="Repeated CV for more stable estimates"
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)   #> 5-fold CV repeated 10 times with different shuffles = 50 scores.
s = cross_val_score(model, X_tr, y_tr, cv=rcv, scoring="roc_auc")  #> 50 AUC scores.
print(f"{len(s)} scores: mean={s.mean():.4f}, 2.5–97.5 percentile = {np.percentile(s, 2.5):.4f}–{np.percentile(s, 97.5):.4f}")  #> Mean and the range of the middle 95%.
```

## Choosing the right splitter {I}

The CV scheme must mimic how the model will be used. Getting this wrong is the most common reason published ML results fail to replicate.

| Data situation | Splitter | Why |
|---|---|---|
| i.i.d. rows, regression | `KFold(shuffle=True)` | standard |
| i.i.d. rows, classification | `StratifiedKFold(shuffle=True)` | keeps class balance in each fold |
| several rows per subject/patient/site | `GroupKFold`, `StratifiedGroupKFold` | the same subject must never appear in both train and validation |
| time series / forecasting | `TimeSeriesSplit` | always train on the past, validate on the future |
| very small datasets | `LeaveOneOut`, `RepeatedStratifiedKFold` | use every sample, reduce variance |

```py title="Visualising splitters" caption="Each row is one CV iteration: blue = training rows, red = validation rows. GroupKFold keeps every group (colour band at bottom) entirely on one side; TimeSeriesSplit only ever validates on data that comes after the training window."
from sklearn.model_selection import GroupKFold, TimeSeriesSplit  #> Import `GroupKFold` and `TimeSeriesSplit` from scikit-learn's `model_selection` module.

n = 60  #> 60 rows for the illustration.
groups = np.repeat(np.arange(12), 5)                        #> 12 subjects with 5 rows each.
yy = np.r_[np.zeros(40), np.ones(20)]  #> Imbalanced labels: 40 zeros, 20 ones.
splitters = [("KFold (shuffled)", KFold(4, shuffle=True, random_state=0)),  #> Four splitters to compare, each with 4 folds...
             ("StratifiedKFold", StratifiedKFold(4, shuffle=True, random_state=0)),  #> ...stratified...
             ("GroupKFold", GroupKFold(4)),  #> ...grouped...
             ("TimeSeriesSplit", TimeSeriesSplit(4))]  #> ...and time-ordered.
fig, axes = plt.subplots(1, 4, figsize=(14, 2.9), sharey=True)  #> Four panels sharing the y-axis.
for ax, (name, sp) in zip(axes, splitters):  #> Loop over `zip(axes, splitters)`, naming each item `ax, (name, sp)`.
    g = groups if name == "GroupKFold" else None                     #> Only GroupKFold uses groups.
    for i, (tr, va) in enumerate(sp.split(np.zeros((n, 1)), yy, g)):  #> Loop over `enumerate(sp.split(np.zeros((n, 1)), yy, g))`, naming each item `i, (tr, va)`.
        row = np.full(n, np.nan); row[tr] = 0; row[va] = 1  #> 0 for training rows, 1 for validation rows, NaN for unused.
        ax.scatter(range(n), [i] * n, c=row, cmap="coolwarm", marker="s", s=22, vmin=0, vmax=1)  #> One row of squares per iteration (blue = train, red = validation).
    ax.scatter(range(n), [4.2] * n, c=groups, cmap="tab20", marker="s", s=22)  #> A band showing each sample's group.
    ax.set(title=name, xlabel="sample index", yticks=[0, 1, 2, 3, 4.2], yticklabels=["1", "2", "3", "4", "group"])  #> Label the rows.
    ax.grid(False)  #> Turn off the grid lines on this panel.
axes[0].set_ylabel("CV iteration")  #> y label on the first panel.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

### Grouped data: the leakage you don't see {I}

```py title="Repeated measurements: KFold is optimistic, GroupKFold is honest"
from sklearn.ensemble import RandomForestClassifier  #> Import `RandomForestClassifier` from scikit-learn's `ensemble` module.
rng = np.random.default_rng(0)  #> Seeded generator.
n_subj, per = 40, 10  #> 40 subjects, 10 recordings each.
subj = np.repeat(np.arange(n_subj), per)                         #> 40 subjects x 10 recordings each.
subj_effect = rng.normal(0, 1, (n_subj, 5))[subj]                 #> Each subject has a personal "fingerprint" in the features.
label = rng.integers(0, 2, n_subj)[subj]                          #> One label per subject, unrelated to the features: no real signal.
Xg = subj_effect + rng.normal(0, 0.3, (n_subj * per, 5))  #> Recordings = subject fingerprint + small noise.
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)  #> A flexible model that can memorise fingerprints.
naive = cross_val_score(rf, Xg, label, cv=KFold(5, shuffle=True, random_state=0)).mean()  #> Rows of the same subject land in both training and validation.
grouped = cross_val_score(rf, Xg, label, cv=GroupKFold(5), groups=subj).mean()   #> Pass `groups=` so whole subjects stay together.
print(f"KFold accuracy:      {naive:.2f}  <- the model recognises subjects it has already seen")  #> Inflated.
print(f"GroupKFold accuracy: {grouped:.2f}  <- honest: near chance")  #> Realistic.
```

:::research If you have multiple samples per subject, group by subject
Patients with several scans, participants with several trials, sites with many plots, molecules with several conformers: the unit of generalisation is the **subject**, so the split must be by subject. Report it explicitly in your Methods.
:::

### Time series {I}

```py title="Forward-chaining validation"
tscv = TimeSeriesSplit(n_splits=4, test_size=10)  #> 4 splits, each validating on the next 10 steps.
for i, (tr, va) in enumerate(tscv.split(np.arange(60))):  #> Loop over `enumerate(tscv.split(np.arange(60)))`, naming each item `i, (tr, va)`.
    print(f"fold {i}: train {tr.min()}–{tr.max()}  validate {va.min()}–{va.max()}")   #> Validation always lies after training.
```

## Hyperparameter tuning with GridSearchCV {I}

:::def
**Hyperparameter tuning**: searching for the hyperparameter values that give the best cross-validated performance.

**Grid search**: trying every combination of hyperparameter values from a predefined grid. **Random search** samples combinations at random instead.
:::


`GridSearchCV` tries every combination in a grid, scores each with cross-validation, and refits the best one on the whole training set.

```py title="Tuning an SVM inside a pipeline"
from sklearn.svm import SVC  #> Import `SVC` from scikit-learn's `svm` module.
from sklearn.model_selection import GridSearchCV  #> Import `GridSearchCV` from scikit-learn's `model_selection` module.

pipe = make_pipeline(StandardScaler(), SVC())  #> The pipeline to tune.
param_grid = {  #> The grid of values to try:
    "svc__C": [0.1, 1, 10, 100],                    #> step-name__parameter: reaches inside the pipeline.
    "svc__gamma": ["scale", 0.001, 0.01, 0.1],  #> Kernel width: how far each training point's influence reaches.
    "svc__kernel": ["rbf"],  #> Fixed: the RBF kernel.
}  #> Close the dictionary opened above.
search = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=0),  #> Score every combination with stratified 5-fold CV...
                      scoring="roc_auc", n_jobs=-1, return_train_score=True)   #> n_jobs=-1 uses all CPU cores.
search.fit(X_tr, y_tr)                                                          #> 16 combinations x 5 folds = 80 fits, then 1 refit.
print("best params:", search.best_params_)  #> The winning combination.
print("best CV ROC AUC:", round(search.best_score_, 4))  #> Its mean CV score.
print("test ROC AUC (once!):", round(search.score(X_te, y_te), 4))             #> search behaves like the refitted best model.
cvres = pd.DataFrame(search.cv_results_)                                        #> Every combination's mean/std scores.
print(cvres[["param_svc__C", "param_svc__gamma", "mean_test_score", "std_test_score", "rank_test_score"]]  #> The top 5 combinations...
      .sort_values("rank_test_score").head(5).round(4).to_string(index=False))  #> ...sorted by rank.
```

```py title="See the landscape" caption="Mean CV ROC AUC for each (C, gamma) pair. A broad bright plateau means results are robust to the exact choice; a single bright cell suggests you are fitting noise."
pivot = cvres.pivot_table(index="param_svc__gamma", columns="param_svc__C", values="mean_test_score")  #> Reshape into a gamma x C grid.
pivot.index = pivot.index.astype(str)  #> Text labels (mixing "scale" and numbers).
fig, ax = plt.subplots(figsize=(6, 3.8))  #> One panel.
im = ax.imshow(pivot.values, cmap="viridis")  #> Draw the grid as a heatmap.
ax.set(xticks=range(pivot.shape[1]), xticklabels=pivot.columns, yticks=range(pivot.shape[0]), yticklabels=pivot.index,  #> Tick labels from the grid values...
       xlabel="C", ylabel="gamma", title="Grid search: CV ROC AUC")  #> ...axis labels and title.
for (i, j), v in np.ndenumerate(pivot.values):  #> Loop over `np.ndenumerate(pivot.values)`, naming each item `(i, j), v`.
    ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v < pivot.values.max() - 0.01 else "black", fontsize=9)  #> Write each score in its cell, black on the brightest cells.
ax.grid(False)  #> Turn off the grid lines on this panel.
fig.colorbar(im)  #> Colour scale.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

### Searching over models and preprocessing too {A}

A grid can be a list of dicts, and a pipeline step can itself be a hyperparameter.

```py title="One search, several model families"
from sklearn.ensemble import RandomForestClassifier  #> Import `RandomForestClassifier` from scikit-learn's `ensemble` module.
from sklearn.pipeline import Pipeline  #> Import `Pipeline` from scikit-learn's `pipeline` module.
pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])  #> A pipeline whose "clf" step will be swapped.
grid = [  #> A list of grids, each searched separately:
    {"clf": [LogisticRegression(max_iter=2000)], "clf__C": [0.01, 0.1, 1, 10]},  #> Logistic regression with 4 values of C.
    {"clf": [RandomForestClassifier(random_state=0)], "clf__n_estimators": [200], "clf__max_depth": [None, 5],  #> Random forest with two depths...
     "scale": ["passthrough"]},                                       #> "passthrough" disables a step (trees don't need scaling).
]  #> Close the list opened above.
gs = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)  #> Search both families.
print(gs.best_params_["clf"], round(gs.best_score_, 4))  #> The winning model and its CV score.
```

## RandomizedSearchCV and successive halving {I}

When there are many hyperparameters, sampling random combinations from distributions finds good settings with far fewer fits than a full grid.

```py title="Random search over a gradient-boosting model"
from scipy.stats import loguniform, randint  #> Import `loguniform` and `randint` from `scipy.stats`.
from sklearn.model_selection import RandomizedSearchCV  #> Import `RandomizedSearchCV` from scikit-learn's `model_selection` module.
from sklearn.ensemble import HistGradientBoostingClassifier  #> Import `HistGradientBoostingClassifier` from scikit-learn's `ensemble` module.

dist = {  #> Distributions to sample hyperparameters from:
    "learning_rate": loguniform(1e-3, 0.5),      #> Sample on a log scale: good for rates and regularisation strengths.
    "max_leaf_nodes": randint(8, 64),            #> Integers 8..63.
    "min_samples_leaf": randint(5, 60),  #> Integers 5..59.
    "l2_regularization": loguniform(1e-4, 10),  #> Log-uniform from 0.0001 to 10.
}  #> Close the dictionary opened above.
rs = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=0), dist, n_iter=25,  #> Try 25 random combinations...
                        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc",  #> ...each scored by 5-fold CV ROC AUC...
                        random_state=0, n_jobs=-1)  #> ...reproducibly, in parallel.
rs.fit(X_tr, y_tr)  #> Run the search.
print({k: round(float(v), 4) if isinstance(v, float) else int(v) for k, v in rs.best_params_.items()})   #> Convert NumPy numbers to plain Python numbers for tidy printing.
print("best CV ROC AUC:", round(rs.best_score_, 4))  #> Best mean CV score.
```

:::tip Beyond random search
`HalvingGridSearchCV` / `HalvingRandomSearchCV` (enable with `from sklearn.experimental import enable_halving_search_cv`) discard poor candidates early. **Optuna** performs Bayesian optimisation and integrates well with scikit-learn and Keras.
:::

## Learning curves and validation curves {I}

:::def
**Learning curve**: training and validation scores plotted against training-set size; it shows whether more data would help.

**Validation curve**: training and validation scores plotted against one hyperparameter; it shows under- and overfitting.
:::


```py title="Diagnose bias vs variance" caption="Left: validation accuracy (labelled Test: the held-out folds) climbs as the training set grows and is still rising, so more data would probably help. Right: tiny C underfits badly; accuracy peaks near C = 1; beyond that training accuracy reaches 1.0 while validation dips slightly, a mild sign of overfitting."
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay  #> Import `LearningCurveDisplay` and `ValidationCurveDisplay` from scikit-learn's `model_selection` module.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))  #> Two panels.
LearningCurveDisplay.from_estimator(make_pipeline(StandardScaler(), SVC()), X_tr, y_tr,  #> Learning curve for a scaled SVC...
                                    train_sizes=np.linspace(0.1, 1.0, 8), cv=5, scoring="accuracy",  #> ...8 training sizes from 10% to 100%, 5-fold CV...
                                    ax=axes[0], score_name="accuracy", n_jobs=-1)   #> Trains on growing subsets of the data.
axes[0].set_title("Learning curve (SVC)")  #> Title.
ValidationCurveDisplay.from_estimator(make_pipeline(StandardScaler(), SVC()), X_tr, y_tr,  #> Validation curve for the same model...
                                      param_name="svc__C", param_range=np.logspace(-3, 3, 13), cv=5,  #> ...sweeping C from 0.001 to 1000...
                                      scoring="accuracy", ax=axes[1], score_name="accuracy", n_jobs=-1)   #> Sweeps ONE hyperparameter.
axes[1].set(xscale="log", title="Validation curve over C")  #> Log x-axis.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Nested cross-validation: an unbiased estimate after tuning {A}

:::def
**Nested cross-validation**: an outer CV loop that estimates performance, with an inner CV loop inside each outer fold that tunes hyperparameters.
:::


`search.best_score_` is optimistically biased, because the same folds were used to choose the hyperparameters. When you have no separate test set (common with small research datasets), wrap the search inside an outer CV loop.

```py title="Inner loop tunes, outer loop evaluates"
inner = StratifiedKFold(5, shuffle=True, random_state=1)  #> Inner CV: used to choose hyperparameters.
outer = StratifiedKFold(5, shuffle=True, random_state=2)  #> Outer CV: used to estimate performance.
tuned_svc = GridSearchCV(make_pipeline(StandardScaler(), SVC()),  #> A grid search object (not fitted yet)...
                         {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001]},  #> ...with 12 combinations...
                         cv=inner, scoring="roc_auc", n_jobs=-1)  #> ...tuned with the inner CV.
nested = cross_val_score(tuned_svc, X, y, cv=outer, scoring="roc_auc")    #> Each outer fold runs a complete grid search on its own training part.
tuned_svc.fit(X, y)  #> A separate, ordinary (non-nested) search on all the data, for comparison.
print(f"non-nested best_score_: {tuned_svc.best_score_:.4f}")  #> Potentially optimistic.
print(f"nested CV estimate:     {nested.mean():.4f} ± {nested.std():.4f}")  #> Unbiased estimate with its spread.
```

Here the two numbers agree closely: the dataset is easy and the grid is small, so there is little room for the search to overfit the folds. With large search spaces, small samples or noisy targets the non-nested score can be inflated by several points, and you cannot know in advance which situation you are in.

:::research Which number goes in the paper?
Report the **nested CV** score (or the score on an untouched test set) as your performance estimate, never `best_score_` from the same search. Report the hyperparameter search space and the selection metric, so others can reproduce the tuning.
:::

:::ex Try it yourself
1. Tune `n_neighbors` (1–30) and `weights` (`"uniform"`, `"distance"`) for kNN on the breast-cancer data with `GridSearchCV`.
2. Use `cross_validate(..., return_estimator=True)` to inspect how much logistic-regression coefficients vary across folds.
3. Compare 5-fold and 10-fold CV estimates and their standard deviations for the same model.
:::
