# Regression: predicting numbers

Regression models predict a continuous target. This chapter compares the main families (linear models, regularised linear models, nearest neighbours, trees and ensembles) on the insurance data, explains the metrics, and shows the diagnostic plots that tell you *how* a model is wrong.

## Set-up: one preprocessor, many models {B}

```py title="Shared data and preprocessing"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.
from sklearn.compose import ColumnTransformer  #> Import `ColumnTransformer` from scikit-learn's `compose` module.
from sklearn.pipeline import Pipeline, make_pipeline  #> Import `Pipeline` and `make_pipeline` from scikit-learn's `pipeline` module.
from sklearn.preprocessing import StandardScaler, OneHotEncoder  #> Import `StandardScaler` and `OneHotEncoder` from scikit-learn's `preprocessing` module.
from sklearn.impute import SimpleImputer  #> Import `SimpleImputer` from scikit-learn's `impute` module.

ins = pd.read_csv("insurance.csv")  #> Load the practice dataset.
X, y = ins.drop(columns="charges"), ins["charges"]  #> Features and target.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)  #> 80/20 split.

num_cols, cat_cols = ["age", "bmi", "children"], ["sex", "smoker", "region"]  #> Column groups.
pre = ColumnTransformer([  #> Preprocessing for the two column groups:
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),  #> Numeric: impute the median, then standardise.
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),  #> Categorical: one-hot encode.
])  #> Close the list and the call or tuple opened above.
print(X_tr.shape, X_te.shape)  #> (800, 6) and (200, 6).
```

## Regression metrics {B}

:::def
**MAE (mean absolute error)**: the average size of prediction errors, in the target's units.

**RMSE (root mean squared error)**: the square root of the average squared error. It penalises large errors more than MAE.

**R² (coefficient of determination)**: the share of the target's variance explained by the model: 1 is perfect, 0 is no better than predicting the mean, negative is worse.
:::


| Metric | Formula (in words) | Units | Read it as |
|---|---|---|---|
| **MAE** | mean of \|error\| | target units | "on average we are off by this much"; robust to outliers |
| **RMSE** | square root of mean squared error | target units | penalises large errors more than MAE |
| **R²** | 1 − SSE / total variance | none, at most 1 | share of variance explained; 0 = as good as predicting the mean, negative = worse |
| **MAPE** | mean of \|error\| / \|true\| | % | relative error; unstable when true values are near 0 |

```py title="A helper to evaluate any regressor"
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error  #> Import `mean_absolute_error`, `root_mean_squared_error`, `r2_score` and `mean_absolute_percentage_error` from scikit-learn's `metrics` module.

def evaluate(model, name):  #> Define a function `evaluate` that takes `model` and `name`.
    model.fit(X_tr, y_tr)                                         #> Fit on training data.
    pred = model.predict(X_te)                                    #> Predict the held-out test set.
    return {"model": name,  #> Send `{"model": name,` back to the caller.
            "MAE": mean_absolute_error(y_te, pred),  #> Average absolute error.
            "RMSE": root_mean_squared_error(y_te, pred),         #> scikit-learn >= 1.4 provides RMSE directly.
            "R2": r2_score(y_te, pred),  #> Share of variance explained.
            "MAPE%": 100 * mean_absolute_percentage_error(y_te, pred)}  #> Average percentage error.
```

## Linear models {B}

:::def
**Linear regression**: a model predicting the target as a weighted sum of the features plus an intercept, fitted by least squares.

**Coefficient**: the weight a linear model gives a feature: the change in prediction per unit change in that feature, holding the others fixed.

**Regularisation**: adding a penalty on model complexity (such as large coefficients) to reduce overfitting. **L2 (Ridge)** shrinks all coefficients; **L1 (Lasso)** can set some to exactly zero.
:::


Linear regression models the target as a weighted sum of features: ŷ = b + w₁x₁ + … + wₚxₚ. **Regularisation** adds a penalty on the size of the weights to reduce overfitting:

- **Ridge (L2)** shrinks all coefficients toward zero; good when many features each matter a little.
- **Lasso (L1)** can set coefficients exactly to zero, performing feature selection.
- **ElasticNet** mixes both. `alpha` sets the penalty strength (higher = simpler model).

```py title="Ordinary, Ridge, Lasso and ElasticNet"
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet  #> Import `LinearRegression`, `Ridge`, `Lasso` and `ElasticNet` from scikit-learn's `linear_model` module.
from sklearn.dummy import DummyRegressor  #> Import `DummyRegressor` from scikit-learn's `dummy` module.

results = [  #> Collect one result dict per model.
    evaluate(Pipeline([("pre", pre), ("m", DummyRegressor())]), "Mean baseline"),  #> Always predicts the training mean.
    evaluate(Pipeline([("pre", pre), ("m", LinearRegression())]), "Linear"),  #> Ordinary least squares.
    evaluate(Pipeline([("pre", pre), ("m", Ridge(alpha=1.0))]), "Ridge"),  #> L2 penalty.
    evaluate(Pipeline([("pre", pre), ("m", Lasso(alpha=10.0))]), "Lasso"),  #> L1 penalty.
    evaluate(Pipeline([("pre", pre), ("m", ElasticNet(alpha=0.1, l1_ratio=0.5))]), "ElasticNet"),  #> Half L1, half L2.
]  #> Close the list opened above.
print(pd.DataFrame(results).round(3).to_string(index=False))   #> to_string(index=False) prints a clean table.
```

### Reading coefficients {I}

```py title="Which features push predictions up or down?"
lin = Pipeline([("pre", pre), ("m", LinearRegression())]).fit(X_tr, y_tr)  #> Fit a linear model to inspect.
coefs = pd.Series(lin.named_steps["m"].coef_,  #> One coefficient per transformed feature...
                  index=lin.named_steps["pre"].get_feature_names_out()).sort_values()   #> Pair each coefficient with its feature name.
print(coefs.round(0))  #> Sorted from most negative to most positive.
print("intercept:", round(lin.named_steps["m"].intercept_, 0))  #> Predicted charges when every feature is 0 (i.e. at average numeric values).
```

Because numeric features were standardised, each numeric coefficient means "change in charges for a one-standard-deviation increase". The `smoker_yes` coefficient is the estimated extra cost of smoking, holding the other features fixed.

:::warn Coefficients are not causal effects
Regression coefficients describe associations in *this* data, conditional on the other features in the model. They can change sign when correlated features are added or removed. For inference (confidence intervals, p-values) use `statsmodels`; scikit-learn focuses on prediction.
:::

### The regularisation path {I}

```py title="Lasso drives coefficients to zero as alpha grows" caption="As the penalty alpha increases, Lasso shrinks coefficients and drops the weakest features first. smoker_yes and age survive longest: they matter most."
alphas = np.logspace(0, 4.3, 40)                          #> 40 values from 1 to ~20,000 on a log scale.
Xt_tr = pre.fit_transform(X_tr)  #> Preprocess once, outside a pipeline, for this plot.
names = pre.get_feature_names_out()  #> Feature names after preprocessing.
paths = np.array([Lasso(alpha=a, max_iter=20000).fit(Xt_tr, y_tr).coef_ for a in alphas])   #> One row of coefficients per alpha.
fig, ax = plt.subplots(figsize=(8, 4))  #> One panel.
for j, name in enumerate(names):  #> Loop over `enumerate(names)`, naming each item `j, name`.
    lw = 2.4 if name in ("cat__smoker_yes", "num__age", "num__bmi") else 1  #> Thicker lines for the key features.
    ax.plot(alphas, paths[:, j], lw=lw, label=name.split("__")[1])  #> One coefficient's path across alphas.
ax.set(xscale="log", xlabel="alpha (penalty strength)", ylabel="coefficient", title="Lasso regularisation path")  #> Log x-axis, labels and title.
ax.legend(fontsize=8, ncol=2, loc="upper right")  #> Compact two-column legend.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Non-linear models {I}

```py title="kNN, decision tree, random forest, gradient boosting"
from sklearn.neighbors import KNeighborsRegressor  #> Import `KNeighborsRegressor` from scikit-learn's `neighbors` module.
from sklearn.tree import DecisionTreeRegressor  #> Import `DecisionTreeRegressor` from scikit-learn's `tree` module.
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor  #> Import `RandomForestRegressor`, `HistGradientBoostingRegressor` and `GradientBoostingRegressor` from scikit-learn's `ensemble` module.
from sklearn.svm import SVR  #> Import `SVR` from scikit-learn's `svm` module.

results += [  #> Append more results to the list.
    evaluate(Pipeline([("pre", pre), ("m", KNeighborsRegressor(n_neighbors=10))]), "kNN (k=10)"),  #> Average of the 10 nearest customers.
    evaluate(Pipeline([("pre", pre), ("m", SVR(C=20000, epsilon=500))]), "SVR (RBF)"),  #> Support vector regression; C and epsilon set for this target's scale.
    evaluate(Pipeline([("pre", pre), ("m", DecisionTreeRegressor(max_depth=5, random_state=0))]), "Tree (depth 5)"),  #> One tree, at most 5 levels deep.
    evaluate(Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1))]), "Random forest"),  #> 300 trees, each leaf with at least 3 samples.
    evaluate(Pipeline([("pre", pre), ("m", GradientBoostingRegressor(random_state=0))]), "Gradient boosting"),  #> Classic gradient boosting.
    evaluate(Pipeline([("pre", pre), ("m", HistGradientBoostingRegressor(random_state=0))]), "HistGradientBoosting"),  #> Fast histogram-based boosting.
]  #> Close the list opened above.
table = pd.DataFrame(results).sort_values("RMSE").round(3)  #> Results as a table, best first.
print(table.to_string(index=False))  #> Print without the row index.
```

```py title="Compare models visually" caption="Test RMSE per model (lower is better). On this dataset the plain linear models win, narrowly ahead of gradient boosting and random forest, and far ahead of kNN and the baseline."
fig, ax = plt.subplots(figsize=(8, 4))  #> One panel.
t = table.sort_values("RMSE", ascending=False)  #> Worst first, so the best bar ends up at the top.
colors = ["#bbb" if m == "Mean baseline" else "#0f9b62" for m in t["model"]]  #> Grey for the baseline, green for real models.
bars = ax.barh(t["model"], t["RMSE"], color=colors)  #> Horizontal bars.
ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)  #> Write each RMSE at the end of its bar.
ax.set(xlabel="test RMSE (charges)", title="Model comparison on the insurance data")  #> Labels.
ax.grid(axis="y", visible=False)  #> Keep only vertical grid lines.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

:::key Simple models first
Why do linear models win here? The relationship in this data is almost linear: charges rise steadily with age and jump by a fixed amount for smokers. Flexible models have nothing extra to discover, and they pay for their flexibility with a little extra variance. On real data with interactions and thresholds the ranking often flips. You only find out by trying a baseline, a linear model and an ensemble, in that order. Note also how badly kNN does: with one-hot columns and few samples, "nearest neighbours" are not very near.
:::

### How trees and ensembles work {I}

:::def
**Decision tree**: a model that makes predictions by asking a sequence of yes/no questions about the features, ending in a leaf with a prediction.

**Ensemble**: a model that combines many models' predictions, usually to improve accuracy and stability.

**Random forest**: an ensemble of decision trees, each trained on a bootstrap sample with random subsets of features, whose predictions are averaged.

**Gradient boosting**: an ensemble that adds shallow trees one at a time, each correcting the errors of the previous ones.
:::


- A **decision tree** splits the data with yes/no questions ("smoker?", "age < 43?") to make groups with similar targets. It is interpretable but overfits when deep.
- A **random forest** averages hundreds of deep trees, each trained on a bootstrap sample with random feature subsets. Averaging cancels out much of the overfitting. It is robust and needs little tuning.
- **Gradient boosting** builds shallow trees *sequentially*, each one correcting the errors of the ensemble so far. It is often the most accurate for tabular data, but more sensitive to hyperparameters (`learning_rate`, `max_iter`/`n_estimators`, `max_depth`). `HistGradientBoosting*` is scikit-learn's fast implementation for larger datasets, with native missing-value support. XGBoost, LightGBM and CatBoost are popular external libraries in the same family.

```py title="Peek inside a shallow tree"
from sklearn.tree import export_text  #> Import `export_text` from scikit-learn's `tree` module.
tree = Pipeline([("pre", pre), ("m", DecisionTreeRegressor(max_depth=2, random_state=0))]).fit(X_tr, y_tr)  #> A tiny tree that is easy to read.
print(export_text(tree.named_steps["m"], feature_names=list(pre.get_feature_names_out())))   #> The learned if/else rules.
```

## Diagnostic plots: how is the model wrong? {I}

:::def
**Residual**: the difference between an observed value and the model's prediction (actual − predicted).
:::


```py title="Predicted vs actual, and residuals" caption="Left: points should hug the diagonal. Right: residuals should be a shapeless cloud around zero; visible structure means the model is missing a pattern."
from sklearn.metrics import PredictionErrorDisplay  #> Import `PredictionErrorDisplay` from scikit-learn's `metrics` module.

rf = Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1))]).fit(X_tr, y_tr)  #> Refit the random forest.
pred = rf.predict(X_te)                                           #> Diagnose the random forest (try it with the linear model too).
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))  #> Two panels.
PredictionErrorDisplay.from_predictions(y_te, pred, kind="actual_vs_predicted", ax=axes[0],  #> Panel 1: actual vs predicted...
                                        scatter_kwargs={"s": 12, "alpha": 0.6})   #> scikit-learn's built-in display class.
axes[0].set_title("Actual vs predicted")  #> Title.
PredictionErrorDisplay.from_predictions(y_te, pred, kind="residual_vs_predicted", ax=axes[1],  #> Panel 2: residual (actual - predicted) vs predicted...
                                        scatter_kwargs={"s": 12, "alpha": 0.6})  #> ...small, semi-transparent points.
axes[1].set_title("Residuals vs predicted")  #> Title.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

:::tip Skewed targets
Costs, incomes and counts are often right-skewed. Modelling `log(y)` can help linear models. In scikit-learn use `TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)`, so predictions and metrics come back in original units automatically.
:::

## Quantifying uncertainty in a test score {A}

:::def
**Bootstrap**: estimating the uncertainty of a statistic by recomputing it on many random resamples (drawn with replacement) of the data.

**Confidence interval (CI)**: a range that would contain the true value of a quantity in a stated share (often 95%) of repeated experiments.
:::


A single test RMSE is an estimate with sampling error. A **bootstrap** over test samples gives a confidence interval.

```py title="Bootstrap 95% confidence interval for test RMSE"
rng = np.random.default_rng(0)  #> Seeded generator for resampling.
y_true_arr, pred_arr = y_te.to_numpy(), pred  #> Plain arrays for fast indexing.
boot = []  #> Will hold 2000 bootstrap RMSEs.
for _ in range(2000):  #> Loop over `range(2000)`, naming each item `_`.
    idx = rng.integers(0, len(y_true_arr), len(y_true_arr))        #> Resample test rows WITH replacement.
    boot.append(root_mean_squared_error(y_true_arr[idx], pred_arr[idx]))  #> RMSE on this resample.
lo, hi = np.percentile(boot, [2.5, 97.5])                          #> The middle 95% of bootstrap RMSEs.
print(f"RMSE = {root_mean_squared_error(y_true_arr, pred_arr):.0f}  (95% CI {lo:.0f} to {hi:.0f})")  #> The point estimate with its interval.
```

:::research Report intervals, not just point estimates
"RMSE 2,574 (95% CI 2,159–3,032)" tells readers how much to trust the number, and whether two models' scores really differ. [[ch:research-choosing]] covers comparing models rigorously.
:::

:::ex Try it yourself
1. Wrap the Ridge pipeline in `TransformedTargetRegressor` with a log transform and compare test RMSE.
2. Add an interaction feature `smoker × bmi` (hint: `PolynomialFeatures(interaction_only=True)` or a custom column) to the linear model. Does it close the gap to the random forest?
3. Plot test RMSE against `max_depth` from 1 to 15 for a single decision tree. Where does overfitting start?
:::
