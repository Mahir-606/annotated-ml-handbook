# Interpreting, explaining and saving models

A model that predicts well is only half the job in research. Reviewers and colleagues will ask *which features matter*, *how* they affect predictions, and whether the model can be reused. This chapter covers built-in importances and their pitfalls, permutation importance, partial dependence, SHAP, and persisting models safely.

## Set-up {B}

```py title="A random forest on the insurance data"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.
from sklearn.compose import ColumnTransformer  #> Import `ColumnTransformer` from scikit-learn's `compose` module.
from sklearn.pipeline import Pipeline  #> Import `Pipeline` from scikit-learn's `pipeline` module.
from sklearn.preprocessing import OneHotEncoder  #> Import `OneHotEncoder` from scikit-learn's `preprocessing` module.
from sklearn.impute import SimpleImputer  #> Import `SimpleImputer` from scikit-learn's `impute` module.
from sklearn.ensemble import RandomForestRegressor  #> Import `RandomForestRegressor` from scikit-learn's `ensemble` module.

ins = pd.read_csv("insurance.csv")  #> Load the practice dataset.
rng = np.random.default_rng(0)  #> Seeded generator.
ins["noise"] = rng.normal(size=len(ins))                 #> A pure-noise feature: a good method should rank it near the bottom.
X, y = ins.drop(columns="charges"), ins["charges"]  #> Features and target.
X = X.astype({"age": float, "children": float})          #> Integer columns -> float. Partial dependence plots refuse integer data (see the Bug hunt box below).
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)  #> 75/25 split.
pre = ColumnTransformer([  #> Preprocessing:
    ("num", SimpleImputer(strategy="median"), ["age", "bmi", "children", "noise"]),  #> Numeric columns: impute only (trees don't need scaling).
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), ["sex", "smoker", "region"]),  #> Categorical columns: one-hot encode.
], verbose_feature_names_out=False)                      #> Keep plain names ("age" instead of "num__age").
rf = Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1))])  #> Preprocessing + random forest.
rf.fit(X_tr, y_tr)  #> Train.
print("test R2:", round(rf.score(X_te, y_te), 3))  #> Test performance.
```

## Impurity-based feature importance {I}

:::def
**Feature importance**: a score of how much a model relies on each feature.
:::


Tree ensembles expose `feature_importances_`: how much each feature reduced the splitting criterion, summed over all trees. It is free to compute but has known biases.

```py title="feature_importances_"
names = rf.named_steps["pre"].get_feature_names_out()  #> Names after one-hot encoding.
imp = pd.Series(rf.named_steps["m"].feature_importances_, index=names).sort_values(ascending=False)  #> Pair importances with names, largest first.
print(imp.round(4))  #> Importances sum to 1.
```

:::warn Impurity importance is biased
It is computed on *training* data, and it favours continuous, high-cardinality features, which offer many possible split points. Notice that the pure-noise column outranks `children`, `sex` and every region column above. Prefer **permutation importance on held-out data** for conclusions.
:::

## Permutation importance {I}

:::def
**Permutation importance**: the drop in a model's score when one feature's values are randomly shuffled, which breaks its link with the target.
:::


Shuffle one feature's values in the test set and measure how much the score drops. If the model relied on that feature, performance falls; if not, nothing changes. It is model-agnostic and computed on held-out data.

```py title="Permutation importance on the test set" caption="Mean drop in test R² when each feature is shuffled (error bars: SD over 20 repeats). Smoking dominates; the noise feature correctly scores about zero."
from sklearn.inspection import permutation_importance  #> Import `permutation_importance` from scikit-learn's `inspection` module.

pi = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=0, scoring="r2", n_jobs=-1)   #> Permutes RAW input columns, before preprocessing.
pim = pd.DataFrame({"mean": pi.importances_mean, "sd": pi.importances_std}, index=X_te.columns).sort_values("mean")  #> Mean and SD of the score drop per feature.
print(pim.round(4).iloc[::-1])  #> Largest first.
fig, ax = plt.subplots(figsize=(7, 3.8))  #> One panel.
ax.barh(pim.index, pim["mean"], xerr=pim["sd"], color="#0f9b62")  #> Horizontal bars with error bars.
ax.set(xlabel="decrease in R² when shuffled", title="Permutation importance (test set)")  #> Labels.
ax.grid(axis="y", visible=False)  #> Keep only vertical grid lines.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

:::note Correlated features share importance
If two features carry the same information, shuffling one barely hurts because the model uses the other. Both then look unimportant. Check correlations first, and consider permuting correlated groups together.
:::

## Partial dependence and ICE plots {I}

:::def
**Partial dependence plot (PDP)**: the model's average prediction as one feature varies, with the other features averaged out.

**ICE plot (individual conditional expectation)**: one line per sample showing how its prediction changes as one feature varies.
:::


A **partial dependence plot (PDP)** shows the average predicted outcome as one feature varies, averaging over the others. **ICE** lines show the same for individual samples, revealing heterogeneity.

```py title="How do age and BMI affect predicted charges?" caption="Thin lines: individual customers (ICE); thick line: the average (PDP). Predicted charges rise steadily with age; for BMI the curve is flat below 30 and rises above it, the hinge built into this dataset."
from sklearn.inspection import PartialDependenceDisplay  #> Import `PartialDependenceDisplay` from scikit-learn's `inspection` module.

fig, axes = plt.subplots(1, 2, figsize=(11, 4))  #> Two panels.
PartialDependenceDisplay.from_estimator(rf, X_te, features=["age", "bmi"], kind="both",   #> "both" = PDP + ICE lines.
                                        subsample=80, random_state=0, ax=axes,  #> Draw ICE lines for 80 random customers...
                                        ice_lines_kw={"alpha": 0.15}, pd_line_kw={"color": "#e0245e", "lw": 3})  #> ...faint ICE lines, bold PDP line.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

```py title="Two-way partial dependence" caption="Joint partial dependence of age and BMI on predicted charges. The bands are mostly vertical, so age dominates; they tilt only at high BMI, where both effects add up. Strongly curved or crossing bands would signal an interaction."
fig, ax = plt.subplots(figsize=(6, 4.4))  #> One panel.
PartialDependenceDisplay.from_estimator(rf, X_te, features=[("age", "bmi")], grid_resolution=25, ax=ax)   #> A tuple of two features gives a 2-D contour; a coarser grid keeps it fast (default 100x100 = 10,000 points).
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

:::note Categorical features in partial dependence
For a categorical feature on its own, pass `categorical_features=["smoker"]` to get a bar per category. Two-way plots that mix a continuous and a categorical feature are not supported. Instead, draw one-way ICE plots coloured by group, or compare PDPs computed on each subgroup.
:::

## SHAP values {A}

:::def
**SHAP value**: the contribution of one feature to one prediction, based on Shapley values from game theory. A sample's SHAP values add up to its prediction minus the average prediction.
:::


**SHAP** (SHapley Additive exPlanations) splits each individual prediction into additive contributions from each feature, grounded in cooperative game theory. It gives both local explanations (one prediction) and global summaries. It lives in the separate `shap` package (`pip install shap`).

```py norun title="SHAP for a tree model (requires the shap package)"
import shap                                                            #> pip install shap

pre_fitted, model = rf.named_steps["pre"], rf.named_steps["m"]  #> The fitted preprocessor and the forest.
X_te_t = pd.DataFrame(pre_fitted.transform(X_te), columns=pre_fitted.get_feature_names_out())   #> SHAP explains the model on the TRANSFORMED features.
explainer = shap.TreeExplainer(model)                                  #> Fast, exact algorithm for tree ensembles.
sv = explainer(X_te_t)                                                 #> An Explanation object: one row of contributions per sample.
shap.plots.beeswarm(sv)                                                #> Global view: each dot is one sample's contribution for one feature.
shap.plots.waterfall(sv[0])                                            #> Local view: how each feature moved sample 0 from the average prediction.
```

:::research Interpretation claims need care
Importance and SHAP describe **the model**, not the world. A feature can be important because it is a proxy for something else, because of leakage, or because of a quirk of the sample. Validate interpretations with domain knowledge, stability across CV folds and seeds, and if possible on external data. Avoid causal language ("X causes Y") unless your design supports it.
:::

## Saving and loading models {B}

:::def
**Serialization**: saving an object, such as a fitted model, to a file so it can be loaded later. `joblib` and `pickle` do this in Python.
:::


Save the **whole fitted pipeline**, so preprocessing travels with the model.

```py title="joblib: the standard way"
import joblib, sklearn, json, platform  #> Import joblib, used to save and load fitted models and scikit-learn and the standard-library `json` module and the `platform` module (system and version info).

joblib.dump(rf, "insurance_rf.joblib")                     #> Serialise the fitted pipeline to disk (compress=3 makes smaller files).
loaded = joblib.load("insurance_rf.joblib")                 #> Load it back later, or on another machine.
print(np.allclose(loaded.predict(X_te), rf.predict(X_te)))  #> Identical predictions.
meta = {"sklearn": sklearn.__version__, "python": platform.python_version(),  #> Metadata to store next to the model...
        "features": list(X.columns), "test_r2": round(rf.score(X_te, y_te), 4)}  #> ...the expected input columns and the test score.
with open("insurance_rf.meta.json", "w") as f:  #> Open a JSON file for writing.
    json.dump(meta, f, indent=2)                           #> Record versions and inputs beside the model file.
print(meta)  #> Display the value of `meta`.
```

:::warn Pickles are version-specific and unsafe
A model saved with one scikit-learn version may not load (or may behave differently) under another, so record versions as above. **Never load a `.joblib`/`.pkl` file from an untrusted source**: unpickling can execute arbitrary code. For long-term or cross-language deployment, consider the `skops` format or exporting to ONNX (`skl2onnx`).
:::

:::bug `ValueError: The column 0 contains integer data. Partial dependence plots are not supported for integer data`
Recent scikit-learn versions refuse to compute partial dependence on integer columns, because the grid of values would be silently rounded. Convert those columns to float first: `X = X.astype({"age": float})`. That is why the set-up code above casts `age` and `children`.
:::

:::ex Try it yourself
1. Compute permutation importance with `scoring="neg_mean_absolute_error"`. Does the ranking change?
2. Draw ICE plots for `children`. Is the effect the same for everyone?
3. Save a fitted pipeline, restart Python, load it, and predict for a brand-new customer DataFrame.
:::
