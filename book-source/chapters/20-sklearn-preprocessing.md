# Preprocessing, feature engineering and pipelines

Models need numbers on sensible scales, with no missing values. Real tables have text categories, gaps and wildly different units. scikit-learn's **transformers** fix each of these problems, and a **Pipeline** chains them with a model so the whole thing is fitted correctly, without leakage, in one call.

## Scaling numeric features {B}

:::def
**Standardisation (z-scoring)**: rescaling a feature to mean 0 and standard deviation 1: (x − mean) / SD.

**Normalisation (min-max scaling)**: rescaling a feature to a fixed range, usually 0–1: (x − min) / (max − min).

**Transformer (scikit-learn)**: an estimator with `fit` and `transform` that changes data, such as a scaler, encoder or imputer.
:::


Distance-based and gradient-based models (kNN, SVM, logistic/linear regression with regularisation, neural networks, PCA) are sensitive to feature scale. Tree-based models (decision trees, random forests, gradient boosting) are not.

```py title="StandardScaler, MinMaxScaler, RobustScaler"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler  #> Import `StandardScaler`, `MinMaxScaler` and `RobustScaler` from scikit-learn's `preprocessing` module.

X = np.array([[1.0, 200.0], [2.0, 300.0], [3.0, 250.0], [4.0, 5000.0]])   #> Column 2 has an outlier (5000).
for scaler in [StandardScaler(), MinMaxScaler(), RobustScaler()]:  #> Loop over `[StandardScaler(), MinMaxScaler(), RobustScaler()]`, naming each item `scaler`.
    Xs = scaler.fit_transform(X)                       #> Learn the statistics from X, then transform X.
    print(f"{type(scaler).__name__:<15}", Xs[:, 1].round(2))  #> The scaler's class name and the scaled second column.
sc = StandardScaler().fit(X)  #> Fit a scaler to inspect what it learned.
print("learned mean_:", sc.mean_, " scale_:", sc.scale_.round(1))   #> Learned statistics end with "_".
print(sc.inverse_transform(sc.transform(X))[0])        #> inverse_transform undoes the scaling.
```

| Scaler | Formula | Use when |
|---|---|---|
| `StandardScaler` | (x − mean) / sd | default choice; roughly symmetric features |
| `MinMaxScaler` | (x − min) / (max − min) | you need a bounded range [0, 1], e.g. pixel intensities |
| `RobustScaler` | (x − median) / IQR | features with outliers |
| `PowerTransformer` / `np.log1p` | makes skewed data more normal | strongly right-skewed features (incomes, counts) |

:::warn Fit on training data, transform both
`scaler.fit_transform(X_train)` then `scaler.transform(X_test)`. Never `fit` on the test set, and never fit on all data before splitting. A Pipeline enforces this for you.
:::

## Encoding categorical features {B}

```py title="One-hot and ordinal encoding"
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  #> Import `OneHotEncoder` and `OrdinalEncoder` from scikit-learn's `preprocessing` module.

df = pd.DataFrame({"region": ["north", "south", "east", "south"],  #> A small table with an unordered category...
                   "size": ["S", "L", "M", "M"]})  #> ...and an ordered one.
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")   #> handle_unknown="ignore": an unseen category becomes all zeros instead of an error.
print(ohe.fit_transform(df[["region"]]))                            #> One 0/1 column per category (alphabetical order).
print(ohe.get_feature_names_out())  #> The names of the new columns, e.g. region_east.
print(ohe.transform(pd.DataFrame({"region": ["west"]})))           #> "west" was never seen during fit: all zeros.
ord_enc = OrdinalEncoder(categories=[["S", "M", "L"]])              #> State the ORDER explicitly for ordered categories.
print(ord_enc.fit_transform(df[["size"]]).ravel())                  #> S=0, M=1, L=2.
```

:::tip Which encoder?
Use **one-hot** for unordered categories (region, colour), **ordinal** for genuinely ordered ones (S < M < L, education level). Tree models also cope well with ordinal codes for unordered categories. For very high-cardinality columns (thousands of postcodes), consider `TargetEncoder` (it is cross-fitted internally to limit leakage) or grouping rare levels with `OneHotEncoder(min_frequency=...)`.
:::

## Imputing missing values {B}

```py title="SimpleImputer and KNNImputer"
from sklearn.impute import SimpleImputer, KNNImputer  #> Import `SimpleImputer` and `KNNImputer` from scikit-learn's `impute` module.

Xm = np.array([[1.0, 10.0], [np.nan, 12.0], [3.0, np.nan], [4.0, 16.0]])  #> Two missing values.
print(SimpleImputer(strategy="median").fit_transform(Xm))          #> Replace NaN with the column median (learned on fit).
print(SimpleImputer(strategy="mean", add_indicator=True).fit_transform(Xm))   #> add_indicator appends 0/1 "was missing" columns.
print(KNNImputer(n_neighbors=2).fit_transform(Xm).round(2))        #> Fill from the 2 most similar rows.
print(SimpleImputer(strategy="most_frequent").fit_transform(np.array([["a"], ["b"], ["a"], [np.nan]], dtype=object)).ravel())   #> For categories.
```

:::note Models that accept NaN directly
`HistGradientBoostingClassifier/Regressor`, `DecisionTree*`, and `RandomForest*` in recent scikit-learn versions handle missing values natively. Most other estimators raise `ValueError: Input X contains NaN`.
:::

## Pipelines: preprocessing + model as one object {B}

:::def
**Pipeline**: a chain of preprocessing steps and a final model that is fitted and applied as one estimator, guaranteeing every step learns only from training data.
:::


A `Pipeline` is a list of steps. Calling `fit` runs `fit_transform` on each transformer, in order, then `fit` on the final model. Calling `predict` runs `transform` through each step, then `predict`. The fitted statistics always come from the data passed to `fit`.

```py title="A simple pipeline"
from sklearn.pipeline import Pipeline, make_pipeline  #> Import `Pipeline` and `make_pipeline` from scikit-learn's `pipeline` module.
from sklearn.linear_model import LogisticRegression  #> Import `LogisticRegression` from scikit-learn's `linear_model` module.
from sklearn.datasets import load_breast_cancer  #> Import `load_breast_cancer` from scikit-learn's `datasets` module.
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.

X, y = load_breast_cancer(return_X_y=True, as_frame=True)       #> 569 tumours, 30 numeric features, target 0 = malignant, 1 = benign.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)  #> Stratified 75/25 split.

pipe = Pipeline([  #> A Pipeline is built from a list of (name, step) pairs.
    ("impute", SimpleImputer(strategy="median")),               #> Step name + transformer object.
    ("scale", StandardScaler()),  #> Then standardise.
    ("clf", LogisticRegression(max_iter=1000)),                 #> The last step is the model.
])  #> Close the list and the call or tuple opened above.
pipe.fit(X_tr, y_tr)                                            #> Each step is fitted on training data only.
print("test accuracy:", round(pipe.score(X_te, y_te), 4))  #> The pipeline scores like a model.
print(pipe.named_steps["scale"].mean_[:3].round(2))             #> Reach inside to inspect a fitted step.
quick = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))   #> Same idea; step names are generated automatically.
print([name for name, _ in quick.steps])  #> Automatic names: lowercase class names.
```

## ColumnTransformer: different processing per column {I}

:::def
**ColumnTransformer**: a scikit-learn tool that applies different preprocessing to different groups of columns and joins the results.
:::


Real tables mix numeric and categorical columns. `ColumnTransformer` routes each group of columns through its own mini-pipeline and concatenates the results.

```py title="The standard tabular preprocessing recipe"
from sklearn.compose import ColumnTransformer, make_column_selector  #> Import `ColumnTransformer` and `make_column_selector` from scikit-learn's `compose` module.

ins = pd.read_csv("insurance.csv")                               #> The Part III dataset (contains missing BMI).
X = ins.drop(columns="charges")  #> Features: every column except the target.
y = ins["charges"]  #> Target.
num_cols = ["age", "bmi", "children"]  #> Numeric columns.
cat_cols = ["sex", "smoker", "region"]  #> Categorical columns.

numeric = Pipeline([("impute", SimpleImputer(strategy="median")),  #> Numeric branch: fill gaps with the median...
                    ("scale", StandardScaler())])  #> ...then standardise.
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),  #> Categorical branch: fill gaps with the most frequent value...
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))])   #> drop="if_binary": one column for yes/no features.
pre = ColumnTransformer([  #> Route each column group to its branch:
    ("num", numeric, num_cols),                                   #> (name, transformer, columns)
    ("cat", categorical, cat_cols),  #> Categorical columns through the categorical branch.
], remainder="drop")                                              #> Columns not listed are dropped ("passthrough" keeps them).

Xt = pre.fit_transform(X)  #> Fit both branches and transform: a numeric matrix comes out.
print(Xt.shape)  #> Display the value of `Xt.shape`.
print(pre.get_feature_names_out())                                #> Track exactly which column is which after transformation.
```

```py title="Plugging it into a model"
from sklearn.linear_model import Ridge  #> Import `Ridge` from scikit-learn's `linear_model` module.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)  #> 80/20 split of the raw table.
model = Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))])       #> A pipeline inside a pipeline.
model.fit(X_tr, y_tr)                                             #> Raw DataFrame in: imputing, scaling and encoding happen inside.
print("test R2:", round(model.score(X_te, y_te), 4))  #> R² on the held-out test set.
new_customer = pd.DataFrame([{"age": 45, "sex": "female", "bmi": np.nan, "children": 2,  #> A new raw record, including a missing BMI...
                              "smoker": "no", "region": "southwest"}])  #> ...continued.
print("predicted charges:", model.predict(new_customer).round(0))  #> Works on raw new data, missing BMI included.
```

:::tip Select columns by type automatically
`make_column_selector(dtype_include="number")` and `make_column_selector(dtype_include=object)` pick columns by dtype, so the preprocessor adapts when columns are added. In pandas 3, string columns match `dtype_include=["object", "string"]`.
:::

## Seeing leakage with your own eyes {I}

:::def
**Data leakage**: information from outside the training data (often from the test set or the target itself) influencing model training, which produces overly optimistic results.
:::


Here is why pipelines matter. We select the "best" features using *all* the data, then cross-validate on pure noise.

```py title="Feature selection outside vs inside the pipeline"
from sklearn.feature_selection import SelectKBest, f_classif  #> Import `SelectKBest` and `f_classif` from scikit-learn's `feature_selection` module.
from sklearn.model_selection import cross_val_score  #> Import `cross_val_score` from scikit-learn's `model_selection` module.

rng = np.random.default_rng(0)  #> Seeded generator.
X_noise = rng.normal(size=(100, 5000))               #> 100 samples, 5000 features of pure random noise.
y_noise = rng.integers(0, 2, 100)                    #> Random labels: no real signal exists. Honest accuracy should be ~0.5.

X_sel = SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)   #> WRONG: picks the 20 features most correlated with y using ALL samples.
wrong = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y_noise, cv=5).mean()  #> CV on the pre-selected features.

right_pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))   #> RIGHT: selection re-done inside each training fold.
right = cross_val_score(right_pipe, X_noise, y_noise, cv=5).mean()  #> CV where selection happens within each training fold.
print(f"selection outside CV: {wrong:.2f}   <- looks impressive, entirely fake")  #> The leaky estimate.
print(f"selection inside CV : {right:.2f}   <- honest: no signal")  #> The honest estimate.
```

:::bug My cross-validation score is great but the model fails on new data
Classic leakage. Any step that learns from data (scaling, imputing, encoding with target information, feature selection, oversampling such as SMOTE, PCA) must happen **inside** the cross-validation loop, which means inside a Pipeline. Also check for duplicate rows across splits and for features that are computed from the target.
:::

## Feature engineering {I}

:::def
**Feature engineering**: creating new, more informative features from raw data, such as ratios, logs, interactions, date parts and counts.
:::


```py title="Creating features inside a pipeline"
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, KBinsDiscretizer, SplineTransformer  #> Import `FunctionTransformer`, `PolynomialFeatures`, `KBinsDiscretizer` and `SplineTransformer` from scikit-learn's `preprocessing` module.

log_tf = FunctionTransformer(np.log1p, feature_names_out="one-to-one")   #> Wrap any function as a transformer (log1p = log(1 + x), safe at 0).
print(log_tf.fit_transform(np.array([[0.0], [9.0], [99.0]])).ravel().round(3))  #> log1p of 0, 9 and 99: 0, 2.303, 4.605.
poly = PolynomialFeatures(degree=2, include_bias=False)                 #> Adds squares and pairwise interactions.
print(poly.fit_transform([[2, 3]]), poly.get_feature_names_out(["a", "b"]))  #> [a, b, a², ab, b²] = [2, 3, 4, 6, 9].
bins = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile", quantile_method="averaged_inverted_cdf")   #> Equal-frequency bins.
print(bins.fit_transform(np.array([[1], [2], [3], [10], [20], [30]])).ravel())  #> Each value replaced by its bin number.
spl = SplineTransformer(n_knots=4, degree=3)                            #> Smooth non-linear basis for one feature.
print(spl.fit_transform(np.linspace(0, 1, 5).reshape(-1, 1)).shape)  #> One input column becomes several spline basis columns.
```

```py title="A custom transformer class"
from sklearn.base import BaseEstimator, TransformerMixin  #> Import `BaseEstimator` and `TransformerMixin` from scikit-learn's `base` module.

class RatioFeatures(BaseEstimator, TransformerMixin):       #> Inherit the standard API (fit_transform, get_params, ...).
    """Add the ratio of two columns as a new column."""  #> Docstring.
    def __init__(self, num="bmi", den="age"):               #> Store hyperparameters only; no logic in __init__.
        self.num, self.den = num, den  #> Store the column names unchanged (scikit-learn convention).
    def fit(self, X, y=None):  #> Define the method `fit`; besides `self` (the object) it takes `X` and `y`.
        return self                                         #> Nothing to learn; must still return self.
    def transform(self, X):  #> Define the method `transform`; besides `self` (the object) it takes `X`.
        X = X.copy()  #> Never modify the caller's DataFrame.
        X[f"{self.num}_per_{self.den}"] = X[self.num] / X[self.den]  #> New column, e.g. bmi_per_age.
        return X  #> Send `X` back to the caller.

print(RatioFeatures().fit_transform(ins[["age", "bmi"]].head(3)).round(3))  #> Use it like any transformer.
```

## Inspecting and displaying pipelines {I}

```py title="get_params, set_params and the step__param syntax"
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))  #> A two-step pipeline.
print(list(pipe.get_params().keys())[-6:])              #> Every nested hyperparameter is addressable as step__param.
pipe.set_params(logisticregression__C=0.1)              #> Double underscore reaches inside a step. Grid search uses exactly this.
print(pipe.named_steps["logisticregression"].C)  #> Display the value of `pipe.named_steps["logisticregression"].C`.
print(pipe)                                             #> Text summary. In Jupyter, pipelines display as an interactive diagram.
```

:::ex Try it yourself
1. Build a pipeline for the insurance data that log-transforms `charges` as the target with `TransformedTargetRegressor(regressor=..., func=np.log1p, inverse_func=np.expm1)`. Does test R² improve?
2. Add `PolynomialFeatures(2)` to the numeric branch. What happens to the number of features?
3. Repeat the leakage experiment with `k=5` and `k=200`. How does the fake score change?
:::
