@kind appendix

# Solutions: Parts V–VIII

Worked solutions to the exercises in the machine-learning, deep-learning, research and debugging chapters. Exact scores depend on random splits; what matters is the method and the conclusion.

## Shared set-up {B}

```py title="Imports used throughout this appendix"
import numpy as np                                                       #> Arrays.
import pandas as pd                                                      #> Tables.
import matplotlib.pyplot as plt                                          #> Plots.
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_digits, make_classification   #> Built-in data.
from sklearn.model_selection import (train_test_split, cross_val_score, cross_validate,   #> Splitting and CV...
                                     cross_val_predict, GridSearchCV, StratifiedKFold, KFold)   #> ...and tuning.
from sklearn.pipeline import make_pipeline, Pipeline                     #> Pipelines.
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor   #> Column routing; target transforms.
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures   #> Preprocessing.
from sklearn.impute import SimpleImputer                                 #> Missing values.
from sklearn.linear_model import LogisticRegression, Ridge               #> Linear models.
from sklearn.neighbors import KNeighborsClassifier                       #> kNN.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier   #> Ensembles.
from sklearn.tree import DecisionTreeRegressor                           #> Single tree.
from sklearn.dummy import DummyClassifier                                #> Baseline.
from sklearn.metrics import root_mean_squared_error, recall_score, precision_score, balanced_accuracy_score   #> Metrics.
ins = pd.read_csv("insurance.csv")                                       #> The practice dataset.
print("ready", ins.shape)                                                #> Check.
```

## [[ch:sklearn-concepts]]: ML concepts and the scikit-learn API

```py title="Exercise 1: three models on the wine data"
Xw, yw = load_wine(return_X_y=True)                                       #> 178 wines, 3 classes.
Xtr, Xte, ytr, yte = train_test_split(Xw, yw, stratify=yw, random_state=0)   #> Stratified split.
for name, m in [("Dummy", DummyClassifier()), ("kNN", make_pipeline(StandardScaler(), KNeighborsClassifier())),   #> Baseline and scaled kNN...
                ("RandomForest", RandomForestClassifier(random_state=0))]:   #> ...and a forest.
    print(f"{name:<13}", round(m.fit(Xtr, ytr).score(Xte, yte), 3))      #> Fit, then test accuracy.
```

```py title="Exercise 2: Ridge tames a degree-15 polynomial" caption="The same degree-15 polynomial fitted by plain least squares (wild wiggles) and by Ridge (smooth): regularisation controls flexibility."
from sklearn.linear_model import LinearRegression                        #> Unregularised fit.
rng = np.random.default_rng(1)                                            #> Same recipe as the chapter.
x = np.sort(rng.uniform(0, 1, 40)); y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 40)   #> Noisy sine.
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.4, random_state=3)   #> Split.
grid = np.linspace(0, 1, 300).reshape(-1, 1)                               #> For drawing curves.
fig, ax = plt.subplots(figsize=(7, 3.4))                                   #> New figure.
ax.scatter(x_tr, y_tr, s=14, color="k")                                    #> Training points.
for name, reg in [("LinearRegression", LinearRegression()), ("Ridge(alpha=1e-3)", Ridge(alpha=1e-3))]:   #> Two models.
    m = make_pipeline(PolynomialFeatures(15), StandardScaler(), reg).fit(x_tr.reshape(-1, 1), y_tr)   #> Scaling makes the penalty treat all powers fairly.
    ax.plot(grid, m.predict(grid), label=f"{name}: test MSE {np.mean((m.predict(x_te.reshape(-1, 1)) - y_te) ** 2):.3f}")   #> Curve with its test error.
ax.set_ylim(-2, 2); ax.legend(fontsize=8); plt.show()                      #> Limits, legend, draw.
```

```py title="Exercise 3: accuracy depends on the split"
Xi, yi = load_iris(return_X_y=True)                                        #> Iris.
accs = [KNeighborsClassifier().fit(a, c).score(b, d)                       #> Fit and score...
        for a, b, c, d in (train_test_split(Xi, yi, test_size=0.25, stratify=yi, random_state=s) for s in range(5))]   #> ...on five different splits.
print(np.round(accs, 3), "range:", round(max(accs) - min(accs), 3))       #> Several points of difference from the split alone.
```

## [[ch:sklearn-preprocessing]]: Preprocessing and pipelines

```py title="Exercises 1 and 2: log target and polynomial features"
X, y = ins.drop(columns="charges"), ins["charges"]                         #> Features and target.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)   #> Split.
def make_pre(poly=False):
    steps = [SimpleImputer(strategy="median"), StandardScaler()] + ([PolynomialFeatures(2, include_bias=False)] if poly else [])   #> Optional polynomial step.
    return ColumnTransformer([("num", make_pipeline(*steps), ["age", "bmi", "children"]),   #> Numeric branch.
                              ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), ["sex", "smoker", "region"])])   #> Categorical branch.
base = Pipeline([("pre", make_pre()), ("m", Ridge())])                     #> Plain Ridge.
logt = TransformedTargetRegressor(regressor=Pipeline([("pre", make_pre()), ("m", Ridge())]), func=np.log1p, inverse_func=np.expm1)   #> Fit on log(1+y), predict back in dollars.
poly = Pipeline([("pre", make_pre(poly=True)), ("m", Ridge())])            #> With squares and interactions of the numeric features.
for name, m in [("plain", base), ("log target", logt), ("poly numeric", poly)]:   #> Compare.
    m.fit(Xtr, ytr)                                                        #> Train.
    print(f"{name:<13} R2 = {m.score(Xte, yte):.4f}")                      #> Test R².
print("features with poly:", poly.named_steps["pre"].get_feature_names_out().size)   #> 3 numeric -> 9 (3 + 3 squares + 3 interactions), plus 5 categorical = 14.
```

The log target **lowers** R² here: the true effects in this data are additive in dollars (a fixed smoking surcharge), and the log makes them multiplicative. Transform the target because the problem calls for it, not by habit.

```py title="Exercise 3: the fake score grows with k"
from sklearn.feature_selection import SelectKBest, f_classif              #> Feature selection.
rng = np.random.default_rng(0)                                             #> Same noise data.
Xn, yn = rng.normal(size=(100, 5000)), rng.integers(0, 2, 100)             #> Pure noise.
for k in [5, 20, 200]:                                                     #> Different numbers of selected features.
    wrong = cross_val_score(LogisticRegression(max_iter=2000), SelectKBest(f_classif, k=k).fit_transform(Xn, yn), yn, cv=5).mean()   #> Leaky.
    right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=k), LogisticRegression(max_iter=2000)), Xn, yn, cv=5).mean()   #> Honest.
    print(f"k={k:<4} leaky={wrong:.2f} honest={right:.2f}")                #> Compare.
```

## [[ch:sklearn-regression]]: Regression

```py title="Exercise 2: the smoker x BMI interaction"
d = ins.copy()                                                             #> Work on a copy.
d["bmi"] = d["bmi"].fillna(d["bmi"].median())                              #> Simple imputation for this demo.
d["smoker_bmi"] = (d["smoker"] == "yes") * d["bmi"]                        #> Interaction: BMI for smokers, 0 for non-smokers.
d["obese_smoker"] = ((d["smoker"] == "yes") & (d["bmi"] >= 30)).astype(int)   #> A threshold interaction.
Xd, yd = d.drop(columns="charges"), d["charges"]                           #> Features and target.
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=42)   #> The chapter's split.
cat = ["sex", "smoker", "region"]                                          #> Categorical columns.
for extra in [[], ["smoker_bmi", "obese_smoker"]]:                         #> Without and with the new features.
    num = ["age", "bmi", "children"] + extra                               #> Numeric columns.
    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(drop="if_binary"), cat)])   #> Preprocessing.
    m = Pipeline([("pre", pre), ("m", Ridge(alpha=1.0))]).fit(Xtr[num + cat], ytr)   #> Fit.
    print("extra features:", extra or "none", "-> RMSE", round(root_mean_squared_error(yte, m.predict(Xte[num + cat]))))   #> Test RMSE.
```

In this dataset the linear model already beat the random forest ([[ch:sklearn-regression]]'s table), and the interactions barely change it, because the data were generated with no smoker × BMI interaction. On real data with genuine interactions, features like these often close the gap to tree models.

```py title="Exercises 1 and 3: log-target Ridge, and tree depth vs error" caption="Training error falls steadily with depth; test error bottoms out around depth 3-4 and then rises: overfitting starts there."
pre = ColumnTransformer([("num", StandardScaler(), ["age", "bmi", "children"]), ("cat", OneHotEncoder(drop="if_binary"), cat)])   #> Shared preprocessing.
cols = ["age", "bmi", "children"] + cat                                    #> Columns used.
plain = Pipeline([("pre", pre), ("m", Ridge())]).fit(Xtr[cols], ytr)       #> Plain Ridge.
logged = TransformedTargetRegressor(regressor=Pipeline([("pre", pre), ("m", Ridge())]), func=np.log1p, inverse_func=np.expm1).fit(Xtr[cols], ytr)   #> Log target.
print("Ex 1 RMSE plain:", round(root_mean_squared_error(yte, plain.predict(Xte[cols]))), "| log target:", round(root_mean_squared_error(yte, logged.predict(Xte[cols]))))   #> Compare.
depths = range(1, 16)                                                      #> Depths to try.
tr_e, te_e = [], []                                                        #> Errors.
for dpt in depths:                                                         #> Each depth.
    t = Pipeline([("pre", pre), ("m", DecisionTreeRegressor(max_depth=dpt, random_state=0))]).fit(Xtr[cols], ytr)   #> Fit a tree.
    tr_e.append(root_mean_squared_error(ytr, t.predict(Xtr[cols])))        #> Training RMSE.
    te_e.append(root_mean_squared_error(yte, t.predict(Xte[cols])))        #> Test RMSE.
fig, ax = plt.subplots(figsize=(7, 3.4))                                   #> New figure.
ax.plot(depths, tr_e, marker="o", label="train"); ax.plot(depths, te_e, marker="o", label="test")   #> Both curves.
ax.set(xlabel="max_depth", ylabel="RMSE", title="Decision tree: depth vs error"); ax.legend(); plt.show()   #> Labels and draw.
print("best test depth:", list(depths)[int(np.argmin(te_e))])              #> Where test error is lowest.
```

## [[ch:sklearn-classification]]: Classification

```py title="Exercise 1: a 98%-recall threshold from cross-validated probabilities"
bc = load_breast_cancer()                                                  #> Data.
Xb, yb = bc.data, 1 - bc.target                                            #> 1 = malignant, as in the chapter.
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, stratify=yb, random_state=0)   #> Split.
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))   #> Model.
oof = cross_val_predict(clf, Xtr, ytr, cv=5, method="predict_proba")[:, 1]   #> Out-of-fold probabilities on training data.
ths = np.linspace(0.01, 0.99, 99)                                          #> Candidate thresholds.
rec = np.array([recall_score(ytr, (oof >= t).astype(int)) for t in ths])   #> Training (OOF) recall at each.
t98 = ths[rec >= 0.98].max()                                               #> Highest threshold still reaching 98% recall.
p = (clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1] >= t98).astype(int)        #> Apply it to the test set.
print(f"threshold={t98:.2f}  test recall={recall_score(yte, p):.3f}  test precision={precision_score(yte, p):.3f}")   #> Report.
```

```py title="Exercise 2: boosting with class weights"
Xi, yi = make_classification(n_samples=4000, n_features=20, n_informative=5, weights=[0.97, 0.03], flip_y=0.01, random_state=1)   #> Same data as the chapter.
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, stratify=yi, test_size=0.3, random_state=1)   #> Split.
for cw in [None, "balanced"]:                                              #> Without and with class weights.
    pr = HistGradientBoostingClassifier(class_weight=cw, random_state=0).fit(Xtr, ytr).predict(Xte)   #> Fit and predict.
    print(f"class_weight={cw}: recall={recall_score(yte, pr):.3f} precision={precision_score(yte, pr, zero_division=0):.3f} bal_acc={balanced_accuracy_score(yte, pr):.3f}")   #> Compare.
```

```py title="Exercise 3: three ROC curves on one axes" caption="ROC curves of three classifiers on the breast-cancer test set, drawn on the same axes for direct comparison."
from sklearn.metrics import RocCurveDisplay                                #> ROC plotting.
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, stratify=yb, random_state=0)   #> Split.
fig, ax = plt.subplots(figsize=(5.5, 4.6))                                 #> One axes for all curves.
for name, m in [("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),   #> Three models...
                ("kNN", make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
                ("RandomForest", RandomForestClassifier(random_state=0))]:
    RocCurveDisplay.from_estimator(m.fit(Xtr, ytr), Xte, yte, name=name, ax=ax)   #> ...each drawn on the same ax.
ax.plot([0, 1], [0, 1], ls=":", color="grey"); plt.show()                  #> Chance line and draw.
```

## [[ch:sklearn-model-selection]]: Model selection

```py title="Exercises 1-3: tune kNN, coefficient stability, 5- vs 10-fold"
gs = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()),   #> Scaled kNN...
                  {"kneighborsclassifier__n_neighbors": range(1, 31), "kneighborsclassifier__weights": ["uniform", "distance"]},   #> ...60 combinations.
                  cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy", n_jobs=-1).fit(Xb, yb)   #> Search.
print("Ex 1:", gs.best_params_, round(gs.best_score_, 4))                  #> Best settings.
res = cross_validate(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xb, yb, cv=5, return_estimator=True)   #> Keep each fold's fitted model.
coefs = np.array([e[-1].coef_[0] for e in res["estimator"]])               #> e[-1] is the last pipeline step; one row per fold.
cv_ = pd.DataFrame({"mean": coefs.mean(0), "sd": coefs.std(0)}, index=bc.feature_names)   #> Mean and SD per feature.
print("Ex 2 (most variable coefficients):\n", cv_.sort_values("sd", ascending=False).head(4).round(3))   #> The least stable ones.
for k in [5, 10]:                                                          #> Fold counts.
    s = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xb, yb, cv=StratifiedKFold(k, shuffle=True, random_state=0))   #> CV scores.
    print(f"Ex 3: {k}-fold mean={s.mean():.4f} sd={s.std():.4f}")          #> Similar means; 10-fold scores vary more (smaller test folds).
```

## [[ch:sklearn-unsupervised]]: Unsupervised learning

```py title="Exercise 1: customer segments"
from sklearn.cluster import KMeans                                         #> k-means.
d = ins.dropna()[["age", "bmi", "charges"]]                                #> Three numeric features.
labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(d))   #> Cluster the scaled data.
prof = d.groupby(labels).mean().round(1)                                   #> Profile: mean of each feature per cluster.
prof["n"] = np.bincount(labels)                                            #> Cluster sizes.
prof["smoker_share"] = ins.dropna().groupby(labels)["smoker"].apply(lambda s: (s == "yes").mean()).round(2)   #> Who's in each cluster?
print(prof)                                                                #> One high-charge cluster that is mostly smokers.
```

```py title="Exercises 2 and 3: PCA of wines; t-SNE perplexity" caption="Left: wine cultivars separate well on the first two principal components. Right three: t-SNE maps of the digits at perplexity 5, 30 and 100; the layout changes but the digit groups persist."
from sklearn.decomposition import PCA                                      #> PCA.
from sklearn.manifold import TSNE                                          #> t-SNE.
Z = make_pipeline(StandardScaler(), PCA(2)).fit_transform(Xw)              #> Wine -> 2 components.
dg = load_digits()                                                         #> Digits.
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))                          #> Four panels.
axes[0].scatter(Z[:, 0], Z[:, 1], c=yw, cmap="viridis", s=12); axes[0].set(title="PCA of wines", xlabel="PC1", ylabel="PC2")   #> PCA panel.
for ax, perp in zip(axes[1:], [5, 30, 100]):                               #> Three perplexities.
    emb = TSNE(perplexity=perp, init="pca", random_state=0).fit_transform(dg.data[:800])   #> 800 digits keep it quick.
    ax.scatter(emb[:, 0], emb[:, 1], c=dg.target[:800], cmap="tab10", s=4); ax.set(title=f"t-SNE perplexity {perp}", xticks=[], yticks=[])   #> Plot.
plt.tight_layout(); plt.show()                                             #> Draw.
```

## [[ch:sklearn-interpret]]: Interpreting and saving models

```py title="Exercises 1-3: MAE-based importance, ICE for children, save and reload" caption="ICE lines for children are nearly parallel: the model gives every customer roughly the same increase per child."
from sklearn.inspection import permutation_importance, PartialDependenceDisplay   #> Interpretation tools.
import joblib                                                              #> Saving models.
X = ins.drop(columns="charges").astype({"age": float, "children": float})  #> Floats for partial dependence.
y = ins["charges"]                                                         #> Target.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)   #> Split.
pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), ["age", "bmi", "children"]),   #> Impute numeric...
                         ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), ["sex", "smoker", "region"])])   #> ...encode categorical.
rf = Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1))]).fit(Xtr, ytr)   #> Fit.
pi = permutation_importance(rf, Xte, yte, scoring="neg_mean_absolute_error", n_repeats=10, random_state=0, n_jobs=-1)   #> MAE-based importance.
print("Ex 1:", pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).round(0).to_dict())   #> Same ranking as with R²: smoker, age, bmi...
fig, ax = plt.subplots(figsize=(6, 3.6))                                   #> New figure.
PartialDependenceDisplay.from_estimator(rf, Xte, ["children"], kind="both", subsample=60, random_state=0, ax=ax)   #> ICE + PDP.
plt.show()                                                                 #> Draw.
joblib.dump(rf, "rf.joblib")                                               #> Save.
reloaded = joblib.load("rf.joblib")                                        #> In a new session you'd start here.
new = pd.DataFrame([{"age": 33.0, "sex": "male", "bmi": 27.5, "children": 1.0, "smoker": "no", "region": "northwest"}])   #> A new customer.
print("Ex 3 prediction:", reloaded.predict(new).round(0))                  #> Works with raw input.
```

## [[ch:tf-basics]]: TensorFlow fundamentals

```py title="Exercises 1-3: minimise a function, two features, a tf.data pipeline"
import os                                                                  #> Environment variables.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                                   #> Quiet TensorFlow.
import tensorflow as tf                                                    #> TensorFlow.
x = tf.Variable(0.0)                                                       #> Start at 0.
for _ in range(100):                                                       #> 100 steps.
    with tf.GradientTape() as tape:
        f = (x - 4.0) ** 2 + 1.0                                           #> The function.
    x.assign_sub(0.1 * tape.gradient(f, x))                                #> x -= 0.1 * df/dx.
print("Ex 1: x =", round(float(x), 4), " f(x) =", round(float((x - 4) ** 2 + 1), 4))   #> Converges to x = 4, f = 1.
rng = np.random.default_rng(0)                                             #> Data for exercise 2.
X2 = rng.uniform(-1, 1, (300, 2)).astype("float32")                        #> Two features.
y2 = (1.5 * X2[:, 0] - 2.0 * X2[:, 1] + 0.5).astype("float32")             #> True weights 1.5, -2.0; intercept 0.5.
W, b = tf.Variable(tf.zeros(2)), tf.Variable(0.0)                          #> Two weights and a bias.
opt = tf.keras.optimizers.SGD(0.2)                                         #> Optimiser.
for _ in range(300):                                                       #> Training steps.
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((tf.linalg.matvec(X2, W) + b - y2) ** 2)     #> matvec computes X2 @ W.
    opt.apply_gradients(zip(tape.gradient(loss, [W, b]), [W, b]))          #> Update.
print("Ex 2: W =", W.numpy().round(3), " b =", round(float(b), 3))         #> Recovers the true values.
Xb32 = StandardScaler().fit_transform(load_breast_cancer().data).astype("float32")   #> Scaled features.
ds = tf.data.Dataset.from_tensor_slices((Xb32, load_breast_cancer().target)).shuffle(569, seed=0).batch(64)   #> Shuffled batches of 64.
print("Ex 3:", [tuple(xb.shape) for xb, _ in ds.take(2)], "batches:", len(list(ds)))   #> (64, 30) batches; ceil(569/64) = 9.
```

## [[ch:keras-mlp]]: Neural networks with Keras

```py title="Exercises 1-3: width, batch normalisation, five seeds"
from tensorflow import keras                                               #> Keras.
from tensorflow.keras import layers                                        #> Layers.
Xtr, Xte, ytr, yte = train_test_split(load_breast_cancer().data, load_breast_cancer().target, test_size=0.2, stratify=load_breast_cancer().target, random_state=0)   #> Split.
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr).astype("float32"), sc.transform(Xte).astype("float32")   #> Scale.
def mlp(hidden, bn=False, seed=0):
    keras.utils.set_random_seed(seed)                                      #> Reproducible.
    m = keras.Sequential([keras.Input(shape=(30,))])                       #> Input layer.
    for u in hidden:                                                       #> Hidden layers.
        m.add(layers.Dense(u, activation="relu"))                          #> Dense layer.
        if bn: m.add(layers.BatchNormalization())                          #> Optional batch normalisation.
    m.add(layers.Dense(1, activation="sigmoid"))                           #> Output.
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])   #> Compile.
    m.fit(Xtr, ytr, epochs=60, batch_size=32, verbose=0, validation_split=0.15,     #> Train...
          callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])   #> ...with early stopping.
    return m.evaluate(Xte, yte, verbose=0)[1]                              #> Test accuracy.
for h in [[8], [256], [32, 16]]:                                           #> Exercise 1: different widths.
    print("Ex 1 hidden", h, "acc", round(mlp(h), 4))                       #> All similar: this problem is easy.
print("Ex 2 with BatchNorm:", round(mlp([32, 16], bn=True), 4))            #> Similar accuracy; BN mainly speeds up training of deeper nets.
accs = [mlp([32, 16], seed=s) for s in range(5)]                           #> Exercise 3: five seeds.
print(f"Ex 3: {np.mean(accs):.4f} ± {np.std(accs):.4f}")                   #> Report mean ± SD.
```

## [[ch:keras-advanced]]: CNNs and sequence models

```py title="Exercise 1: augmentation in the CNN"
dg = load_digits()                                                         #> Digits.
Xd = (dg.images / 16.0).astype("float32")[..., None]                       #> (1797, 8, 8, 1).
Xtr, Xte, ytr, yte = train_test_split(Xd, dg.target, test_size=0.2, stratify=dg.target, random_state=0)   #> Split.
def cnn(augment):
    keras.utils.set_random_seed(0)                                         #> Reproducible.
    aug = [layers.RandomRotation(0.05), layers.RandomTranslation(0.1, 0.1)] if augment else []   #> Optional augmentation layers.
    m = keras.Sequential([keras.Input(shape=(8, 8, 1)), *aug,              #> Augmentation goes first; active only during training.
                          layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
                          layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
                          layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dense(10, activation="softmax")])   #> The chapter's CNN.
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])   #> Compile.
    m.fit(Xtr, ytr, epochs=30, batch_size=64, verbose=0, validation_split=0.15,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])   #> Train.
    return m.evaluate(Xte, yte, verbose=0)[1]                              #> Test accuracy.
print("without augmentation:", round(cnn(False), 4), "| with:", round(cnn(True), 4))   #> On tiny 8x8 images the gain is small; on real photos it is often large.
```

```py title="Exercises 2 and 3: GRU, window length, multi-step forecasts"
t = np.arange(1200, dtype="float32")                                       #> Same series as the chapter.
series = (np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 17) + 0.001 * t + np.random.default_rng(0).normal(0, 0.15, t.size)).astype("float32")
mu, sd_ = series[:1000].mean(), series[:1000].std(); s = (series - mu) / sd_   #> Normalise with training statistics.
def windows(arr, w, h):
    X = np.stack([arr[i:i + w] for i in range(len(arr) - w - h + 1)])      #> Input windows.
    Y = np.stack([arr[i + w:i + w + h] for i in range(len(arr) - w - h + 1)])   #> The next h values.
    return X, Y                                                            #> Arrays.
for w, h, cell in [(10, 1, "GRU"), (40, 1, "GRU"), (80, 1, "GRU"), (40, 5, "GRU")]:   #> Exercise 2 (windows) and 3 (h = 5).
    Xa, Ya = windows(s[:1000], w, h); Xb_, Yb_ = windows(s[1000 - w:], w, h)   #> Train and test windows (chronological).
    keras.utils.set_random_seed(0)                                         #> Reproducible.
    m = keras.Sequential([keras.Input(shape=(w,)), layers.Reshape((w, 1)), layers.GRU(32), layers.Dense(h)])   #> GRU, h outputs.
    m.compile(optimizer="adam", loss="mse"); m.fit(Xa, Ya, epochs=20, batch_size=32, verbose=0)   #> Train.
    rmse = np.sqrt(np.mean((m.predict(Xb_, verbose=0) - Yb_) ** 2, axis=0)) * sd_   #> RMSE per forecast step, original units.
    print(f"{cell} window={w:<3} horizon={h}: test RMSE per step = {rmse.round(3)}")   #> For h = 5, error grows with the horizon.
```

## [[ch:research-protocol]]: The research protocol

```py title="Exercises 1 and 2: boosting as the primary model; nested CV instead of a test set"
bc = load_breast_cancer(); Xb, yb = bc.data, 1 - bc.target                 #> Data, 1 = malignant.
outer = StratifiedKFold(5, shuffle=True, random_state=2026)                #> Outer loop: evaluation.
inner = StratifiedKFold(5, shuffle=True, random_state=1)                   #> Inner loop: tuning.
hgb = GridSearchCV(HistGradientBoostingClassifier(random_state=0),         #> Tuned boosting...
                   {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}, cv=inner, scoring="roc_auc", n_jobs=-1)
lr = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear")),   #> ...vs tuned logistic regression.
                  {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=inner, scoring="roc_auc", n_jobs=-1)
for name, est in [("HistGB", hgb), ("LogReg", lr)]:                        #> Nested CV for each.
    sc_ = cross_val_score(est, Xb, yb, cv=outer, scoring="roc_auc")        #> Each outer fold runs a full grid search.
    print(f"{name}: nested CV ROC AUC = {sc_.mean():.4f} ± {sc_.std():.4f}")   #> Unbiased estimates.
```

Both models land within about one standard deviation of each other, so the conclusion doesn't change: the simpler, interpretable logistic regression is a defensible choice. Nested CV uses all 569 cases for evaluation, which gives a tighter estimate than a single 143-case test set.

**Exercise 3 (Methods paragraph).** A model answer, filled from the [[ch:research-protocol]] run: *"We used the Wisconsin Diagnostic Breast Cancer dataset (569 masses, 30 features; 37% malignant). Data were split, stratified by outcome, into training (n = 426) and test (n = 143) sets (seed 2026). Six candidate pipelines (dummy, logistic regression, kNN, RBF-SVM, random forest, histogram gradient boosting) were compared by 5 × 5 repeated stratified cross-validation on the training set, using ROC AUC as the primary metric. All preprocessing (standardisation) was embedded in scikit-learn pipelines. L1/L2-regularised logistic regression was tuned by grid search over C ∈ [10⁻³, 10²]. The decision threshold was fixed on out-of-fold training predictions to target ≥ 95% sensitivity. The final model was evaluated once on the test set; 95% confidence intervals were obtained from 2,000 bootstrap resamples. Analyses used Python 3.13 and scikit-learn 1.9."*

## [[ch:research-choosing]]: Comparing models

```py title="Exercises 1-3: corrected t-test on wines, bootstrap PR AUC, false-positive simulation"
from scipy import stats                                                    #> Statistics.
from sklearn.model_selection import RepeatedStratifiedKFold                #> Repeated CV.
from sklearn.svm import SVC                                                #> SVM.
from sklearn.metrics import average_precision_score                        #> PR AUC.
def corrected_ttest(a, b, n_train, n_test):
    d = a - b; var = d.var(ddof=1) * (1 / len(d) + n_test / n_train)       #> Nadeau-Bengio variance.
    t = d.mean() / np.sqrt(var); return d.mean(), 2 * stats.t.sf(abs(t), len(d) - 1)   #> Difference and p-value.
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)     #> 50 shared splits.
a = cross_val_score(make_pipeline(StandardScaler(), SVC()), Xw, yw, cv=cv)                          #> SVM accuracy.
b = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xw, yw, cv=cv)   #> LogReg accuracy.
diff, p = corrected_ttest(a, b, int(len(yw) * 0.9), len(yw) - int(len(yw) * 0.9))   #> 10-fold sizes.
print(f"Ex 1: SVC - LogReg = {diff:+.4f}, corrected p = {p:.3f}")          #> No evidence of a difference.
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=1)   #> Breast-cancer split.
pa = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]   #> Model A.
pb = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]   #> Model B.
rng = np.random.default_rng(0); diffs = []                                 #> Bootstrap.
for _ in range(2000):
    i = rng.integers(0, len(yte), len(yte))                                #> Resample test cases.
    if yte[i].min() != yte[i].max():                                       #> Need both classes.
        diffs.append(average_precision_score(yte[i], pa[i]) - average_precision_score(yte[i], pb[i]))   #> Paired difference.
print("Ex 2: PR AUC difference 95% CI:", np.percentile(diffs, [2.5, 97.5]).round(4))   #> Includes 0?
hits = 0                                                                   #> Exercise 3: simulation.
for rep in range(200):                                                     #> 200 simulated comparisons.
    s1 = 0.85 + rng.normal(0, 0.02, 10)                                    #> Model 1: 10 scores around a true accuracy of 0.85.
    s2 = s1 + rng.normal(0, 0.01, 10)                                      #> Model 2: the same true accuracy; independent small differences.
    hits += stats.ttest_rel(s1, s2).pvalue < 0.05                          #> Naive paired test.
print(f"Ex 3: naive test 'significant' in {hits / 2:.1f}% of 200 null comparisons")   #> Close to the nominal 5% in this simple setting.
```

In this simple simulation the scores are independent across splits, so the naive test happens to behave. Real CV scores share training data across folds, which makes their differences correlated; that is what makes the naive test overconfident in [[ch:research-choosing]] (p ≈ 0.0003 naive vs 0.28 corrected).

## [[ch:research-reporting]]: Reproducibility and reporting

```py title="Exercise 2: permutation test for the insurance regression"
from sklearn.model_selection import permutation_test_score                 #> Label-shuffling test.
d = ins.dropna()                                                           #> Complete rows.
pre = ColumnTransformer([("num", StandardScaler(), ["age", "bmi", "children"]), ("cat", OneHotEncoder(drop="if_binary"), ["sex", "smoker", "region"])])   #> Preprocessing.
score, perm, p = permutation_test_score(Pipeline([("pre", pre), ("m", Ridge())]), d.drop(columns="charges"), d["charges"],   #> Pipeline, X, y...
                                        scoring="r2", cv=5, n_permutations=50, random_state=0, n_jobs=-1)   #> ...50 shuffles.
print(f"real R2={score:.3f} | shuffled R2 mean={perm.mean():.3f} | p={p:.3f}")   #> Shuffled scores sit near 0 (slightly negative): no leakage.
```

**Exercise 1** is the `run_experiment.py` pattern from [[ch:research-reporting]] with the [[ch:research-protocol]] steps inside `main()`: put the seed, CV settings, candidate models and grid in the config, and write the metrics JSON, the fitted model and the figures into `results/run_<date>/`. **Exercise 3** is personal: every item you cannot fill in yet is a task for your project plan.

## [[ch:debug-strategy]]: How to debug

**Exercise 1.** In Jupyter, run `df.groupby("Region")["charges"].mean()` on the insurance data (note the capital R). After the `KeyError`, run `%debug` in a new cell; at the `(Pdb)` prompt type `u` until you reach your own frame, then `p df.columns.tolist()` to see the real name, `region`. Type `q` to quit. In a script, run `python -m pdb script.py`, type `c` to run until the crash, and inspect the same way.

**Exercise 2.** A good minimal reproducible example: a few lines of fake data built in the code (no files), only the imports needed, the single failing call, the full error message, and your versions (`python -c "import sklearn; print(sklearn.__version__)"`).

**Exercise 3.** Copy the `validate()` function from [[ch:debug-strategy]], edit the `expected` schema to match your columns and dtypes, add range checks for your variables, and call it immediately after every `pd.read_csv`.

## [[ch:debug-encyclopedia]]: The error encyclopedia

```py title="Three errors caused on purpose, then fixed"
from sklearn.linear_model import LinearRegression                         #> For error 1.
df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.1, 5.9]})            #> Tiny data.
try:
    LinearRegression().fit(df["x"], df["y"])                               #> Error 1: 1-D X.
except ValueError as e:
    print("caused:", str(e).split("\n")[0])                                #> First line of the message.
print("fixed :", LinearRegression().fit(df[["x"]], df["y"]).coef_.round(3))   #> Double brackets give a 2-D X.
try:
    np.ones((100, 3)) * np.ones(100)                                       #> Error 2: broadcasting.
except ValueError as e:
    print("caused:", e)                                                    #> The message.
print("fixed :", (np.ones((100, 3)) * np.ones(100)[:, None]).shape)        #> A column vector broadcasts.
try:
    {"lr": 0.01}["batch_size"]                                             #> Error 3: KeyError.
except KeyError as e:
    print("caused: KeyError", e)                                           #> The missing key.
print("fixed :", {"lr": 0.01}.get("batch_size", 32))                       #> A default instead of a crash.
```

## [[ch:debug-models]]: When the model is wrong

```py title="Exercises 1 and 2: breaking the sanity check; a deliberate leak"
keras.utils.set_random_seed(0)                                             #> Reproducible.
Xs, ys = (load_digits().data[:32] / 16).astype("float32"), load_digits().target[:32]   #> 32 samples.
for lr_, loss in [(1e-2, "sparse_categorical_crossentropy"), (1e-6, "sparse_categorical_crossentropy"), (1e-2, "mse")]:   #> Correct; tiny learning rate; wrong loss.
    net = keras.Sequential([keras.Input(shape=(64,)), layers.Dense(128, activation="relu"), layers.Dense(10, activation="softmax")])   #> Same network.
    net.compile(optimizer=keras.optimizers.Adam(lr_), loss=loss, metrics=["accuracy"])   #> Vary LR and loss.
    try:
        acc = net.fit(Xs, ys, epochs=150, verbose=0).history["accuracy"][-1]   #> Final training accuracy.
        print(f"lr={lr_:g} loss={loss:<32} train acc on 32 samples = {acc:.3f}")   #> Only the first setting memorises.
    except ValueError as e:                                                #> MSE compares 10 outputs with 1 integer label...
        print(f"lr={lr_:g} loss={loss:<32} fails:", str(e).split(" for ")[0])   #> ...so Keras refuses: the shapes don't match.
d = ins.dropna().copy()                                                    #> Exercise 2.
d["leak"] = d["charges"] + np.random.default_rng(0).normal(0, 500, len(d))   #> A feature computed from the target!
Xtr, Xte, ytr, yte = train_test_split(d.drop(columns="charges"), d["charges"], test_size=0.25, random_state=0)   #> Split.
pre = ColumnTransformer([("num", "passthrough", ["age", "bmi", "children", "leak"]), ("cat", OneHotEncoder(drop="if_binary"), ["sex", "smoker", "region"])])   #> Includes the leak.
rf = Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))]).fit(Xtr, ytr)   #> Fit.
pi = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=0, n_jobs=-1)   #> Importances.
print("test R2 with leak:", round(rf.score(Xte, yte), 4))                  #> Suspiciously close to 1.
print(pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False).round(3).head(3).to_dict())   #> The leak dominates: a red flag.
```

**Exercise 3** is the checklist at the end of [[ch:debug-models]]: go through it item by item for your own project.
