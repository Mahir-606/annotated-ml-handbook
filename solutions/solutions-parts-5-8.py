"""The Annotated ML Handbook: Solutions: Parts V–VIII. Companion script by Mahir Absar Khan."""

# %%
# Repository setup (not part of the book's examples): run this cell first.
import os, sys
from pathlib import Path

# Files the examples create (CSV, models, figures) go into ./outputs next to the notebook.
OUT = Path.cwd() if Path.cwd().name == "outputs" else Path.cwd() / "outputs"
OUT.mkdir(exist_ok=True)
os.chdir(OUT)

# insurance.csv is the practice dataset created in Chapter 11 and reused by later chapters.
# If it is not here yet, copy it from the repository's data/ folder (or download it in Colab).
if not Path("insurance.csv").exists():
    import shutil, urllib.request
    for base in OUT.parents:
        if (base / "data" / "insurance.csv").exists():
            shutil.copy(base / "data" / "insurance.csv", "insurance.csv")
            break
    else:
        try:
            urllib.request.urlretrieve("https://raw.githubusercontent.com/Mahir-606/"
                                       "annotated-ml-handbook/main/data/insurance.csv", "insurance.csv")
        except Exception as e:
            print("Could not fetch insurance.csv; run Chapter 11 first.", e)

# The figure style used in the book.
import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams.update({
    "figure.figsize": (7.2, 4.0), "figure.dpi": 100,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler(color=["#1f4fff", "#e0245e", "#0f9b62", "#ff6a00",
                                     "#7b3ff2", "#0091b3", "#d99a00", "#d42a2a"]),
    "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlelocation": "left",
    "font.size": 10.5, "legend.frameon": False, "image.cmap": "viridis",
})

# Plotly: interactive in Jupyter/Colab; tools/run_all.py stores static PNGs so GitHub can show them.
try:
    import plotly.io as pio
    pio.renderers.default = os.environ.get("PLOTLY_RENDERER") or (
        "colab" if "google.colab" in sys.modules else "notebook_connected")
except ImportError:
    pass


# %% [markdown]
# ## Shared set-up · *Beginner*
#
# **Imports used throughout this appendix**


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_digits, make_classification
from sklearn.model_selection import (train_test_split, cross_val_score, cross_validate,
                                     cross_val_predict, GridSearchCV, StratifiedKFold, KFold)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import root_mean_squared_error, recall_score, precision_score, balanced_accuracy_score
ins = pd.read_csv("insurance.csv")
print("ready", ins.shape)


# %% [markdown]
# ## Chapter 19: ML concepts and the scikit-learn API
#
# **Exercise 1: three models on the wine data**


# %%
Xw, yw = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xw, yw, stratify=yw, random_state=0)
for name, m in [("Dummy", DummyClassifier()), ("kNN", make_pipeline(StandardScaler(), KNeighborsClassifier())),
                ("RandomForest", RandomForestClassifier(random_state=0))]:
    print(f"{name:<13}", round(m.fit(Xtr, ytr).score(Xte, yte), 3))


# %% [markdown]
# **Exercise 2: Ridge tames a degree-15 polynomial**


# %%
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(1)
x = np.sort(rng.uniform(0, 1, 40)); y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 40)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.4, random_state=3)
grid = np.linspace(0, 1, 300).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.scatter(x_tr, y_tr, s=14, color="k")
for name, reg in [("LinearRegression", LinearRegression()), ("Ridge(alpha=1e-3)", Ridge(alpha=1e-3))]:
    m = make_pipeline(PolynomialFeatures(15), StandardScaler(), reg).fit(x_tr.reshape(-1, 1), y_tr)
    ax.plot(grid, m.predict(grid), label=f"{name}: test MSE {np.mean((m.predict(x_te.reshape(-1, 1)) - y_te) ** 2):.3f}")
ax.set_ylim(-2, 2); ax.legend(fontsize=8); plt.show()


# %% [markdown]
# **Exercise 3: accuracy depends on the split**


# %%
Xi, yi = load_iris(return_X_y=True)
accs = [KNeighborsClassifier().fit(a, c).score(b, d)
        for a, b, c, d in (train_test_split(Xi, yi, test_size=0.25, stratify=yi, random_state=s) for s in range(5))]
print(np.round(accs, 3), "range:", round(max(accs) - min(accs), 3))


# %% [markdown]
# ## Chapter 20: Preprocessing and pipelines
#
# **Exercises 1 and 2: log target and polynomial features**


# %%
X, y = ins.drop(columns="charges"), ins["charges"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
def make_pre(poly=False):
    steps = [SimpleImputer(strategy="median"), StandardScaler()] + ([PolynomialFeatures(2, include_bias=False)] if poly else [])
    return ColumnTransformer([("num", make_pipeline(*steps), ["age", "bmi", "children"]),
                              ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), ["sex", "smoker", "region"])])
base = Pipeline([("pre", make_pre()), ("m", Ridge())])
logt = TransformedTargetRegressor(regressor=Pipeline([("pre", make_pre()), ("m", Ridge())]), func=np.log1p, inverse_func=np.expm1)
poly = Pipeline([("pre", make_pre(poly=True)), ("m", Ridge())])
for name, m in [("plain", base), ("log target", logt), ("poly numeric", poly)]:
    m.fit(Xtr, ytr)
    print(f"{name:<13} R2 = {m.score(Xte, yte):.4f}")
print("features with poly:", poly.named_steps["pre"].get_feature_names_out().size)


# %% [markdown]
# **Exercise 3: the fake score grows with k**


# %%
from sklearn.feature_selection import SelectKBest, f_classif
rng = np.random.default_rng(0)
Xn, yn = rng.normal(size=(100, 5000)), rng.integers(0, 2, 100)
for k in [5, 20, 200]:
    wrong = cross_val_score(LogisticRegression(max_iter=2000), SelectKBest(f_classif, k=k).fit_transform(Xn, yn), yn, cv=5).mean()
    right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=k), LogisticRegression(max_iter=2000)), Xn, yn, cv=5).mean()
    print(f"k={k:<4} leaky={wrong:.2f} honest={right:.2f}")


# %% [markdown]
# ## Chapter 21: Regression
#
# **Exercise 2: the smoker x BMI interaction**


# %%
d = ins.copy()
d["bmi"] = d["bmi"].fillna(d["bmi"].median())
d["smoker_bmi"] = (d["smoker"] == "yes") * d["bmi"]
d["obese_smoker"] = ((d["smoker"] == "yes") & (d["bmi"] >= 30)).astype(int)
Xd, yd = d.drop(columns="charges"), d["charges"]
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=42)
cat = ["sex", "smoker", "region"]
for extra in [[], ["smoker_bmi", "obese_smoker"]]:
    num = ["age", "bmi", "children"] + extra
    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(drop="if_binary"), cat)])
    m = Pipeline([("pre", pre), ("m", Ridge(alpha=1.0))]).fit(Xtr[num + cat], ytr)
    print("extra features:", extra or "none", "-> RMSE", round(root_mean_squared_error(yte, m.predict(Xte[num + cat]))))


# %% [markdown]
# **Exercises 1 and 3: log-target Ridge, and tree depth vs error**


# %%
pre = ColumnTransformer([("num", StandardScaler(), ["age", "bmi", "children"]), ("cat", OneHotEncoder(drop="if_binary"), cat)])
cols = ["age", "bmi", "children"] + cat
plain = Pipeline([("pre", pre), ("m", Ridge())]).fit(Xtr[cols], ytr)
logged = TransformedTargetRegressor(regressor=Pipeline([("pre", pre), ("m", Ridge())]), func=np.log1p, inverse_func=np.expm1).fit(Xtr[cols], ytr)
print("Ex 1 RMSE plain:", round(root_mean_squared_error(yte, plain.predict(Xte[cols]))), "| log target:", round(root_mean_squared_error(yte, logged.predict(Xte[cols]))))
depths = range(1, 16)
tr_e, te_e = [], []
for dpt in depths:
    t = Pipeline([("pre", pre), ("m", DecisionTreeRegressor(max_depth=dpt, random_state=0))]).fit(Xtr[cols], ytr)
    tr_e.append(root_mean_squared_error(ytr, t.predict(Xtr[cols])))
    te_e.append(root_mean_squared_error(yte, t.predict(Xte[cols])))
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(depths, tr_e, marker="o", label="train"); ax.plot(depths, te_e, marker="o", label="test")
ax.set(xlabel="max_depth", ylabel="RMSE", title="Decision tree: depth vs error"); ax.legend(); plt.show()
print("best test depth:", list(depths)[int(np.argmin(te_e))])


# %% [markdown]
# ## Chapter 22: Classification
#
# **Exercise 1: a 98%-recall threshold from cross-validated probabilities**


# %%
bc = load_breast_cancer()
Xb, yb = bc.data, 1 - bc.target
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, stratify=yb, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
oof = cross_val_predict(clf, Xtr, ytr, cv=5, method="predict_proba")[:, 1]
ths = np.linspace(0.01, 0.99, 99)
rec = np.array([recall_score(ytr, (oof >= t).astype(int)) for t in ths])
t98 = ths[rec >= 0.98].max()
p = (clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1] >= t98).astype(int)
print(f"threshold={t98:.2f}  test recall={recall_score(yte, p):.3f}  test precision={precision_score(yte, p):.3f}")


# %% [markdown]
# **Exercise 2: boosting with class weights**


# %%
Xi, yi = make_classification(n_samples=4000, n_features=20, n_informative=5, weights=[0.97, 0.03], flip_y=0.01, random_state=1)
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, stratify=yi, test_size=0.3, random_state=1)
for cw in [None, "balanced"]:
    pr = HistGradientBoostingClassifier(class_weight=cw, random_state=0).fit(Xtr, ytr).predict(Xte)
    print(f"class_weight={cw}: recall={recall_score(yte, pr):.3f} precision={precision_score(yte, pr, zero_division=0):.3f} bal_acc={balanced_accuracy_score(yte, pr):.3f}")


# %% [markdown]
# **Exercise 3: three ROC curves on one axes**


# %%
from sklearn.metrics import RocCurveDisplay
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, stratify=yb, random_state=0)
fig, ax = plt.subplots(figsize=(5.5, 4.6))
for name, m in [("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                ("kNN", make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
                ("RandomForest", RandomForestClassifier(random_state=0))]:
    RocCurveDisplay.from_estimator(m.fit(Xtr, ytr), Xte, yte, name=name, ax=ax)
ax.plot([0, 1], [0, 1], ls=":", color="grey"); plt.show()


# %% [markdown]
# ## Chapter 23: Model selection
#
# **Exercises 1-3: tune kNN, coefficient stability, 5- vs 10-fold**


# %%
gs = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()),
                  {"kneighborsclassifier__n_neighbors": range(1, 31), "kneighborsclassifier__weights": ["uniform", "distance"]},
                  cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy", n_jobs=-1).fit(Xb, yb)
print("Ex 1:", gs.best_params_, round(gs.best_score_, 4))
res = cross_validate(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xb, yb, cv=5, return_estimator=True)
coefs = np.array([e[-1].coef_[0] for e in res["estimator"]])
cv_ = pd.DataFrame({"mean": coefs.mean(0), "sd": coefs.std(0)}, index=bc.feature_names)
print("Ex 2 (most variable coefficients):\n", cv_.sort_values("sd", ascending=False).head(4).round(3))
for k in [5, 10]:
    s = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xb, yb, cv=StratifiedKFold(k, shuffle=True, random_state=0))
    print(f"Ex 3: {k}-fold mean={s.mean():.4f} sd={s.std():.4f}")


# %% [markdown]
# ## Chapter 24: Unsupervised learning
#
# **Exercise 1: customer segments**


# %%
from sklearn.cluster import KMeans
d = ins.dropna()[["age", "bmi", "charges"]]
labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(d))
prof = d.groupby(labels).mean().round(1)
prof["n"] = np.bincount(labels)
prof["smoker_share"] = ins.dropna().groupby(labels)["smoker"].apply(lambda s: (s == "yes").mean()).round(2)
print(prof)


# %% [markdown]
# **Exercises 2 and 3: PCA of wines; t-SNE perplexity**


# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
Z = make_pipeline(StandardScaler(), PCA(2)).fit_transform(Xw)
dg = load_digits()
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
axes[0].scatter(Z[:, 0], Z[:, 1], c=yw, cmap="viridis", s=12); axes[0].set(title="PCA of wines", xlabel="PC1", ylabel="PC2")
for ax, perp in zip(axes[1:], [5, 30, 100]):
    emb = TSNE(perplexity=perp, init="pca", random_state=0).fit_transform(dg.data[:800])
    ax.scatter(emb[:, 0], emb[:, 1], c=dg.target[:800], cmap="tab10", s=4); ax.set(title=f"t-SNE perplexity {perp}", xticks=[], yticks=[])
plt.tight_layout(); plt.show()


# %% [markdown]
# ## Chapter 25: Interpreting and saving models
#
# **Exercises 1-3: MAE-based importance, ICE for children, save and reload**


# %%
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import joblib
X = ins.drop(columns="charges").astype({"age": float, "children": float})
y = ins["charges"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), ["age", "bmi", "children"]),
                         ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), ["sex", "smoker", "region"])])
rf = Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1))]).fit(Xtr, ytr)
pi = permutation_importance(rf, Xte, yte, scoring="neg_mean_absolute_error", n_repeats=10, random_state=0, n_jobs=-1)
print("Ex 1:", pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).round(0).to_dict())
fig, ax = plt.subplots(figsize=(6, 3.6))
PartialDependenceDisplay.from_estimator(rf, Xte, ["children"], kind="both", subsample=60, random_state=0, ax=ax)
plt.show()
joblib.dump(rf, "rf.joblib")
reloaded = joblib.load("rf.joblib")
new = pd.DataFrame([{"age": 33.0, "sex": "male", "bmi": 27.5, "children": 1.0, "smoker": "no", "region": "northwest"}])
print("Ex 3 prediction:", reloaded.predict(new).round(0))


# %% [markdown]
# ## Chapter 26: TensorFlow fundamentals
#
# **Exercises 1-3: minimise a function, two features, a tf.data pipeline**


# %%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
x = tf.Variable(0.0)
for _ in range(100):
    with tf.GradientTape() as tape:
        f = (x - 4.0) ** 2 + 1.0
    x.assign_sub(0.1 * tape.gradient(f, x))
print("Ex 1: x =", round(float(x), 4), " f(x) =", round(float((x - 4) ** 2 + 1), 4))
rng = np.random.default_rng(0)
X2 = rng.uniform(-1, 1, (300, 2)).astype("float32")
y2 = (1.5 * X2[:, 0] - 2.0 * X2[:, 1] + 0.5).astype("float32")
W, b = tf.Variable(tf.zeros(2)), tf.Variable(0.0)
opt = tf.keras.optimizers.SGD(0.2)
for _ in range(300):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((tf.linalg.matvec(X2, W) + b - y2) ** 2)
    opt.apply_gradients(zip(tape.gradient(loss, [W, b]), [W, b]))
print("Ex 2: W =", W.numpy().round(3), " b =", round(float(b), 3))
Xb32 = StandardScaler().fit_transform(load_breast_cancer().data).astype("float32")
ds = tf.data.Dataset.from_tensor_slices((Xb32, load_breast_cancer().target)).shuffle(569, seed=0).batch(64)
print("Ex 3:", [tuple(xb.shape) for xb, _ in ds.take(2)], "batches:", len(list(ds)))


# %% [markdown]
# ## Chapter 27: Neural networks with Keras
#
# **Exercises 1-3: width, batch normalisation, five seeds**


# %%
from tensorflow import keras
from tensorflow.keras import layers
Xtr, Xte, ytr, yte = train_test_split(load_breast_cancer().data, load_breast_cancer().target, test_size=0.2, stratify=load_breast_cancer().target, random_state=0)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr).astype("float32"), sc.transform(Xte).astype("float32")
def mlp(hidden, bn=False, seed=0):
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([keras.Input(shape=(30,))])
    for u in hidden:
        m.add(layers.Dense(u, activation="relu"))
        if bn: m.add(layers.BatchNormalization())
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    m.fit(Xtr, ytr, epochs=60, batch_size=32, verbose=0, validation_split=0.15,
          callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return m.evaluate(Xte, yte, verbose=0)[1]
for h in [[8], [256], [32, 16]]:
    print("Ex 1 hidden", h, "acc", round(mlp(h), 4))
print("Ex 2 with BatchNorm:", round(mlp([32, 16], bn=True), 4))
accs = [mlp([32, 16], seed=s) for s in range(5)]
print(f"Ex 3: {np.mean(accs):.4f} ± {np.std(accs):.4f}")


# %% [markdown]
# ## Chapter 28: CNNs and sequence models
#
# **Exercise 1: augmentation in the CNN**


# %%
dg = load_digits()
Xd = (dg.images / 16.0).astype("float32")[..., None]
Xtr, Xte, ytr, yte = train_test_split(Xd, dg.target, test_size=0.2, stratify=dg.target, random_state=0)
def cnn(augment):
    keras.utils.set_random_seed(0)
    aug = [layers.RandomRotation(0.05), layers.RandomTranslation(0.1, 0.1)] if augment else []
    m = keras.Sequential([keras.Input(shape=(8, 8, 1)), *aug,
                          layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
                          layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
                          layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dense(10, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(Xtr, ytr, epochs=30, batch_size=64, verbose=0, validation_split=0.15,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    return m.evaluate(Xte, yte, verbose=0)[1]
print("without augmentation:", round(cnn(False), 4), "| with:", round(cnn(True), 4))


# %% [markdown]
# **Exercises 2 and 3: GRU, window length, multi-step forecasts**


# %%
t = np.arange(1200, dtype="float32")
series = (np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 17) + 0.001 * t + np.random.default_rng(0).normal(0, 0.15, t.size)).astype("float32")
mu, sd_ = series[:1000].mean(), series[:1000].std(); s = (series - mu) / sd_
def windows(arr, w, h):
    X = np.stack([arr[i:i + w] for i in range(len(arr) - w - h + 1)])
    Y = np.stack([arr[i + w:i + w + h] for i in range(len(arr) - w - h + 1)])
    return X, Y
for w, h, cell in [(10, 1, "GRU"), (40, 1, "GRU"), (80, 1, "GRU"), (40, 5, "GRU")]:
    Xa, Ya = windows(s[:1000], w, h); Xb_, Yb_ = windows(s[1000 - w:], w, h)
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.Input(shape=(w,)), layers.Reshape((w, 1)), layers.GRU(32), layers.Dense(h)])
    m.compile(optimizer="adam", loss="mse"); m.fit(Xa, Ya, epochs=20, batch_size=32, verbose=0)
    rmse = np.sqrt(np.mean((m.predict(Xb_, verbose=0) - Yb_) ** 2, axis=0)) * sd_
    print(f"{cell} window={w:<3} horizon={h}: test RMSE per step = {rmse.round(3)}")


# %% [markdown]
# ## Chapter 29: The research protocol
#
# **Exercises 1 and 2: boosting as the primary model; nested CV instead of a test set**


# %%
bc = load_breast_cancer(); Xb, yb = bc.data, 1 - bc.target
outer = StratifiedKFold(5, shuffle=True, random_state=2026)
inner = StratifiedKFold(5, shuffle=True, random_state=1)
hgb = GridSearchCV(HistGradientBoostingClassifier(random_state=0),
                   {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}, cv=inner, scoring="roc_auc", n_jobs=-1)
lr = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear")),
                  {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=inner, scoring="roc_auc", n_jobs=-1)
for name, est in [("HistGB", hgb), ("LogReg", lr)]:
    sc_ = cross_val_score(est, Xb, yb, cv=outer, scoring="roc_auc")
    print(f"{name}: nested CV ROC AUC = {sc_.mean():.4f} ± {sc_.std():.4f}")


# %% [markdown]
# ## Chapter 30: Comparing models
#
# **Exercises 1-3: corrected t-test on wines, bootstrap PR AUC, false-positive simulation**


# %%
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
def corrected_ttest(a, b, n_train, n_test):
    d = a - b; var = d.var(ddof=1) * (1 / len(d) + n_test / n_train)
    t = d.mean() / np.sqrt(var); return d.mean(), 2 * stats.t.sf(abs(t), len(d) - 1)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
a = cross_val_score(make_pipeline(StandardScaler(), SVC()), Xw, yw, cv=cv)
b = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xw, yw, cv=cv)
diff, p = corrected_ttest(a, b, int(len(yw) * 0.9), len(yw) - int(len(yw) * 0.9))
print(f"Ex 1: SVC - LogReg = {diff:+.4f}, corrected p = {p:.3f}")
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=1)
pa = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
pb = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
rng = np.random.default_rng(0); diffs = []
for _ in range(2000):
    i = rng.integers(0, len(yte), len(yte))
    if yte[i].min() != yte[i].max():
        diffs.append(average_precision_score(yte[i], pa[i]) - average_precision_score(yte[i], pb[i]))
print("Ex 2: PR AUC difference 95% CI:", np.percentile(diffs, [2.5, 97.5]).round(4))
hits = 0
for rep in range(200):
    s1 = 0.85 + rng.normal(0, 0.02, 10)
    s2 = s1 + rng.normal(0, 0.01, 10)
    hits += stats.ttest_rel(s1, s2).pvalue < 0.05
print(f"Ex 3: naive test 'significant' in {hits / 2:.1f}% of 200 null comparisons")


# %% [markdown]
# ## Chapter 31: Reproducibility and reporting
#
# **Exercise 2: permutation test for the insurance regression**


# %%
from sklearn.model_selection import permutation_test_score
d = ins.dropna()
pre = ColumnTransformer([("num", StandardScaler(), ["age", "bmi", "children"]), ("cat", OneHotEncoder(drop="if_binary"), ["sex", "smoker", "region"])])
score, perm, p = permutation_test_score(Pipeline([("pre", pre), ("m", Ridge())]), d.drop(columns="charges"), d["charges"],
                                        scoring="r2", cv=5, n_permutations=50, random_state=0, n_jobs=-1)
print(f"real R2={score:.3f} | shuffled R2 mean={perm.mean():.3f} | p={p:.3f}")


# %% [markdown]
# ## Chapter 32: How to debug
#
# ## Chapter 33: The error encyclopedia
#
# **Three errors caused on purpose, then fixed**


# %%
from sklearn.linear_model import LinearRegression
df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.1, 5.9]})
try:
    LinearRegression().fit(df["x"], df["y"])
except ValueError as e:
    print("caused:", str(e).split("\n")[0])
print("fixed :", LinearRegression().fit(df[["x"]], df["y"]).coef_.round(3))
try:
    np.ones((100, 3)) * np.ones(100)
except ValueError as e:
    print("caused:", e)
print("fixed :", (np.ones((100, 3)) * np.ones(100)[:, None]).shape)
try:
    {"lr": 0.01}["batch_size"]
except KeyError as e:
    print("caused: KeyError", e)
print("fixed :", {"lr": 0.01}.get("batch_size", 32))


# %% [markdown]
# ## Chapter 34: When the model is wrong
#
# **Exercises 1 and 2: breaking the sanity check; a deliberate leak**


# %%
keras.utils.set_random_seed(0)
Xs, ys = (load_digits().data[:32] / 16).astype("float32"), load_digits().target[:32]
for lr_, loss in [(1e-2, "sparse_categorical_crossentropy"), (1e-6, "sparse_categorical_crossentropy"), (1e-2, "mse")]:
    net = keras.Sequential([keras.Input(shape=(64,)), layers.Dense(128, activation="relu"), layers.Dense(10, activation="softmax")])
    net.compile(optimizer=keras.optimizers.Adam(lr_), loss=loss, metrics=["accuracy"])
    try:
        acc = net.fit(Xs, ys, epochs=150, verbose=0).history["accuracy"][-1]
        print(f"lr={lr_:g} loss={loss:<32} train acc on 32 samples = {acc:.3f}")
    except ValueError as e:
        print(f"lr={lr_:g} loss={loss:<32} fails:", str(e).split(" for ")[0])
d = ins.dropna().copy()
d["leak"] = d["charges"] + np.random.default_rng(0).normal(0, 500, len(d))
Xtr, Xte, ytr, yte = train_test_split(d.drop(columns="charges"), d["charges"], test_size=0.25, random_state=0)
pre = ColumnTransformer([("num", "passthrough", ["age", "bmi", "children", "leak"]), ("cat", OneHotEncoder(drop="if_binary"), ["sex", "smoker", "region"])])
rf = Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))]).fit(Xtr, ytr)
pi = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=0, n_jobs=-1)
print("test R2 with leak:", round(rf.score(Xte, yte), 4))
print(pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False).round(3).head(3).to_dict())
