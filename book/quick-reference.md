# Appendix A: Quick-reference sheets

Dense one-page summaries of the commands you will use most. Each links back to a chapter for the full explanation.

## Python essentials

| Task | Code |
|---|---|
| f-string formatting | `f"{x:.3f}"`, `f"{p:.1%}"`, `f"{n:,}"`, `f"{x=}"` |
| Conditional expression | `a if cond else b` |
| Loop with index / in parallel | `for i, v in enumerate(xs):` · `for a, b in zip(xs, ys):` |
| Comprehensions | `[f(x) for x in xs if c(x)]` · `{k: v for k, v in d.items()}` · `{x for x in xs}` |
| Unpacking | `a, b = b, a` · `first, *rest = xs` · `f(*args, **kwargs)` |
| Dictionary safety | `d.get(k, default)` · `k in d` · `d \| other` (merge) |
| Copies | `xs.copy()` · `copy.deepcopy(nested)` |
| Sorting | `sorted(xs, key=len, reverse=True)` · `xs.sort()` (in place, returns None) |
| Errors | `try: … except ValueError as e: … else: … finally: …` · `raise ValueError("msg")` |
| Files | `with open(p, encoding="utf-8") as f:` · `Path("a") / "b.csv"` · `json.dump/load` |
| Counting / grouping | `Counter(xs).most_common(3)` · `defaultdict(list)` |
| Timing | `time.perf_counter()` · `%timeit expr` (Jupyter) |
| Reproducibility | `random.seed(0)` |

## NumPy

| Task | Code |
|---|---|
| Create | `np.array(lst)` · `np.zeros((r, c))` · `np.ones(n)` · `np.arange(a, b, step)` · `np.linspace(a, b, n)` · `np.eye(n)` |
| Random (modern) | `rng = np.random.default_rng(0)` · `rng.normal(m, s, size)` · `rng.integers(lo, hi, size)` · `rng.choice(a, size)` · `rng.permutation(n)` |
| Inspect | `a.shape` · `a.ndim` · `a.dtype` · `a.size` |
| Reshape | `a.reshape(r, -1)` · `a.ravel()` · `a.T` · `a[:, None]` · `np.expand_dims(a, 0)` |
| Index | `a[i, j]` · `a[:, 1]` · `a[mask]` · `a[[0, 2]]` · `np.where(cond, x, y)` · `np.flatnonzero(cond)` |
| Combine | `np.concatenate([a, b], axis=0)` · `np.vstack` · `np.hstack` · `np.column_stack` · `np.stack` |
| Aggregate | `a.sum(axis=0)` · `a.mean(axis=1, keepdims=True)` · `np.median` · `np.percentile(a, [25, 75])` · `a.argmax()` |
| Missing | `np.isnan(a)` · `np.nanmean(a)` · `np.nan_to_num(a)` |
| Linear algebra | `A @ B` · `np.linalg.solve(A, b)` · `np.linalg.inv` · `np.linalg.eigh` · `np.linalg.svd` · `np.linalg.norm` |
| Sort / unique | `np.sort` · `np.argsort` · `np.unique(a, return_counts=True)` |
| Compare floats | `np.allclose(a, b)` · `np.isclose(a, b)` |
| Broadcasting rule | compare shapes from the right; each pair equal or 1 |

## pandas

| Task | Code |
|---|---|
| Read / write | `pd.read_csv(p, usecols=, dtype=, parse_dates=, na_values=)` · `df.to_csv(p, index=False)` · `pd.read_excel` · `pd.read_parquet` |
| Inspect | `df.head()` · `df.info()` · `df.describe()` · `df.shape` · `df.dtypes` · `df["c"].value_counts()` · `df.isna().sum()` |
| Select | `df["c"]` (Series) · `df[["a", "b"]]` (DataFrame) · `df.loc[rows, cols]` · `df.iloc[0:5, 0:3]` |
| Filter | `df[(df.a > 1) & (df.b == "x")]` · `df[df.c.isin([...])]` · `df.query("a > 1 and b == 'x'")` |
| Assign safely | `df.loc[cond, "col"] = v` (never chained) · `df.assign(new=lambda d: …)` |
| Clean | `df.drop_duplicates()` · `df.dropna(subset=[…])` · `df.fillna(v)` · `pd.to_numeric(s, errors="coerce")` · `pd.to_datetime(s, format=…)` |
| Text | `s.str.strip()` · `.str.lower()` · `.str.contains("x")` · `.str.replace(",", "")` · `.str.extract(r"(\d+)")` |
| Recode / bin | `s.map(dict)` · `s.replace(dict)` · `pd.cut(s, bins, labels=)` · `pd.qcut(s, 4)` · `np.select(conds, choices)` |
| Group | `df.groupby("g")["x"].mean()` · `.agg(n=("x", "size"), m=("x", "mean"))` · `.transform("mean")` |
| Reshape | `df.pivot_table(values=, index=, columns=, aggfunc=)` · `df.melt(id_vars=)` · `pd.crosstab(a, b)` |
| Combine | `pd.merge(a, b, on="k", how="left", validate="one_to_one")` · `pd.concat([a, b], ignore_index=True)` |
| Time series | `s.resample("ME").mean()` · `s.rolling(7).mean()` · `s.shift(1)` · `s.diff()` · `df["d"].dt.month` |
| To NumPy | `df.to_numpy()` · `s.to_numpy()` |

## Matplotlib / Seaborn

| Task | Code |
|---|---|
| Figure + axes | `fig, ax = plt.subplots(figsize=(7, 4))` · `fig, axes = plt.subplots(2, 3)` |
| Plot types | `ax.plot` · `ax.scatter` · `ax.bar` / `ax.barh` · `ax.hist(x, bins=30)` · `ax.boxplot` · `ax.imshow` + `fig.colorbar` |
| Labels | `ax.set(title=, xlabel=, ylabel=, xlim=, yscale="log")` · `ax.legend()` · `fig.suptitle()` |
| Layout | `fig.tight_layout()` · `layout="constrained"` · `plt.subplot_mosaic([...])` |
| Save | `fig.savefig("f.pdf")` · `fig.savefig("f.png", dpi=300, bbox_inches="tight")` (before `plt.show()`) |
| Seaborn | `sns.histplot` · `sns.kdeplot` · `sns.scatterplot(hue=)` · `sns.boxplot` · `sns.violinplot` · `sns.heatmap(annot=True)` · `sns.pairplot` · `sns.regplot` |

## scikit-learn

| Task | Code |
|---|---|
| Split | `train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)` |
| Core API | `est.fit(X, y)` · `est.predict(X)` · `est.predict_proba(X)[:, 1]` · `est.score(X, y)` · `tr.fit_transform(X_train)` · `tr.transform(X_test)` |
| Preprocess | `StandardScaler` · `MinMaxScaler` · `RobustScaler` · `OneHotEncoder(handle_unknown="ignore")` · `OrdinalEncoder` · `SimpleImputer` · `PolynomialFeatures` |
| Pipelines | `make_pipeline(...)` · `Pipeline([("name", step), ...])` · `ColumnTransformer([("num", pipe, cols), ...])` |
| Baselines | `DummyClassifier(strategy="most_frequent")` · `DummyRegressor()` |
| Regressors | `LinearRegression` · `Ridge` · `Lasso` · `ElasticNet` · `KNeighborsRegressor` · `SVR` · `RandomForestRegressor` · `HistGradientBoostingRegressor` |
| Classifiers | `LogisticRegression` · `KNeighborsClassifier` · `SVC` · `GaussianNB` · `DecisionTreeClassifier` · `RandomForestClassifier` · `HistGradientBoostingClassifier` |
| Cross-validation | `cross_val_score(est, X, y, cv=5, scoring=…)` · `cross_validate(…, return_train_score=True)` · `StratifiedKFold` · `GroupKFold` · `TimeSeriesSplit` · `RepeatedStratifiedKFold` |
| Tuning | `GridSearchCV(pipe, {"step__param": [...]}, cv=, scoring=)` · `RandomizedSearchCV(…, n_iter=)` · `.best_params_` · `.best_score_` · `.cv_results_` |
| Classification metrics | `accuracy_score` · `precision_score` · `recall_score` · `f1_score(average=)` · `roc_auc_score(y, proba)` · `average_precision_score` · `confusion_matrix` · `classification_report` |
| Regression metrics | `mean_absolute_error` · `root_mean_squared_error` · `r2_score` |
| Displays | `ConfusionMatrixDisplay` · `RocCurveDisplay` · `PrecisionRecallDisplay` · `PredictionErrorDisplay` · `LearningCurveDisplay` · `CalibrationDisplay` |
| Unsupervised | `KMeans(n_clusters, n_init=10)` · `DBSCAN(eps, min_samples)` · `PCA(n_components=0.95)` · `TSNE` · `silhouette_score` |
| Interpret | `permutation_importance` · `PartialDependenceDisplay` · `feature_importances_` · `coef_` |
| Save | `joblib.dump(pipe, "m.joblib")` · `joblib.load(…)` |

## TensorFlow / Keras

| Task | Code |
|---|---|
| Seed | `keras.utils.set_random_seed(0)` |
| Model | `keras.Sequential([keras.Input(shape=(p,)), layers.Dense(64, activation="relu"), layers.Dense(1, activation="sigmoid")])` |
| Compile | `model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=…, metrics=["accuracy"])` |
| Train | `model.fit(X, y, validation_data=(Xv, yv), epochs=100, batch_size=32, callbacks=[...])` |
| Evaluate / predict | `model.evaluate(X, y)` · `model.predict(X)` |
| Callbacks | `EarlyStopping(patience=10, restore_best_weights=True)` · `ReduceLROnPlateau` · `ModelCheckpoint` · `TerminateOnNaN` |
| Regularise | `layers.Dropout(0.3)` · `kernel_regularizer=keras.regularizers.l2(1e-4)` · `layers.BatchNormalization()` |
| Layers | `Dense` · `Conv2D` · `MaxPooling2D` · `Flatten` · `GlobalAveragePooling2D` · `LSTM` · `GRU` · `Conv1D` · `Embedding` · `Normalization` |
| Output ↔ loss | sigmoid + `binary_crossentropy` · softmax + `sparse_categorical_crossentropy` (int labels) · linear + `mse` |
| Data | `tf.data.Dataset.from_tensor_slices((X, y)).shuffle(n).batch(32).prefetch(tf.data.AUTOTUNE)` |
| Autodiff | `with tf.GradientTape() as tape: …` · `tape.gradient(loss, vars)` · `opt.apply_gradients(zip(g, vars))` |
| Save | `model.save("m.keras")` · `keras.models.load_model("m.keras")` |

---

[📚 Contents](README.md)
