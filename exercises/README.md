# Try it yourself: all exercises

Every chapter ends with a few exercises. They are collected here so you can practise without the book open. Try each one before looking at the solutions.

| Part | Solutions |
|---|---|
| I–IV (Chapters 1–18) | [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb) |
| V–VIII (Chapters 19–34) | [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb) |


## Part I: Python

### Chapter 1: Values, variables and types

Code: [01-python-basics.ipynb](../notebooks/part-1-python/01-python-basics.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Store the radius `r = 2.5` and print the circle's area to 3 decimal places using `math.pi` and an f-string.
2. Given `email = "  Ada.Lovelace@Example.COM "`, produce `"ada.lovelace@example.com"` in one line.
3. Predict, then check: `7 // -2`, `bool("False")`, `"3" * 3`.

### Chapter 2: Data structures: lists, tuples, dictionaries, sets

Code: [02-python-collections.ipynb](../notebooks/part-1-python/02-python-collections.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. From `words = ["Data", "science", "IS", "fun", "data"]`, build the set of lowercase unique words.
2. Build a dict mapping each number 1–10 to `"fizz"` if divisible by 3, else the number itself.
3. Explain why `t = ([1, 2], 3); t[0].append(9)` works even though tuples are immutable.

### Chapter 3: Control flow: decisions and loops

Code: [03-python-control.ipynb](../notebooks/part-1-python/03-python-control.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Loop over `range(1, 31)` and print "Fizz" for multiples of 3, "Buzz" for multiples of 5, "FizzBuzz" for both, else the number.
2. Use `zip` to compute the dot product of `[1, 2, 3]` and `[4, 5, 6]` in a loop. (Answer: 32.)
3. Write a `while` loop that halves `lr = 1.0` until it is below `0.01` and counts how many halvings it took.

### Chapter 4: Functions: reusable, testable logic

Code: [04-python-functions.ipynb](../notebooks/part-1-python/04-python-functions.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Write `def describe(values)` that returns a dict with `n`, `mean`, `min` and `max`.
2. Write a decorator `@count_calls` that prints how many times a function has been called.
3. Write a generator `def running_mean(xs)` that yields the mean of the items seen so far.

### Chapter 5: Classes and objects

Code: [05-python-oop.ipynb](../notebooks/part-1-python/05-python-oop.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Write a class `Counter` with methods `add(item)` and `most_common()`.
2. Give `Dataset` a `__len__` method so that `len(ds)` works.
3. Subclass `Model` as `MedianRegressor`.

### Chapter 6: Errors, files and modules

Code: [06-python-errors-files.ipynb](../notebooks/part-1-python/06-python-errors-files.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Write `read_numbers(path)` that returns a list of floats from a text file, skipping lines that can't be converted (catch `ValueError`).
2. Save a dict of hyperparameters to `config.json` and load it back.
3. Use `Path.glob("**/*.csv")` to count CSV files in a folder tree.

### Chapter 7: The Pythonic toolkit

Code: [07-python-toolkit.ipynb](../notebooks/part-1-python/07-python-toolkit.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Use `Counter` to find the three most common words in a paragraph (lowercase it and split on whitespace first).
2. Use `itertools.product` to list all 12 combinations of `C in [0.1, 1, 10]` and `kernel in ["linear", "rbf", "poly", "sigmoid"]`.
3. Write a regex that extracts all e-mail addresses from a string.


## Part II: NumPy

### Chapter 8: NumPy arrays: shape, dtype and indexing

Code: [08-numpy-arrays.ipynb](../notebooks/part-2-numpy/08-numpy-arrays.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Create a 5x5 array of the numbers 1–25 and extract its centre 3x3 block.
2. From `x = np.random.default_rng(0).normal(size=1000)`, compute the fraction of values more than 2 standard deviations from the mean (expect about 0.05).
3. Reshape `np.arange(24)` into shape `(2, 3, 4)` and compute the sum along each axis. Predict each result's shape first.

### Chapter 9: Computing with NumPy: vectorization, broadcasting, statistics, linear algebra

Code: [09-numpy-compute.ipynb](../notebooks/part-2-numpy/09-numpy-compute.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Min-max scale each column of a random `(100, 4)` matrix to [0, 1] using broadcasting.
2. Using `np.unique(..., return_counts=True)`, report the class proportions of `rng.choice([0, 1], 500, p=[.8, .2])`.
3. Solve the system 3x + 2y = 7, x − y = −1 with `np.linalg.solve`.

### Chapter 10: Machine learning from scratch with NumPy

Code: [10-numpy-ml-scratch.ipynb](../notebooks/part-2-numpy/10-numpy-ml-scratch.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Add L2 regularisation to `gradient_descent`: add `2 * alpha * w` to the gradient (excluding the intercept) and watch the slope shrink as `alpha` grows.
2. Change the logistic regression's class centres to (−0.5, −0.5) and (0.5, 0.5). What happens to accuracy, and why?
3. Make `knn_predict` work for more than two classes using `np.bincount` per row.


## Part III: Pandas

### Chapter 11: pandas foundations: Series, DataFrames and selection

Code: [11-pandas-basics.ipynb](../notebooks/part-3-pandas/11-pandas-basics.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. How many female smokers in the southeast are over 40?
2. Add a column `bmi_class` with "under" (< 18.5), "normal" (< 25), "over" (< 30), "obese" (≥ 30) using `pd.cut`.
3. Show the 5 youngest customers with 4 children, sorted by charges.

### Chapter 12: Cleaning data with pandas

Code: [12-pandas-cleaning.ipynb](../notebooks/part-3-pandas/12-pandas-cleaning.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. In `insurance.csv`, fill missing BMI with the median BMI of the same sex and smoker group (`groupby(...).transform("median")`).
2. Count how many charges are outliers by the IQR rule. Are they mostly smokers?
3. Write a check function that raises an error if any age is outside 18–64 or any charge is negative.

### Chapter 13: Grouping, reshaping, joining and time series

Code: [13-pandas-transform.ipynb](../notebooks/part-3-pandas/13-pandas-transform.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. For each region, find the share of customers who are smokers and the median charge for smokers vs non-smokers in one table.
2. Make a long-format table of `age`, `bmi` and `charges` with `melt`, then compute each variable's mean with `groupby`.
3. Resample `sales` to weekly sums and plot them.


## Part IV: Visualization

### Chapter 14: Matplotlib and Seaborn: from first plot to publication figure

Code: [14-viz-matplotlib.ipynb](../notebooks/part-4-visualization/14-viz-matplotlib.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Plot the histogram of `charges` and of `np.log(charges)` side by side. Which looks closer to normal?
2. Recreate the pair plot with `hue="region"`. Does region matter?
3. Make a bar chart of mean charges by number of children with error bars (`sns.barplot` shows a 95% CI by default).

### Chapter 15: The Matplotlib chart gallery

Code: [15-viz-mpl-gallery.ipynb](../notebooks/part-4-visualization/15-viz-mpl-gallery.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Draw a grouped bar chart of mean `charges` by `region` for smokers and non-smokers, using `insurance.csv`.
2. Make an ECDF of `bmi` for each sex on one axes. Which group has the higher median?
3. Save any chart from this chapter as a 600-dpi PNG and as a PDF, and compare the file sizes.

### Chapter 16: The Seaborn chart gallery

Code: [16-viz-seaborn-gallery.ipynb](../notebooks/part-4-visualization/16-viz-seaborn-gallery.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Use `catplot(kind="violin")` to show `charges` by `children`, split into columns by `smoker`.
2. Make a `jointplot` of `bmi` vs `charges` with `kind="reg"`. What does the fitted line suggest?
3. Draw a `clustermap` of the insurance numeric columns (scaled) for 80 random customers, with smokers marked in the row colours.

### Chapter 17: Interactive charts with Plotly

Code: [17-viz-plotly.ipynb](../notebooks/part-4-visualization/17-viz-plotly.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Make an animated bar chart of the five most populous countries' life expectancy over time (`animation_frame="year"`).
2. Draw a sunburst of `tips` by `day`, then `time`, then `sex`, with `values="tip"`.
3. Export one chart as a 300-dpi-equivalent PNG for a 7-inch-wide figure (hint: `width=700, scale=3`).

### Chapter 18: Figures for papers, theses and assignments

Code: [18-viz-export.ipynb](../notebooks/part-4-visualization/18-viz-export.ipynb) · Solutions: [solutions-parts-1-4.ipynb](../solutions/solutions-parts-1-4.ipynb)

1. Rebuild one figure from [Chapter 15](../notebooks/part-4-visualization/15-viz-mpl-gallery.ipynb) at 3.5 inches wide with 8-pt fonts and save it with `save_figure()`.
2. Make a 7 x 3 inch, three-panel figure (a, b, c) from the insurance data and export it as PDF and PNG.
3. Open your PNG in an image viewer and confirm its pixel width equals inches × dpi.


## Part V: scikit-learn

### Chapter 19: Machine-learning concepts and the scikit-learn API

Code: [19-sklearn-concepts.ipynb](../notebooks/part-5-scikit-learn/19-sklearn-concepts.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Load `load_wine()` and compare `DummyClassifier`, `KNeighborsClassifier` and `RandomForestClassifier` on a stratified split.
2. In the polynomial demo, replace `LinearRegression` with `Ridge(alpha=1e-3)` for degree 15. What happens to the wiggles?
3. Re-run the iris example with five different `random_state` values. How much does test accuracy vary? (This is why we cross-validate.)

### Chapter 20: Preprocessing, feature engineering and pipelines

Code: [20-sklearn-preprocessing.ipynb](../notebooks/part-5-scikit-learn/20-sklearn-preprocessing.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Build a pipeline for the insurance data that log-transforms `charges` as the target with `TransformedTargetRegressor(regressor=..., func=np.log1p, inverse_func=np.expm1)`. Does test R² improve?
2. Add `PolynomialFeatures(2)` to the numeric branch. What happens to the number of features?
3. Repeat the leakage experiment with `k=5` and `k=200`. How does the fake score change?

### Chapter 21: Regression: predicting numbers

Code: [21-sklearn-regression.ipynb](../notebooks/part-5-scikit-learn/21-sklearn-regression.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Wrap the Ridge pipeline in `TransformedTargetRegressor` with a log transform and compare test RMSE.
2. Add an interaction feature `smoker × bmi` (hint: `PolynomialFeatures(interaction_only=True)` or a custom column) to the linear model. Does it close the gap to the random forest?
3. Plot test RMSE against `max_depth` from 1 to 15 for a single decision tree. Where does overfitting start?

### Chapter 22: Classification: predicting categories

Code: [22-sklearn-classification.ipynb](../notebooks/part-5-scikit-learn/22-sklearn-classification.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Find the threshold that gives at least 98% recall on the training data via cross-validated probabilities (`cross_val_predict(..., method="predict_proba")`) and report test precision at that threshold.
2. Repeat the imbalance experiment with `HistGradientBoostingClassifier(class_weight="balanced")`.
3. Plot ROC curves of three models on one axes by passing the same `ax` to each `RocCurveDisplay`.

### Chapter 23: Model selection: cross-validation and hyperparameter tuning

Code: [23-sklearn-model-selection.ipynb](../notebooks/part-5-scikit-learn/23-sklearn-model-selection.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Tune `n_neighbors` (1–30) and `weights` (`"uniform"`, `"distance"`) for kNN on the breast-cancer data with `GridSearchCV`.
2. Use `cross_validate(..., return_estimator=True)` to inspect how much logistic-regression coefficients vary across folds.
3. Compare 5-fold and 10-fold CV estimates and their standard deviations for the same model.

### Chapter 24: Unsupervised learning: clustering and dimensionality reduction

Code: [24-sklearn-unsupervised.ipynb](../notebooks/part-5-scikit-learn/24-sklearn-unsupervised.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Cluster the insurance customers (scaled `age`, `bmi`, `charges`) with k-means and profile each cluster with `groupby(labels).mean()`.
2. Run PCA on the wine dataset and plot the first two components coloured by cultivar.
3. Compare t-SNE maps of the digits with perplexity 5, 30 and 100.

### Chapter 25: Interpreting, explaining and saving models

Code: [25-sklearn-interpret.ipynb](../notebooks/part-5-scikit-learn/25-sklearn-interpret.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Compute permutation importance with `scoring="neg_mean_absolute_error"`. Does the ranking change?
2. Draw ICE plots for `children`. Is the effect the same for everyone?
3. Save a fitted pipeline, restart Python, load it, and predict for a brand-new customer DataFrame.


## Part VI: TensorFlow & Keras

### Chapter 26: TensorFlow fundamentals: tensors, gradients and data pipelines

Code: [26-tf-basics.ipynb](../notebooks/part-6-tensorflow-keras/26-tf-basics.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Use `GradientTape` to find the minimum of f(x) = (x − 4)² + 1 by gradient descent, starting from x = 0.
2. Extend the manual training loop to two input features.
3. Build a `tf.data` pipeline from the breast-cancer data (`load_breast_cancer`) with batches of 64.

### Chapter 27: Neural networks with Keras

Code: [27-keras-mlp.ipynb](../notebooks/part-6-tensorflow-keras/27-keras-mlp.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Replace the two hidden layers with one layer of 8 units. Does test accuracy change? What about 256 units?
2. Add `layers.BatchNormalization()` after each Dense layer and compare the training curves.
3. Wrap `build_model` in a loop over five seeds and report test accuracy as mean ± SD.

### Chapter 28: Deep learning architectures: CNNs, sequence models and custom training

Code: [28-keras-advanced.ipynb](../notebooks/part-6-tensorflow-keras/28-keras-advanced.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Add the `augment` block as the first layer of the CNN and compare test accuracy.
2. Replace `LSTM(32)` with `GRU(32)`, and try `window = 10` and `window = 80`.
3. Turn the forecaster into a 5-step-ahead model by changing the targets and using `Dense(5)`.


## Part VII: ML for Research

### Chapter 29: The ML research protocol: a complete worked study

Code: [29-research-protocol.ipynb](../notebooks/part-7-ml-for-research/29-research-protocol.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Re-run the protocol with `HistGradientBoostingClassifier` as the primary model. Is the conclusion different?
2. Replace the hold-out test set by nested cross-validation ([Chapter 23](../notebooks/part-5-scikit-learn/23-sklearn-model-selection.ipynb)) and compare the estimates.
3. Write the Methods paragraph for this study using the checklist in [Chapter 31](../notebooks/part-7-ml-for-research/31-research-reporting.ipynb).

### Chapter 30: Choosing models and metrics, and comparing them rigorously

Code: [30-research-choosing.ipynb](../notebooks/part-7-ml-for-research/30-research-choosing.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Use the corrected t-test to compare `SVC` and `LogisticRegression` on the wine dataset.
2. Run the paired bootstrap with `average_precision_score` instead of ROC AUC.
3. Simulate: draw two models with identical true accuracy and count how often the naive t-test on CV scores declares p < 0.05.

### Chapter 31: Reproducibility, leakage checklists and writing it up

Code: [31-research-reporting.ipynb](../notebooks/part-7-ml-for-research/31-research-reporting.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Turn the [Chapter 29](../notebooks/part-7-ml-for-research/29-research-protocol.ipynb) case study into a `run_experiment.py` with a config dictionary, and save its outputs to `results/`.
2. Run `permutation_test_score` on the insurance regression pipeline with `scoring="r2"`.
3. Fill in the Methods template for your own project, marking every item you cannot yet fill: those are your to-do list.


## Part VIII: Debugging

### Chapter 32: How to debug: reading tracebacks and a systematic method

Code: [32-debug-strategy.ipynb](../notebooks/part-8-debugging/32-debug-strategy.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Deliberately cause a `KeyError` in a pandas `groupby`, then use `%debug` (or `python -m pdb`) to inspect the DataFrame's columns at the failure point.
2. Write a minimal reproducible example for the last error you were stuck on.
3. Add a `validate()` function to your own project's loading code.

### Chapter 33: The error encyclopedia

Code: [33-debug-encyclopedia.ipynb](../notebooks/part-8-debugging/33-debug-encyclopedia.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

Pick any three errors from this chapter, reproduce them in your own environment, then fix each one. Being able to *cause* an error on purpose is the fastest way to recognise it in the wild.

### Chapter 34: When the code runs but the model is wrong

Code: [34-debug-models.ipynb](../notebooks/part-8-debugging/34-debug-models.ipynb) · Solutions: [solutions-parts-5-8.ipynb](../solutions/solutions-parts-5-8.ipynb)

1. Break the "overfit 32 samples" check on purpose (learning rate 1e-6, or the wrong loss) and watch it fail.
2. Introduce a leak into the insurance pipeline (add a feature equal to `charges` plus noise) and see how the permutation importance and test score react.
3. Apply the pre-submission checklist to your own current project.
