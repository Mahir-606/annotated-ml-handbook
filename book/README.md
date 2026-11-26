# The Annotated ML Handbook

**by Mahir Absar Khan** · First edition, 2026

*For every learner who has stared at a traceback at 2 a.m., sighed, and tried one more time.*

Read it here on GitHub, chapter by chapter (the notebooks show the text, the code, the real outputs and every figure), open any chapter in Colab to run it, or read the designed [web edition](https://mahir-606.github.io/annotated-ml-handbook/) and download the [PDF](https://mahir-606.github.io/annotated-ml-handbook/The-Annotated-ML-Handbook.pdf).

## Front matter

- [Preface](preface.md)
- [Acknowledgements](acknowledgements.md)
- [How to use this book](how-to-use-this-book.md)


## Part I: Python

*The language itself: values, data structures, control flow, functions, classes, errors, files, and the Pythonic toolkit every ML codebase is built from.*

1. **[Values, variables and types](../notebooks/part-1-python/01-python-basics.ipynb)**  
   <sub>Variables: names for values · Numbers and arithmetic · Strings: working with text · Booleans, comparisons and logic · Converting between types · `None`: the value for "nothing" · Comments, input and getting help</sub>
2. **[Data structures: lists, tuples, dictionaries, sets](../notebooks/part-1-python/02-python-collections.ipynb)**  
   <sub>Lists · Tuples · Dictionaries · Sets · Comprehensions · Mutability, references and copies</sub>
3. **[Control flow: decisions and loops](../notebooks/part-1-python/03-python-control.ipynb)**  
   <sub>if, elif, else · for loops · while loops, break and continue · match: structural pattern matching · Iterables, iterators and why loops are lazy</sub>
4. **[Functions: reusable, testable logic](../notebooks/part-1-python/04-python-functions.ipynb)**  
   <sub>Defining and calling functions · Default values, *args and **kwargs · Scope: where names live · Lambda functions and functions as values · Closures and decorators · Generators: producing values lazily · Type hints and docstrings</sub>
5. **[Classes and objects](../notebooks/part-1-python/05-python-oop.ipynb)**  
   <sub>Your first class · Special ("dunder") methods · Inheritance · Properties, class attributes and static methods · Dataclasses: classes without boilerplate</sub>
6. **[Errors, files and modules](../notebooks/part-1-python/06-python-errors-files.ipynb)**  
   <sub>Exceptions and try/except · Reading a traceback · Working with files · Modules, packages and imports · Logging instead of print</sub>
7. **[The Pythonic toolkit](../notebooks/part-1-python/07-python-toolkit.ipynb)**  
   <sub>collections: smarter containers · itertools: loops without the loops · functools: partial and reduce · Context managers · Randomness and reproducibility · Dates and times · Regular expressions · Measuring speed</sub>

## Part II: NumPy

*Fast n-dimensional arrays: shapes, indexing, vectorization, broadcasting, linear algebra and random numbers. This is the maths engine under pandas, scikit-learn and TensorFlow.*

8. **[NumPy arrays: shape, dtype and indexing](../notebooks/part-2-numpy/08-numpy-arrays.ipynb)**  
   <sub>Why arrays instead of lists? · Creating arrays · The three attributes you must always know · Reshaping · Indexing and slicing · Views vs copies · Combining and splitting arrays · Understanding axes</sub>
9. **[Computing with NumPy: vectorization, broadcasting, statistics, linear algebra](../notebooks/part-2-numpy/09-numpy-compute.ipynb)**  
   <sub>Universal functions (ufuncs) · Broadcasting · Statistics and aggregation · Sorting, searching, unique values · Missing values: NaN · Random numbers · Linear algebra</sub>
10. **[Machine learning from scratch with NumPy](../notebooks/part-2-numpy/10-numpy-ml-scratch.ipynb)**  
   <sub>The core idea of supervised learning · Making a dataset and splitting it · Linear regression, the closed-form way · Linear regression by gradient descent · Why feature scaling matters · Logistic regression from scratch · Softmax and one-hot encoding · k-nearest neighbours in five lines</sub>

## Part III: Pandas

*Tables you can reason about: loading, inspecting, selecting, cleaning, reshaping, joining and summarising real-world data before it ever meets a model.*

11. **[pandas foundations: Series, DataFrames and selection](../notebooks/part-3-pandas/11-pandas-basics.ipynb)**  
   <sub>Series and DataFrame · The practice dataset · Reading and writing data · Inspecting a DataFrame · Selecting data · Adding, changing and removing columns · Copy-on-Write and the "chained assignment" trap · The index · From pandas to NumPy and back</sub>
12. **[Cleaning data with pandas](../notebooks/part-3-pandas/12-pandas-cleaning.ipynb)**  
   <sub>A deliberately messy dataset · Step 1: tidy column names · Step 2: duplicates · Step 3: fix types · Step 4: clean text and categories · Step 5: missing values · Step 6: outliers and impossible values · Transforming columns with map, apply and vectorized ops · Categorical dtype · A reusable cleaning function</sub>
13. **[Grouping, reshaping, joining and time series](../notebooks/part-3-pandas/13-pandas-transform.ipynb)**  
   <sub>groupby: split, apply, combine · Pivot tables and crosstabs · Reshaping: wide ↔ long · Combining tables · Time series · Method chaining: a full mini-analysis</sub>

## Part IV: Visualization

*Seeing your data and your models: Matplotlib's figure anatomy, Seaborn's statistical plots, choosing the right chart, and exporting publication-quality figures.*

14. **[Matplotlib and Seaborn: from first plot to publication figure](../notebooks/part-4-visualization/14-viz-matplotlib.ipynb)**  
   <sub>The anatomy of a figure · The essential chart types · Subplots, shared axes and layouts · Seaborn: statistical plots in one line · Plotting directly from pandas · Styles, colours and publication-quality export</sub>
15. **[The Matplotlib chart gallery](../notebooks/part-4-visualization/15-viz-mpl-gallery.ipynb)**  
   <sub>Set-up for the gallery · Charts for change over time · Charts for relationships · Charts for comparing categories · Charts for distributions · Charts for uncertainty · Charts for matrices and grids · Charts for direction, cycles and schedules · Combining charts: layouts, insets and annotations · Getting the figure out: PNG, PDF and SVG</sub>
16. **[The Seaborn chart gallery](../notebooks/part-4-visualization/16-viz-seaborn-gallery.ipynb)**  
   <sub>Set-up · Distribution plots · Relationship plots · Regression plots · Categorical plots · Matrix plots · Multi-variable plots · Styling Seaborn · Saving Seaborn figures</sub>
17. **[Interactive charts with Plotly](../notebooks/part-4-visualization/17-viz-plotly.ipynb)**  
   <sub>Set-up · Scatter and bubble charts · Line and area charts · Bar charts · Distributions · Relationships among many variables · Density and heatmaps · Parts of a whole · Flows: the Sankey diagram · 3-D charts · Maps · Customising and exporting Plotly charts</sub>
18. **[Figures for papers, theses and assignments](../notebooks/part-4-visualization/18-viz-export.ipynb)**  
   <sub>Raster vs vector: which format? · Size first, then resolution · Saving with `savefig`: every option that matters · Huge scatter plots: rasterise the points, keep the text sharp · Multi-panel figures with panel letters · One function to export everything · Exporting Seaborn and Plotly figures · Putting figures into documents · Writing a good figure caption · Colour and accessibility · Final-figure checklist</sub>

## Part V: scikit-learn

*Classical machine learning done properly: the estimator API, preprocessing pipelines, regression, classification, model selection, clustering, dimensionality reduction and interpretation.*

19. **[Machine-learning concepts and the scikit-learn API](../notebooks/part-5-scikit-learn/19-sklearn-concepts.ipynb)**  
   <sub>The kinds of machine learning · The workflow at a glance · The estimator API: fit, predict, transform · Baselines: is the model actually learning anything? · Overfitting, underfitting and the bias–variance trade-off · Why the test set is sacred · Reproducibility knobs</sub>
20. **[Preprocessing, feature engineering and pipelines](../notebooks/part-5-scikit-learn/20-sklearn-preprocessing.ipynb)**  
   <sub>Scaling numeric features · Encoding categorical features · Imputing missing values · Pipelines: preprocessing + model as one object · ColumnTransformer: different processing per column · Seeing leakage with your own eyes · Feature engineering · Inspecting and displaying pipelines</sub>
21. **[Regression: predicting numbers](../notebooks/part-5-scikit-learn/21-sklearn-regression.ipynb)**  
   <sub>Set-up: one preprocessor, many models · Regression metrics · Linear models · Non-linear models · Diagnostic plots: how is the model wrong? · Quantifying uncertainty in a test score</sub>
22. **[Classification: predicting categories](../notebooks/part-5-scikit-learn/22-sklearn-classification.ipynb)**  
   <sub>The data and a first model · The confusion matrix and the metrics built from it · Visual evaluation: confusion matrix, ROC and PR curves · Choosing the decision threshold · Comparing classifiers · Class imbalance · Probability calibration · Multiclass problems</sub>
23. **[Model selection: cross-validation and hyperparameter tuning](../notebooks/part-5-scikit-learn/23-sklearn-model-selection.ipynb)**  
   <sub>k-fold cross-validation · Choosing the right splitter · Hyperparameter tuning with GridSearchCV · RandomizedSearchCV and successive halving · Learning curves and validation curves · Nested cross-validation: an unbiased estimate after tuning</sub>
24. **[Unsupervised learning: clustering and dimensionality reduction](../notebooks/part-5-scikit-learn/24-sklearn-unsupervised.ipynb)**  
   <sub>k-means clustering · Other clustering algorithms · Principal component analysis (PCA) · t-SNE and UMAP for visualisation · Anomaly detection</sub>
25. **[Interpreting, explaining and saving models](../notebooks/part-5-scikit-learn/25-sklearn-interpret.ipynb)**  
   <sub>Set-up · Impurity-based feature importance · Permutation importance · Partial dependence and ICE plots · SHAP values · Saving and loading models</sub>

## Part VI: TensorFlow & Keras

*Deep learning from tensors and automatic differentiation up to dense, convolutional and recurrent networks, callbacks, custom training loops and saving models.*

26. **[TensorFlow fundamentals: tensors, gradients and data pipelines](../notebooks/part-6-tensorflow-keras/26-tf-basics.ipynb)**  
   <sub>Tensors · Operations · Variables and automatic differentiation · Speed: tf.function · Input pipelines with tf.data · Reproducibility</sub>
27. **[Neural networks with Keras](../notebooks/part-6-tensorflow-keras/27-keras-mlp.ipynb)**  
   <sub>The Keras workflow in five steps · Choosing the output layer and loss · Callbacks: early stopping, checkpoints, learning-rate schedules · Multiclass classification · Regression · The Functional API · Saving and loading</sub>
28. **[Deep learning architectures: CNNs, sequence models and custom training](../notebooks/part-6-tensorflow-keras/28-keras-advanced.ipynb)**  
   <sub>Convolutional neural networks · Transfer learning · Sequence models: forecasting a time series · Custom training loops · Other architectures in one table</sub>

## Part VII: ML for Research

*A step-by-step protocol for using ML in a research project: from question and data to validated, interpreted, reproducible results and a written Methods section.*

29. **[The ML research protocol: a complete worked study](../notebooks/part-7-ml-for-research/29-research-protocol.ipynb)**  
   <sub>The protocol at a glance · Step 1: define the question before touching the data · Step 2: load and document the data · Step 3: exploratory analysis · Step 4: split and lock away the test set · Steps 5–6: baselines and candidate models under repeated CV · Step 7: tune the chosen model · Step 8: the one-time test evaluation · Step 9: interpretation, with stability · Step 10: robustness checks · Step 11: make it reproducible</sub>
30. **[Choosing models and metrics, and comparing them rigorously](../notebooks/part-7-ml-for-research/30-research-choosing.ipynb)**  
   <sub>A model-selection map · Choosing the metric · Is model A really better than model B? · How much data do I need?</sub>
31. **[Reproducibility, leakage checklists and writing it up](../notebooks/part-7-ml-for-research/31-research-reporting.ipynb)**  
   <sub>A project layout that scales · A configuration-driven experiment script · Reproducibility checklist · The data-leakage checklist · What to report · A Methods-section template · Ethics and responsible use</sub>

## Part VIII: Debugging

*When you are stuck: how to read tracebacks, an encyclopedia of real error messages with causes and fixes, and a symptom-to-cause guide for models that misbehave.*

32. **[How to debug: reading tracebacks and a systematic method](../notebooks/part-8-debugging/32-debug-strategy.ipynb)**  
   <sub>The debugging loop · Anatomy of a traceback · Inspection tools · Shrinking the problem · Defensive coding for data science</sub>
33. **[The error encyclopedia](../notebooks/part-8-debugging/33-debug-encyclopedia.ipynb)**  
   <sub>Python errors · NumPy errors · pandas errors · scikit-learn errors · TensorFlow / Keras errors · Notebook and environment gotchas</sub>
34. **[When the code runs but the model is wrong](../notebooks/part-8-debugging/34-debug-models.ipynb)**  
   <sub>Reading training curves · Symptom → likely cause → what to try · Sanity checks that catch silent bugs · A pre-submission checklist</sub>

## Appendices

- **Appendix A** · [Quick-reference sheets](quick-reference.md)
- **Appendix B** · [Glossary and further reading](glossary.md)
- **Appendix C** · [Solutions: Parts I–IV](../solutions/solutions-parts-1-4.ipynb)
- **Appendix D** · [Solutions: Parts V–VIII](../solutions/solutions-parts-5-8.ipynb)
- [Index of definitions](index-of-definitions.md)
- [All exercises on one page](../exercises/README.md)

## About the author

**Mahir Absar Khan** is an industrial and production engineer and a Research Assistant at the Computational Intelligence and Operations Laboratory. He researches machine learning and LLMs in supply chains, manufacturing, and healthcare, and has co-authored papers at ACM, ICML, IJCAI, and AAAI venues.
