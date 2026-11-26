# Appendix B: Glossary and further reading

## Glossary

| Term | Meaning |
|---|---|
| **Accuracy** | Share of predictions that are correct. Misleading under class imbalance. |
| **Activation function** | Non-linear function applied by a neural-network layer (ReLU, sigmoid, softmax, tanh). |
| **Autodiff (automatic differentiation)** | Computing exact gradients of code automatically; `tf.GradientTape`. |
| **Axis** | A dimension of an array. `axis=0` runs down rows, `axis=1` across columns. |
| **Baseline** | A trivial model (predict the mean or the majority class) that any useful model must beat. |
| **Batch / mini-batch** | Subset of samples used for one gradient update. |
| **Bias (statistical)** | Systematic error from overly simple assumptions; leads to underfitting. |
| **Bootstrap** | Resampling with replacement to estimate uncertainty, e.g. confidence intervals for a metric. |
| **Broadcasting** | NumPy's rule for combining arrays of different shapes without copying. |
| **Calibration** | Agreement between predicted probabilities and observed frequencies. |
| **Class imbalance** | One class is much rarer than others. |
| **Confusion matrix** | Table of true vs predicted classes (TP, FP, FN, TN). |
| **Copy-on-Write** | pandas 3 behaviour: derived DataFrames act as independent copies. |
| **Cross-validation (CV)** | Repeatedly splitting training data into fit/validation folds to estimate performance. |
| **Data leakage** | Information unavailable at prediction time (or from the test set) influencing training. |
| **DataFrame** | pandas' labelled 2-D table. |
| **Dimensionality reduction** | Representing data with fewer features (PCA, t-SNE, UMAP). |
| **Dropout** | Randomly disabling units during training to reduce overfitting. |
| **dtype** | The element type of an array or column (`int64`, `float32`, `str`, `bool`). |
| **Early stopping** | Halting training when validation performance stops improving. |
| **Embedding** | A learned dense vector representing a category, word or item. |
| **Ensemble** | Combining many models (random forest, boosting) for better predictions. |
| **Epoch** | One full pass over the training data. |
| **Estimator** | Any scikit-learn object with `fit`. |
| **F1 score** | Harmonic mean of precision and recall. |
| **Feature** | An input variable (column) used for prediction. |
| **Feature engineering** | Creating informative features from raw data. |
| **Generalisation** | Performance on new, unseen data. |
| **Gradient descent** | Iteratively updating parameters against the gradient of the loss. |
| **Hyperparameter** | A setting chosen before training (tree depth, `C`, learning rate). |
| **Imputation** | Filling in missing values. |
| **Learning curve** | Performance as a function of training-set size. |
| **Learning rate** | Step size of each gradient update. |
| **Logit** | Unnormalised score before sigmoid/softmax. |
| **Loss function** | The quantity a model minimises during training (MSE, cross-entropy). |
| **Nested CV** | Outer CV for evaluation, inner CV for tuning; gives an unbiased estimate after tuning. |
| **One-hot encoding** | One 0/1 column per category. |
| **Overfitting** | Fitting noise in the training data; poor generalisation. |
| **Parameter** | A value learned from data (weights, coefficients). |
| **Permutation importance** | Drop in score when a feature's values are shuffled. |
| **Pipeline** | Chained preprocessing + model fitted as one estimator, preventing leakage. |
| **Precision** | TP / (TP + FP): how often a positive prediction is right. |
| **Recall / sensitivity** | TP / (TP + FN): share of actual positives detected. |
| **Regularisation** | Penalising model complexity (L1, L2, dropout) to reduce overfitting. |
| **ROC AUC** | Probability that a random positive is ranked above a random negative. |
| **Seed / random_state** | Integer that fixes random number generation for reproducibility. |
| **Series** | pandas' labelled 1-D array (one column). |
| **Specificity** | TN / (TN + FP): share of actual negatives correctly identified. |
| **Stratification** | Keeping class proportions equal across splits. |
| **Target / label** | The variable to predict. |
| **Tensor** | n-dimensional array in TensorFlow. |
| **Test set** | Held-out data used once for final evaluation. |
| **Transformer (scikit-learn)** | Object with `fit`/`transform` (scalers, encoders). Not the same as the Transformer neural architecture. |
| **Underfitting** | Model too simple to capture the pattern. |
| **Validation set** | Data used to tune and select models, separate from the test set. |
| **Variance (statistical)** | Sensitivity of a model to the particular training sample; leads to overfitting. |
| **Vectorisation** | Replacing Python loops with whole-array operations. |

## Further reading

Official documentation is the most accurate and up-to-date reference for every library in this book:

- Python tutorial and library reference: [docs.python.org](https://docs.python.org/3/)
- NumPy user guide: [numpy.org/doc](https://numpy.org/doc/stable/)
- pandas user guide: [pandas.pydata.org/docs](https://pandas.pydata.org/docs/)
- Matplotlib tutorials and gallery: [matplotlib.org](https://matplotlib.org/stable/)
- Seaborn tutorial: [seaborn.pydata.org](https://seaborn.pydata.org/)
- scikit-learn user guide, including "Common pitfalls and recommended practices": [scikit-learn.org](https://scikit-learn.org/stable/user_guide.html)
- TensorFlow and Keras guides: [tensorflow.org](https://www.tensorflow.org/guide) and [keras.io](https://keras.io/guides/)

Classic books to go deeper: *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani; free online, with a Python edition); *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman); *Deep Learning with Python* (Chollet); *Python for Data Analysis* (McKinney, by pandas' creator).

## A suggested learning path

| Weeks | Focus | Chapters | Milestone |
|---|---|---|---|
| 1–2 | Python fundamentals | 1–7 | Write functions that clean a small text file |
| 3 | NumPy | 8–10 | Implement linear regression with gradient descent |
| 4–5 | pandas and visualization | 11–18 | Clean, summarise and plot a public dataset; export figures for a report |
| 6–8 | scikit-learn | 19–25 | A full pipeline with CV, tuning and interpretation |
| 9–10 | Deep learning | 26–28 | Train and evaluate a CNN on an image dataset |
| 11–12 | Research practice and debugging | 29–34 | Run the full protocol on your own data and write the Methods section |

---

[📚 Contents](README.md)
