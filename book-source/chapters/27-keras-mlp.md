# Neural networks with Keras

A neural network is a stack of **layers**. Each dense layer computes `activation(X @ W + b)`, and stacking several with non-linear activations lets the network learn complex functions. Keras turns this into a few lines: define, compile, fit, evaluate. This chapter builds classifiers and regressors on tabular data, then covers the tools for controlling training: callbacks, regularisation and saving.

## The Keras workflow in five steps {B}

:::def
**Neural network**: a model made of layers of simple units (neurons). Each unit computes a weighted sum of its inputs plus a bias and passes it through an activation function.

**Layer**: a building block of a neural network that transforms its input. A **dense (fully connected)** layer connects every input to every unit.

**Activation function**: the non-linear function applied by a unit, such as ReLU (max(0, z)), sigmoid or softmax. It lets networks learn non-linear patterns.

**MLP (multilayer perceptron)**: a neural network made of stacked dense layers.

**Weights and biases**: the learnable parameters of a layer: weights scale the inputs, biases shift the result.
:::


```py title="1–2: data and model definition"
import os  #> Import the standard-library `os` module (operating-system tools).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  #> Hide TensorFlow's informational logs (must be set before importing TensorFlow).
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.
import tensorflow as tf  #> Import TensorFlow under the short name `tf`.
from tensorflow import keras  #> Import `keras` from `tensorflow`.
from tensorflow.keras import layers  #> Import `layers` from `tensorflow.keras`.
from sklearn.datasets import load_breast_cancer  #> Import `load_breast_cancer` from scikit-learn's `datasets` module.
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.
from sklearn.preprocessing import StandardScaler  #> Import `StandardScaler` from scikit-learn's `preprocessing` module.

keras.utils.set_random_seed(0)                                  #> Reproducible weights and shuffling.
X, y = load_breast_cancer(return_X_y=True)  #> Features and labels as NumPy arrays.
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)  #> First carve off a 20% test set.
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=0)   #> A separate validation set to monitor training.
scaler = StandardScaler().fit(X_tr)                             #> Neural nets need scaled inputs. Fit on training data only.
X_tr, X_va, X_te = [scaler.transform(a).astype("float32") for a in (X_tr, X_va, X_te)]   #> float32 for TensorFlow.
print(X_tr.shape, X_va.shape, X_te.shape)  #> Training, validation and test sizes.

model = keras.Sequential([                                       #> A plain stack of layers, applied in order.
    keras.Input(shape=(X_tr.shape[1],)),                         #> Declare the input: 30 features per sample.
    layers.Dense(32, activation="relu"),                         #> Hidden layer: 32 units, ReLU activation max(0, z).
    layers.Dense(16, activation="relu"),  #> Second hidden layer: 16 units.
    layers.Dense(1, activation="sigmoid"),                       #> Output: 1 unit + sigmoid = probability of class 1.
])  #> Close the list and the call or tuple opened above.
model.summary()                                                  #> Layer shapes and parameter counts.
```

The parameter counts check out by hand: the first Dense layer has 30 × 32 weights + 32 biases = 992.

```py title="3–5: compile, fit, evaluate"
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),   #> Adam: the default optimiser for most problems.
              loss="binary_crossentropy",                         #> The loss matching a sigmoid output and 0/1 labels.
              metrics=["accuracy", keras.metrics.AUC(name="auc")])  #> Extra metrics to report; they don't affect training.
history = model.fit(X_tr, y_tr,  #> Train on the training set...
                    validation_data=(X_va, y_va),                #> Evaluate on the validation set after every epoch.
                    epochs=200, batch_size=32, verbose=0)        #> 200 passes over the data. verbose=0 = silent; 1 shows a progress bar per epoch.
test_loss, test_acc, test_auc = model.evaluate(X_te, y_te, verbose=0)  #> Returns the loss followed by each metric, in the order compiled.
print(f"test loss={test_loss:.4f} accuracy={test_acc:.4f} AUC={test_auc:.4f}")  #> Test-set results.
proba = model.predict(X_te[:5], verbose=0).ravel()               #> predict returns shape (n, 1); ravel flattens it.
print("probabilities:", proba.round(3), "-> labels:", (proba >= 0.5).astype(int))  #> Threshold at 0.5 to get class labels.
```

```py title="Plot the training history" caption="Training loss keeps falling, but validation loss bottoms out and then creeps up: the network is starting to overfit. Early stopping (next section) catches this automatically."
h = pd.DataFrame(history.history)                                #> history.history: dict of per-epoch metric lists.
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))  #> Two panels.
h[["loss", "val_loss"]].plot(ax=axes[0], title="Loss (log scale)", logy=True)   #> A log scale makes the late divergence visible.
h[["accuracy", "val_accuracy"]].plot(ax=axes[1], title="Accuracy")  #> Training and validation accuracy per epoch.
for ax in axes: ax.set_xlabel("epoch")  #> Label both x-axes.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
print("epoch with lowest val_loss:", int(h["val_loss"].idxmin()) + 1)  #> +1 because epochs are counted from 1 but the index starts at 0.
```

## Choosing the output layer and loss {B}

This table prevents the most common Keras mistakes.

| Task | Output layer | Loss | Labels look like |
|---|---|---|---|
| Binary classification | `Dense(1, activation="sigmoid")` | `"binary_crossentropy"` | 0/1, shape `(n,)` |
| Multiclass, integer labels | `Dense(k, activation="softmax")` | `"sparse_categorical_crossentropy"` | 0…k−1, shape `(n,)` |
| Multiclass, one-hot labels | `Dense(k, activation="softmax")` | `"categorical_crossentropy"` | shape `(n, k)` |
| Multi-label (several can be true) | `Dense(k, activation="sigmoid")` | `"binary_crossentropy"` | 0/1 matrix `(n, k)` |
| Regression | `Dense(1)` (no activation) | `"mse"`, `"mae"`, `"huber"` | real numbers |

:::bug Accuracy stuck around chance, or a shape error about logits and labels
Usually the output/loss/labels don't match. Examples: `softmax` with 2 units but `binary_crossentropy`; integer labels with `categorical_crossentropy` (use `sparse_...`); or `Dense(1)` without sigmoid for a classifier. If your last layer has **no** activation, pass `from_logits=True` to the loss, e.g. `keras.losses.SparseCategoricalCrossentropy(from_logits=True)`.
:::

## Callbacks: early stopping, checkpoints, learning-rate schedules {I}

:::def
**Callback**: a function Keras calls automatically during training, for example at the end of each epoch, to stop early, save the model or change the learning rate.

**Early stopping**: ending training when validation performance stops improving, to prevent overfitting.

**Dropout**: randomly switching off a fraction of units during each training step, which forces the network to be robust and reduces overfitting.
:::


```py title="Stop when validation loss stops improving"
keras.utils.set_random_seed(0)  #> Reset all seeds so runs are repeatable.
def build_model(units=32, dropout=0.0, l2=0.0):                   #> A function that returns a fresh, compiled model.
    reg = keras.regularizers.l2(l2) if l2 else None                #> L2 weight penalty (weight decay).
    m = keras.Sequential([  #> Build the network:
        keras.Input(shape=(X_tr.shape[1],)),  #> Input: one value per feature.
        layers.Dense(units, activation="relu", kernel_regularizer=reg),  #> Hidden layer with optional L2 penalty.
        layers.Dropout(dropout),                                   #> Randomly zero this fraction of units during TRAINING only.
        layers.Dense(units // 2, activation="relu", kernel_regularizer=reg),  #> A narrower second hidden layer.
        layers.Dense(1, activation="sigmoid"),  #> Probability output.
    ])  #> Close the list and the call or tuple opened above.
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])  #> Optimiser, loss and metric.
    return m  #> Send `m` back to the caller.

callbacks = [  #> Callbacks run automatically at the end of each epoch:
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,       #> Stop after 10 epochs without improvement...
                                  restore_best_weights=True),           #> ...and roll back to the best epoch's weights.
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),   #> Halve the learning rate when progress stalls.
    keras.callbacks.ModelCheckpoint("best_mlp.keras", monitor="val_loss", save_best_only=True),   #> Save the best model to disk.
]  #> Close the list opened above.
m = build_model(dropout=0.3, l2=1e-3)  #> A regularised model.
hist = m.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=300, batch_size=32, callbacks=callbacks, verbose=0)  #> Up to 300 epochs; early stopping will end it sooner.
print("stopped after", len(hist.history["loss"]), "epochs")  #> How many epochs actually ran.
print("test accuracy:", round(m.evaluate(X_te, y_te, verbose=0)[1], 4))  #> [1] picks accuracy out of [loss, accuracy].
```

### Regularisation compared {I}

```py title="Dropout and L2 reduce the train–validation gap" caption="Validation loss curves (no early stopping). The unregularised network's validation loss climbs as it overfits; dropout and L2 keep it lower and flatter."
fig, ax = plt.subplots(figsize=(8, 3.8))  #> One panel.
for label, kw in [("no regularisation", {}), ("dropout 0.4", {"dropout": 0.4}), ("L2 1e-2", {"l2": 1e-2})]:  #> Loop over `[("no regularisation", {}), ("dropout 0.4", {"dropout": 0.4}), ("L2 1e-2", {"l2": 1e-2})]`, naming each item `label, kw`.
    keras.utils.set_random_seed(1)  #> Same seed for each variant, for a fair comparison.
    hh = build_model(units=64, **kw).fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=120, batch_size=32, verbose=0)  #> Train this variant for 120 epochs.
    ax.plot(hh.history["val_loss"], label=label)  #> Its validation loss curve.
ax.set(xlabel="epoch", ylabel="validation loss", title="Effect of regularisation", ylim=(0, 0.4))  #> Labels; y-axis zoomed in.
ax.legend()  #> Show a legend built from each plotted series' `label`.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

| Technique | Keras | Effect |
|---|---|---|
| Early stopping | `callbacks.EarlyStopping` | stop before overfitting; nearly always use it |
| Dropout | `layers.Dropout(0.2–0.5)` | randomly removes units in training; reduces co-adaptation |
| L2 / L1 weight penalty | `kernel_regularizer=regularizers.l2(1e-4)` | keeps weights small |
| Batch normalisation | `layers.BatchNormalization()` | stabilises and speeds up training of deeper nets |
| Data augmentation | `layers.RandomFlip`, `RandomRotation`, … | more varied training data (images) |
| Smaller network | fewer units/layers | less capacity to memorise |

## Multiclass classification {I}

```py title="Wine cultivars with softmax"
from sklearn.datasets import load_wine  #> Import `load_wine` from scikit-learn's `datasets` module.
Xw, yw = load_wine(return_X_y=True)  #> Wine data: 13 features, 3 classes.
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(Xw, yw, test_size=0.25, stratify=yw, random_state=0)  #> Stratified split.
sc = StandardScaler().fit(Xw_tr)  #> Fit the scaler on training data.
Xw_tr, Xw_te = sc.transform(Xw_tr).astype("float32"), sc.transform(Xw_te).astype("float32")  #> Scale both and convert to float32.

keras.utils.set_random_seed(0)  #> Reset all seeds so runs are repeatable.
mc = keras.Sequential([keras.Input(shape=(13,)),  #> Input: 13 features...
                       layers.Dense(32, activation="relu"),  #> A hidden layer of 32 units.
                       layers.Dense(3, activation="softmax")])          #> One unit per class; softmax makes probabilities that sum to 1.
mc.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])   #> Integer labels 0, 1, 2 -> "sparse".
mc.fit(Xw_tr, yw_tr, epochs=80, batch_size=16, validation_split=0.2, verbose=0)               #> validation_split: hold out the last 20% of training rows.
probs = mc.predict(Xw_te, verbose=0)  #> Shape (n, 3): one probability per class.
print(probs[:3].round(3), probs[:3].sum(axis=1))                        #> Each row sums to 1.
print("test accuracy:", round(float(np.mean(probs.argmax(axis=1) == yw_te)), 4))  #> argmax picks the most probable class.
```

:::warn `validation_split` takes the *last* rows
`validation_split=0.2` uses the final 20% of the arrays **before** shuffling. If your data is sorted (by class, date or subject), shuffle first or pass an explicit `validation_data=(X_val, y_val)`.
:::

## Regression {I}

```py title="Predicting insurance charges with an MLP"
ins = pd.read_csv("insurance.csv").dropna()  #> Load and drop the rows with missing BMI.
Xi = pd.get_dummies(ins.drop(columns="charges"), drop_first=True, dtype=float)   #> Quick one-hot encoding for the example.
yi = ins["charges"].to_numpy(dtype="float32") / 1000                            #> Scale the target to thousands: helps optimisation.
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.2, random_state=0)  #> 80/20 split.
sci = StandardScaler().fit(Xi_tr)  #> Scaler fitted on training features.
Xi_tr, Xi_te = sci.transform(Xi_tr).astype("float32"), sci.transform(Xi_te).astype("float32")  #> Scale and convert.

keras.utils.set_random_seed(0)  #> Reset all seeds so runs are repeatable.
reg = keras.Sequential([keras.Input(shape=(Xi_tr.shape[1],)),  #> Input layer sized to the encoded features...
                        layers.Dense(64, activation="relu"),  #> Hidden layer of 64 units.
                        layers.Dense(32, activation="relu"),  #> A hidden layer of 32 units.
                        layers.Dense(1)])                                       #> Linear output for regression.
reg.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])  #> Minimise MSE; also report MAE.
reg.fit(Xi_tr, yi_tr, validation_split=0.2, epochs=400, batch_size=32, verbose=0,  #> Train with 20% of the training rows held out for validation...
        callbacks=[keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)])  #> ...stopping after 25 epochs without improvement.
mse, mae = reg.evaluate(Xi_te, yi_te, verbose=0)  #> Test MSE and MAE (in thousands).
print(f"test RMSE = {np.sqrt(mse) * 1000:,.0f}   MAE = {mae * 1000:,.0f}  (charges)")  #> Back to original units by multiplying by 1000.
```

The MLP does well here, but this split (missing rows dropped, a different seed) differs from [[ch:sklearn-regression]]'s, so the numbers cannot be ranked against each other. Compare models only under the same cross-validation protocol. As a rule, **for tabular data, gradient boosting and linear models are strong baselines** that neural networks often fail to beat. Deep learning shines on images, audio, text and very large datasets.

## The Functional API {A}

`Sequential` handles single-input, single-output stacks. The **Functional API** builds any graph of layers: multiple inputs, multiple outputs, skip connections.

```py title="Two inputs, one output, with a skip connection"
num_in = keras.Input(shape=(5,), name="numeric")                  #> First input: 5 numeric features.
cat_in = keras.Input(shape=(1,), name="category", dtype="int32")  #> Second input: an integer category ID.
emb = layers.Flatten()(layers.Embedding(input_dim=10, output_dim=4)(cat_in))   #> Learn a 4-D vector for each of 10 categories.
h = layers.Dense(16, activation="relu")(num_in)                   #> Each layer is CALLED on a tensor, returning a new tensor.
h = layers.Concatenate()([h, emb, num_in])                        #> Merge branches; num_in is also passed straight through (skip).
out = layers.Dense(1, activation="sigmoid", name="prob")(h)  #> Probability output.
fm = keras.Model(inputs=[num_in, cat_in], outputs=out)  #> Build the model from its inputs and output.
fm.compile(optimizer="adam", loss="binary_crossentropy")  #> Compile as usual.
rng = np.random.default_rng(0)  #> Seeded generator for fake data.
fm.fit({"numeric": rng.normal(size=(64, 5)).astype("float32"), "category": rng.integers(0, 10, (64, 1))},  #> Inputs passed as a dict keyed by input name...
       rng.integers(0, 2, 64), epochs=2, verbose=0)               #> Feed inputs by name with a dict.
print([t.name for t in fm.inputs], fm.output_shape, fm.count_params())  #> Input names, output shape and number of parameters.
```

## Saving and loading {B}

```py title="The native .keras format"
m.save("cancer_mlp.keras")                                       #> Architecture + weights + optimizer state in one file.
restored = keras.models.load_model("cancer_mlp.keras")  #> Load it back.
print(np.allclose(restored.predict(X_te, verbose=0), m.predict(X_te, verbose=0)))  #> Same predictions: True.
m.save_weights("cancer_mlp.weights.h5")                          #> Weights only (you must rebuild the same architecture to load them).
```

:::research Save the scaler too
A Keras model expects inputs scaled exactly as in training. Save the fitted `StandardScaler` with `joblib.dump(scaler, "scaler.joblib")` next to the model, or put preprocessing inside the model with `layers.Normalization` (call `.adapt(X_train)` on it before training).
:::

:::ex Try it yourself
1. Replace the two hidden layers with one layer of 8 units. Does test accuracy change? What about 256 units?
2. Add `layers.BatchNormalization()` after each Dense layer and compare the training curves.
3. Wrap `build_model` in a loop over five seeds and report test accuracy as mean ± SD.
:::
