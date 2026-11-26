# Deep learning architectures: CNNs, sequence models and custom training

Dense networks treat every input as an unrelated number. **Convolutional networks (CNNs)** exploit the spatial structure of images; **recurrent and 1-D convolutional networks** exploit order in sequences. This chapter builds both on small datasets that train in seconds on a CPU, then shows custom training loops and transfer learning.

## Convolutional neural networks {I}

:::def
**Convolutional neural network (CNN)**: a network that slides small learnable filters over its input to detect local patterns. It is the standard architecture for images.

**Convolution**: sliding a small filter (kernel) over the input and computing a weighted sum at each position, producing a feature map.

**Pooling**: downsampling a feature map, for example by keeping the maximum in each 2 × 2 block, which reduces size and adds robustness to small shifts.

**Channel**: one layer of values in an image tensor: 1 for grayscale, 3 for RGB, or many feature maps inside a CNN.
:::


A convolution layer slides small learnable filters (for example 3×3) across the image. Each filter detects a local pattern, such as an edge, wherever it appears. **Pooling** layers downsample, so deeper layers see larger regions. A typical CNN: [Conv → Pool] × n → Flatten → Dense → output.

```py title="Prepare 8×8 digit images"
import os  #> Import the standard-library `os` module (operating-system tools).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  #> Hide TensorFlow's informational logs (must be set before importing TensorFlow).
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.
import tensorflow as tf  #> Import TensorFlow under the short name `tf`.
from tensorflow import keras  #> Import `keras` from `tensorflow`.
from tensorflow.keras import layers  #> Import `layers` from `tensorflow.keras`.
from sklearn.datasets import load_digits  #> Import `load_digits` from scikit-learn's `datasets` module.
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.

digits = load_digits()  #> 1797 handwritten digits as 8x8 images.
X = (digits.images / 16.0).astype("float32")          #> Pixel values 0–16 scaled to 0–1. Shape (1797, 8, 8).
X = X[..., np.newaxis]                                #> Add a CHANNEL axis -> (1797, 8, 8, 1). Grayscale = 1 channel, RGB = 3.
y = digits.target  #> Labels 0-9.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)  #> Stratified 80/20 split.
print(X_tr.shape, X_te.shape, np.bincount(y_tr))  #> Shapes and the count of each digit in training.
```

```py figonly caption="Sixteen training images with their labels. At 8×8 pixels they are tiny, which keeps training fast while showing every idea that scales to large images."
fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
for ax, img, lab in zip(axes.ravel(), X_tr[:16], y_tr[:16]):
    ax.imshow(img[..., 0], cmap="gray_r"); ax.set_title(str(lab), fontsize=11); ax.axis("off")
plt.tight_layout(); plt.show()
```

```py title="Build and train a small CNN"
keras.utils.set_random_seed(0)  #> Reset all seeds so runs are repeatable.
cnn = keras.Sequential([  #> Build the CNN layer by layer:
    keras.Input(shape=(8, 8, 1)),                                     #> (height, width, channels). The batch dimension is implicit.
    layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),   #> 32 filters of 3x3; "same" padding keeps 8x8. Output (8, 8, 32).
    layers.MaxPooling2D(pool_size=2),                                 #> Keep the max of each 2x2 block -> (4, 4, 32).
    layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),   #> -> (4, 4, 64).
    layers.MaxPooling2D(pool_size=2),                                 #> -> (2, 2, 64).
    layers.Flatten(),                                                 #> Unroll to a vector of 256 values.
    layers.Dropout(0.3),  #> Drop 30% of values during training.
    layers.Dense(64, activation="relu"),  #> A dense layer combines the extracted features.
    layers.Dense(10, activation="softmax"),                           #> 10 digit classes.
])  #> Close the list and the call or tuple opened above.
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  #> Integer labels -> sparse categorical cross-entropy.
cnn.summary()  #> Shapes after each layer and parameter counts.
hist = cnn.fit(X_tr, y_tr, validation_split=0.15, epochs=40, batch_size=64, verbose=0,  #> Train, holding out 15% of training rows for validation...
               callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])  #> ...with early stopping.
print("epochs run:", len(hist.history["loss"]), "| test accuracy:", round(cnn.evaluate(X_te, y_te, verbose=0)[1], 4))  #> How long it trained and how well it does.
```

```py title="Where does it go wrong?" caption="Left: the confusion matrix shows which digits get mixed up. Right: misclassified test images, with predicted and true labels."
from sklearn.metrics import ConfusionMatrixDisplay  #> Import `ConfusionMatrixDisplay` from scikit-learn's `metrics` module.
pred = cnn.predict(X_te, verbose=0).argmax(axis=1)                    #> Most probable class per image.
fig = plt.figure(figsize=(12, 4.4))  #> An empty figure; we add panels by hand.
ax0 = fig.add_subplot(1, 2, 1)  #> Left half of the figure.
ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax0, colorbar=False, cmap="Oranges")  #> 10x10 confusion matrix.
ax0.set_title("Confusion matrix (test)"); ax0.grid(False)  #> Title, no grid.
wrong = np.flatnonzero(pred != y_te)[:8]                              #> Indices of the first 8 mistakes.
for k, i in enumerate(wrong):  #> Loop over `enumerate(wrong)`, naming each item `k, i`.
    ax = fig.add_subplot(2, 8, 5 + k + (4 if k >= 4 else 0))  #> Place each small image in a 2x8 grid on the right half.
    ax.imshow(X_te[i, ..., 0], cmap="gray_r"); ax.axis("off")  #> Show the image without axes.
    ax.set_title(f"pred {pred[i]}\ntrue {y_te[i]}", fontsize=9, color="#d42a2a")  #> Predicted and true labels.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

### Data augmentation {I}

:::def
**Data augmentation**: creating varied training examples by randomly transforming existing ones (rotations, flips, crops) while keeping their labels.
:::


Augmentation layers randomly transform training images (flip, rotate, zoom, shift), effectively enlarging the dataset. They are active only during training.

```py title="Augmentation as layers inside the model"
augment = keras.Sequential([  #> A small model made only of augmentation layers:
    layers.RandomRotation(0.05),                     #> Rotate by up to ±5% of a full turn (±18 degrees).
    layers.RandomTranslation(0.1, 0.1),              #> Shift up to 10% vertically and horizontally.
    layers.RandomZoom(0.1),  #> Zoom in or out by up to 10%.
], name="augment")                                   #> No RandomFlip here: a mirrored "2" is not a valid digit!
sample = augment(X_tr[:1].repeat(5, axis=0), training=True)   #> training=True forces random transforms so we can look at them.
print(sample.shape)  #> Display the value of `sample.shape`.
```

:::tip Choose augmentations that preserve the label
Horizontal flips are fine for cats and chest X-rays (with care), wrong for digits and text. Rotations are fine for microscopy, wrong for street signs. Augmentation encodes your assumptions about which changes don't matter.
:::

## Transfer learning {A}

:::def
**Transfer learning**: starting from a model pretrained on a large dataset and adapting (fine-tuning) it to a new task with less data.
:::


For real image problems with limited data, start from a network pretrained on ImageNet and retrain only the top. This usually beats training from scratch by a wide margin.

```py norun title="Fine-tuning a pretrained image model (downloads weights on first use)"
base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",   #> Pretrained convolutional base, without its classifier.
                                         input_shape=(224, 224, 3))  #> Images of 224x224 pixels with 3 colour channels.
base.trainable = False                                         #> Freeze the pretrained weights at first.
inputs = keras.Input(shape=(224, 224, 3))  #> The new model's input.
x = augment(inputs)                                            #> Your augmentation pipeline.
x = base(x, training=False)                                    #> training=False keeps BatchNorm layers in inference mode.
x = layers.GlobalAveragePooling2D()(x)                         #> Average each feature map to one number.
x = layers.Dropout(0.3)(x)  #> Regularise the new head.
outputs = layers.Dense(n_classes, activation="softmax")(x)     #> A new head for YOUR classes.
model = keras.Model(inputs, outputs)  #> Assemble the model.
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])  #> Compile with a normal learning rate.
model.fit(train_ds, validation_data=val_ds, epochs=10)         #> Stage 1: train the head only.
base.trainable = True                                          #> Stage 2: unfreeze and fine-tune everything...
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])   #> ...with a 100x smaller learning rate (recompile after changing trainable!).
model.fit(train_ds, validation_data=val_ds, epochs=5)  #> Fine-tune for a few epochs.
```

`train_ds` would typically come from `keras.utils.image_dataset_from_directory("data/train", image_size=(224, 224), batch_size=32)`, which expects one sub-folder per class.

## Sequence models: forecasting a time series {I}

:::def
**RNN / LSTM**: recurrent neural networks process sequences step by step while keeping a memory (hidden state); the LSTM is a variant that remembers long-range patterns well.

**Forecasting horizon**: how far ahead a forecast predicts: one step, a week, a year.
:::


For sequences we feed the model a **window** of past values and ask it to predict the next one. Keras' `timeseries_dataset_from_array` builds the windows for you.

```py title="Windowed data from a noisy seasonal series"
keras.utils.set_random_seed(0)  #> Reset all seeds so runs are repeatable.
t = np.arange(1200, dtype="float32")  #> 1200 time steps.
series = (np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 17)  #> A 50-step cycle plus a 17-step cycle...
          + 0.001 * t + np.random.default_rng(0).normal(0, 0.15, t.size)).astype("float32")   #> Two cycles + trend + noise.
split = 1000                                                    #> Train on the first 1000 steps, test on the last 200: chronological!
mean, std = series[:split].mean(), series[:split].std()         #> Normalise with TRAINING statistics only.
s = (series - mean) / std  #> Standardised series.
window = 40                                                     #> Use the previous 40 values to predict the next one.

def make_ds(arr, shuffle):  #> Define a function `make_ds` that takes `arr` and `shuffle`.
    return keras.utils.timeseries_dataset_from_array(  #> Send `keras.utils.timeseries_dataset_from_array(` back to the caller.
        data=arr[:-1], targets=arr[window:],                    #> Input windows arr[i:i+40] paired with target arr[i+40].
        sequence_length=window, batch_size=32, shuffle=shuffle, seed=0)  #> Windows of 40 steps, batches of 32.

train_ds = make_ds(s[:split], shuffle=True)  #> Training windows from the first 1000 steps.
test_ds = make_ds(s[split - window:], shuffle=False)            #> Start the test windows 40 steps early so the first prediction is for step 1000.
xb, yb = next(iter(train_ds))  #> Take one batch to inspect.
print(xb.shape, yb.shape)                                        #> (batch, 40 time steps); a channel axis is added inside the model.
```

```py title="LSTM vs 1-D CNN vs a naive baseline"
def train(model):  #> Define a function `train` that takes `model`.
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")  #> Regression loss.
    model.fit(train_ds, epochs=30, verbose=0, shuffle=False)   #> The dataset already shuffles its windows; shuffle=False avoids a redundant-shuffle warning.
    return model  #> Send `model` back to the caller.

lstm = train(keras.Sequential([keras.Input(shape=(window,)),  #> LSTM model: input of 40 values...
                               layers.Reshape((window, 1)),             #> RNNs expect (time steps, features per step).
                               layers.LSTM(32),                         #> 32 memory units; returns the final hidden state.
                               layers.Dense(1)]))  #> ...one output value: the next step.
conv = train(keras.Sequential([keras.Input(shape=(window,)),  #> 1-D CNN model: input of 40 values...
                               layers.Reshape((window, 1)),  #> Add a feature axis: (40,) -> (40, 1).
                               layers.Conv1D(32, kernel_size=5, activation="relu"),   #> Filters slide along time.
                               layers.Flatten(),                        #> Keep every time position. (Global average pooling would forget which values are most recent.)
                               layers.Dense(1)]))  #> ...one output value: the next step.
y_test = np.concatenate([y.numpy() for _, y in test_ds])  #> Collect all test targets from the dataset.
naive = s[split - 1:-1]                                          #> Baseline: "tomorrow = today".
rmse = lambda p: float(np.sqrt(np.mean((p - y_test) ** 2)) * std)   #> Back to original units.
p_lstm = lstm.predict(test_ds, verbose=0).ravel()  #> LSTM predictions.
p_conv = conv.predict(test_ds, verbose=0).ravel()  #> Conv1D predictions.
print(f"test RMSE  naive={rmse(naive):.3f}  LSTM={rmse(p_lstm):.3f}  Conv1D={rmse(p_conv):.3f}")  #> Compare all three on the same test steps.
```

```py title="Forecasts against the truth" caption="One-step-ahead forecasts on the 200 held-out time steps. Always compare with a naive baseline: for many real series, 'same as last time' is hard to beat."
fig, ax = plt.subplots(figsize=(10, 3.6))  #> One wide panel.
ax.plot(y_test * std + mean, color="black", lw=1, label="actual")  #> True values, back in original units.
ax.plot(p_lstm * std + mean, label="LSTM")  #> LSTM forecasts.
ax.plot(p_conv * std + mean, label="Conv1D", alpha=0.8)  #> Conv1D forecasts.
ax.set(xlabel="test time step", ylabel="value", title="One-step-ahead forecasts")  #> Labels.
ax.legend(ncol=3)  #> Legend in one row.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

:::bug My forecaster is worse than "same as last time"
Check three things. First, normalisation: did you scale with training statistics and un-scale predictions? Second, architecture: a global pooling layer (`GlobalAveragePooling1D`) or a too-short window throws away *when* things happened, and for forecasting the most recent values matter most (`Flatten` or an RNN keeps the order). Third, alignment: are the targets really the *next* step after each window? An off-by-one shift silently trains the model to predict the present.
:::

:::research Evaluate forecasts honestly
Split by time (never shuffle across the split), normalise with training statistics, compare against naive and seasonal-naive baselines, and if you tune hyperparameters, use `TimeSeriesSplit` or a rolling-origin evaluation. Multi-step forecasts degrade with the horizon, so report errors per horizon.
:::

## Custom training loops {A}

When `fit()` is not flexible enough (research on new losses, GANs, unusual schedules), write the loop yourself. It is the GradientTape pattern from [[ch:tf-basics]] applied to a Keras model.

```py title="A complete custom loop"
keras.utils.set_random_seed(0)  #> Reset all seeds so runs are repeatable.
model = keras.Sequential([keras.Input(shape=(8, 8, 1)), layers.Flatten(),  #> A simple dense network on the flattened image...
                          layers.Dense(64, activation="relu"), layers.Dense(10)])   #> No softmax: we output logits...
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)               #> ...and tell the loss so (more numerically stable).
optimizer = keras.optimizers.Adam(1e-3)  #> The optimiser.
train_acc = keras.metrics.SparseCategoricalAccuracy()  #> A running accuracy metric.
ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(2000, seed=0).batch(64)  #> Shuffled mini-batches of 64.

@tf.function                                                          #> Compile the step into a fast graph.
def train_step(xb, yb):  #> Define a function `train_step` that takes `xb` and `yb`.
    with tf.GradientTape() as tape:  #> Record the forward pass.
        logits = model(xb, training=True)                             #> training=True activates dropout/batch-norm training behaviour.
        loss = loss_fn(yb, logits)  #> Compare predictions with the labels.
    grads = tape.gradient(loss, model.trainable_variables)  #> Gradients for every trainable weight.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  #> Update the weights.
    train_acc.update_state(yb, logits)  #> Accumulate accuracy over the epoch.
    return loss  #> Send `loss` back to the caller.

for epoch in range(5):  #> Loop over `range(5)`, naming each item `epoch`.
    train_acc.reset_state()                                           #> Reset the running metric each epoch.
    for xb, yb in ds:  #> Loop over `ds`, naming each item `xb, yb`.
        loss = train_step(xb, yb)  #> One training step per batch.
    print(f"epoch {epoch + 1}: loss={float(loss):.4f}  train acc={float(train_acc.result()):.4f}")  #> Report the last batch's loss and the epoch's accuracy.
test_logits = model(X_te, training=False)  #> Predict on test images (inference mode).
print("test accuracy:", round(float(np.mean(np.argmax(test_logits, axis=1) == y_te)), 4))  #> The highest logit is the predicted class.
```

## Other architectures in one table {I}

| Data | Architecture | Keras building blocks |
|---|---|---|
| Images | CNN, ResNet, EfficientNet, Vision Transformer | `Conv2D`, `keras.applications.*` |
| Sequences / time series | LSTM, GRU, 1-D CNN, Transformer | `LSTM`, `GRU`, `Conv1D`, `MultiHeadAttention` |
| Text | Embeddings + Transformer | `TextVectorization`, `Embedding`, `MultiHeadAttention`; KerasHub for pretrained models |
| Tabular | MLP with embeddings for categories (often beaten by boosting) | `Dense`, `Embedding` |
| Compression / anomaly detection | Autoencoder | encoder–decoder of `Dense`/`Conv` layers |
| Generation | VAE, GAN, diffusion | custom training loops |

:::note Keras 3 and other back ends
Keras 3 can also run on JAX or PyTorch back ends (set the `KERAS_BACKEND` environment variable before importing Keras). The code in this part runs on the default TensorFlow back end.
:::

:::ex Try it yourself
1. Add the `augment` block as the first layer of the CNN and compare test accuracy.
2. Replace `LSTM(32)` with `GRU(32)`, and try `window = 10` and `window = 80`.
3. Turn the forecaster into a 5-step-ahead model by changing the targets and using `Dense(5)`.
:::
