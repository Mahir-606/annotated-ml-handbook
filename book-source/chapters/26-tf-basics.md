@part 6

# TensorFlow fundamentals: tensors, gradients and data pipelines

TensorFlow is a library for numerical computing on CPUs, GPUs and TPUs with **automatic differentiation**: it can compute the gradient of any calculation you write. Keras, its high-level API, builds neural networks from layers. This chapter covers the low-level pieces that every Keras model is made of, so that the next chapters (and their error messages) make sense.

:::note Installation and hardware
`pip install tensorflow` installs a CPU build that runs everywhere; GPU support on Linux is `pip install "tensorflow[and-cuda]"`. On Windows, use WSL2 for GPU training. Everything in this part runs comfortably on a laptop CPU.
:::

## Tensors {B}

:::def
**Tensor**: TensorFlow's n-dimensional array, with a shape and a dtype, that can live on a CPU, GPU or TPU.

**GPU**: a graphics processor with thousands of small cores, which speeds up the matrix maths of deep learning enormously.
:::


A **tensor** is TensorFlow's n-dimensional array, very much like a NumPy array, with a `shape` and a `dtype`.

```py title="Creating and inspecting tensors"
import os  #> Import the standard-library `os` module (operating-system tools).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"           #> Hide TensorFlow's informational start-up logs (set BEFORE importing).
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import tensorflow as tf  #> Import TensorFlow under the short name `tf`.

print("TensorFlow", tf.__version__)  #> Check the installed version.
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])          #> An immutable tensor from a nested list.
print(a)  #> Display the value of `a`.
print(a.shape, a.dtype, a.ndim)                    #> TensorShape([2, 2]), float32, 2.
b = tf.zeros((2, 3)); c = tf.ones((3,)); r = tf.random.normal((2, 2), seed=1)  #> Zeros, ones and random normal tensors.
print(b.shape, c.numpy(), r.dtype)                 #> `.numpy()` converts a tensor to a NumPy array.
x = tf.constant(np.arange(6).reshape(2, 3))        #> From a NumPy array; NumPy int64 stays int64.
print(x.dtype, tf.cast(x, tf.float32).dtype)       #> `tf.cast` changes dtype (TensorFlow does not mix dtypes automatically).
```

:::warn float32 is the default for deep learning
`tf.constant([1.0])` is `float32`, but NumPy arrays default to `float64`. Mixing them raises `InvalidArgumentError: cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor but is a double tensor`. Convert data with `X.astype("float32")`.
:::

## Operations {B}

```py title="Maths, reductions, reshaping, indexing"
m = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  #> A 2x3 tensor.
print(m + 10, m * m, sep="\n")                     #> Element-wise with broadcasting, as in NumPy.
print(tf.matmul(m, tf.transpose(m)))               #> Matrix product (also written m @ tf.transpose(m)).
print(tf.reduce_sum(m), tf.reduce_mean(m, axis=0).numpy())   #> NumPy's sum/mean are reduce_* in TensorFlow.
print(tf.reshape(m, (3, 2)).shape, tf.expand_dims(m, 0).shape)   #> Reshape; add a leading batch dimension.
print(m[:, 1].numpy(), tf.argmax(m, axis=1).numpy())             #> Slicing and argmax work like NumPy.
print(tf.nn.softmax(tf.constant([2.0, 1.0, 0.1])).numpy().round(3))   #> Neural-network functions live in tf.nn.
```

## Variables and automatic differentiation {I}

:::def
**Automatic differentiation (autodiff)**: computing exact derivatives of any calculation automatically, by recording its operations. `tf.GradientTape` does this.

**tf.Variable**: a tensor whose value can change, used for model weights.

**Optimizer**: the algorithm that updates the weights using the gradients, such as SGD or Adam.
:::


A `tf.Variable` holds a value that can change, such as a weight. `tf.GradientTape` records the operations performed inside it, and can then compute gradients of a result with respect to any variable.

```py title="Gradients with GradientTape"
w = tf.Variable(3.0)                           #> A trainable scalar.
with tf.GradientTape() as tape:                #> Start recording.
    loss = w ** 2 + 2 * w                      #> loss = w² + 2w
grad = tape.gradient(loss, w)                  #> d(loss)/dw = 2w + 2 = 8 at w = 3.
print(float(loss), float(grad))  #> 15.0 and 8.0.

x = tf.constant([1.0, 2.0, 3.0])  #> A constant tensor.
with tf.GradientTape() as tape:  #> Record the operations below.
    tape.watch(x)                              #> Constants are not tracked automatically; `watch` asks the tape to track x.
    y = tf.reduce_sum(tf.sin(x))  #> sum(sin(x)).
print(tape.gradient(y, x).numpy().round(4), np.cos([1, 2, 3]).round(4))   #> Gradient of sum(sin x) is cos x. Matches.
```

### Training a model by hand {I}

This is what `model.fit()` does under the hood: predict, compute the loss, compute gradients, and let an optimiser update the weights.

```py title="Linear regression with a manual training loop" caption="The two learnable weights move from their initial values to the true values (slope 3, intercept 2) as training proceeds."
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.
rng = np.random.default_rng(0)  #> Seeded NumPy generator.
X = rng.uniform(-1, 1, (200, 1)).astype("float32")  #> 200 inputs between -1 and 1, as float32.
y = (3 * X[:, 0] + 2 + rng.normal(0, 0.2, 200)).astype("float32")   #> True slope 3, intercept 2.

W = tf.Variable(tf.zeros((1,)))                 #> Slope, initialised at 0.
b = tf.Variable(0.0)                            #> Intercept.
opt = tf.keras.optimizers.SGD(learning_rate=0.1)   #> Plain stochastic gradient descent.
trace = []  #> Records the parameters after each step.
for step in range(100):  #> Loop over `range(100)`, naming each item `step`.
    with tf.GradientTape() as tape:  #> Record the operations below.
        pred = X[:, 0] * W + b                  #> Forward pass.
        loss = tf.reduce_mean((pred - y) ** 2)  #> MSE loss.
    grads = tape.gradient(loss, [W, b])         #> Backward pass: gradients for every variable.
    opt.apply_gradients(zip(grads, [W, b]))     #> Update: variable -= learning_rate * gradient.
    trace.append((float(W[0]), float(b)))  #> Store current slope and intercept.
print(f"learned slope={W.numpy()[0]:.3f}, intercept={b.numpy():.3f}, final loss={loss.numpy():.4f}")  #> Close to 3 and 2.
fig, ax = plt.subplots(figsize=(7, 3.4))  #> One panel.
ax.plot([t[0] for t in trace], label="slope W"); ax.plot([t[1] for t in trace], label="intercept b")  #> Both parameters over time.
ax.axhline(3, ls=":", color="grey"); ax.axhline(2, ls=":", color="grey")  #> Dotted lines at the true values.
ax.set(xlabel="training step", ylabel="value", title="Parameters during gradient descent"); ax.legend()  #> Labels and legend.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Speed: tf.function {A}

Decorating a Python function with `@tf.function` compiles it into a TensorFlow graph, which usually runs faster and can be exported. Keras does this for you inside `fit()`.

```py title="Graph compilation"
@tf.function                                         #> Traced once per input signature, then runs as an optimised graph.
def train_step(X, y):  #> Define a function `train_step` that takes `X` and `y`.
    with tf.GradientTape() as tape:  #> Record the operations below.
        loss = tf.reduce_mean((X[:, 0] * W + b - y) ** 2)  #> MSE loss.
    grads = tape.gradient(loss, [W, b])  #> Gradients for W and b.
    opt.apply_gradients(zip(grads, [W, b]))  #> Update the variables.
    return loss  #> Send `loss` back to the caller.

print(float(train_step(X, y)))  #> Run one compiled step and show the loss.
```

:::bug Python side-effects inside `tf.function`
`print()` inside a `tf.function` runs only while the function is being traced (usually once), not on every call; use `tf.print()` instead. Avoid changing Python lists or global variables inside it. While debugging, run eagerly with `tf.config.run_functions_eagerly(True)`.
:::

## Input pipelines with tf.data {I}

:::def
**Mini-batch**: a small subset of the training data (such as 32 samples) used for one gradient update.
:::


`tf.data.Dataset` streams data in shuffled mini-batches, applies transformations in parallel, and prefetches the next batch while the model trains. It scales from NumPy arrays to datasets far larger than memory.

```py title="Building a dataset pipeline"
Xd = rng.normal(size=(1000, 4)).astype("float32")  #> 1000 samples, 4 features.
yd = (Xd.sum(axis=1) > 0).astype("int32")  #> Label 1 if the features sum to more than 0.
ds = tf.data.Dataset.from_tensor_slices((Xd, yd))      #> One element per row: (features, label).
ds = (ds.shuffle(buffer_size=1000, seed=0)            #> Shuffle within a buffer (use the dataset size if it fits in memory).
        .map(lambda x, t: (x * 2.0, t), num_parallel_calls=tf.data.AUTOTUNE)   #> Apply a transformation to each element, in parallel.
        .batch(32)                                    #> Group into mini-batches of 32.
        .prefetch(tf.data.AUTOTUNE))                  #> Prepare the next batch while the current one trains.
for xb, tb in ds.take(1):                             #> Peek at one batch.
    print(xb.shape, tb.shape, tb[:8].numpy())  #> (32, 4) features, (32,) labels, and the first 8 labels.
print("batches per epoch:", len(list(ds)))            #> ceil(1000 / 32) = 32.
```

:::tip When to use tf.data
For small in-memory tables, passing NumPy arrays straight to `model.fit(X, y, batch_size=32)` is simplest. Reach for `tf.data` for large datasets, images on disk (`tf.keras.utils.image_dataset_from_directory`), text, or heavy on-the-fly augmentation.
:::

## Reproducibility {B}

```py title="One call seeds Python, NumPy and TensorFlow"
tf.keras.utils.set_random_seed(42)                  #> Sets the Python `random`, NumPy and TensorFlow seeds together.
print(tf.random.uniform((3,)).numpy().round(4))  #> Three random numbers.
tf.keras.utils.set_random_seed(42)  #> Reset all seeds to the same value.
print(tf.random.uniform((3,)).numpy().round(4))     #> Identical.
```

:::research Determinism on GPUs
Some GPU operations are non-deterministic even with seeds. For bit-exact reruns call `tf.config.experimental.enable_op_determinism()` (slower). In practice, report the mean and standard deviation over several seeds: deep-learning results vary run to run.
:::

:::ex Try it yourself
1. Use `GradientTape` to find the minimum of f(x) = (x − 4)² + 1 by gradient descent, starting from x = 0.
2. Extend the manual training loop to two input features.
3. Build a `tf.data` pipeline from the breast-cancer data (`load_breast_cancer`) with batches of 64.
:::
