@part 2

# NumPy arrays: shape, dtype and indexing

NumPy's `ndarray` is a grid of values that all share one type, stored in a single contiguous block of memory. That design is why NumPy is fast, and why pandas, scikit-learn, TensorFlow and most of scientific Python are built on top of it. Every dataset you feed to a model eventually becomes an array.

## Why arrays instead of lists? {B}

:::def
**NumPy array (ndarray)**: a grid of values that all share one type, stored in one contiguous block of memory. It can have any number of dimensions.

**Vectorization**: applying an operation to a whole array at once in compiled code, instead of looping over elements in Python. Typically 10–100 times faster.
:::


```py title="The same computation, list vs array"
import numpy as np                          #> The universal convention: import numpy as np.
import time  #> Import the standard-library `time` module.

values = list(range(1_000_000))             #> One million Python integers in a list.
arr = np.arange(1_000_000)                  #> The same numbers as a NumPy array.

t0 = time.perf_counter()  #> Start the stopwatch.
squared_list = [v * v for v in values]      #> Python visits each item one at a time.
t1 = time.perf_counter()  #> Time after the list version.
squared_arr = arr * arr                     #> NumPy multiplies the whole array in compiled C code: *vectorization*.
t2 = time.perf_counter()  #> Time after the array version.
print(f"list: {(t1 - t0) * 1000:.1f} ms | array: {(t2 - t1) * 1000:.1f} ms | speed-up ~{(t1 - t0) / (t2 - t1):.0f}x")  #> Both timings in milliseconds and how many times faster the array was.
print(squared_list[:5], squared_arr[:5])    #> Same numbers, very different speed.
```

:::key Think in whole arrays
The core NumPy skill is replacing loops over elements with operations on whole arrays. If you find yourself writing `for i in range(len(arr))`, there is almost always a vectorized alternative.
:::

## Creating arrays {B}

```py title="The array factory functions"
a = np.array([1, 2, 3])                      #> From a Python list: a 1-D array.
M = np.array([[1, 2, 3], [4, 5, 6]])         #> From a nested list: a 2-D array (2 rows, 3 columns).
print(a, M, sep="\n")  #> Print the 1-D and 2-D arrays on separate lines.
print(np.zeros((2, 3)))                      #> Array of zeros; the shape is given as a tuple (rows, cols).
print(np.ones(4), np.full((2, 2), 7))        #> Ones; or any constant with `full`.
print(np.eye(3))                             #> 3x3 identity matrix.
print(np.arange(0, 10, 2.5))                 #> Like range() but returns an array and allows float steps.
print(np.linspace(0, 1, 5))                  #> 5 evenly spaced points from 0 to 1 INCLUSIVE: perfect for plotting x-axes.
print(np.empty(3).shape)                     #> `empty` allocates without initialising (contents are garbage), so only use it when you'll fill every value.
```

## The three attributes you must always know {B}

:::def
**Shape**: a tuple giving an array's size along each dimension. *Example:* `(150, 4)` means 150 rows and 4 columns.

**Dimension (ndim)**: the number of axes an array has: 1 for a vector, 2 for a table or matrix, 3 for a stack of images, and so on.

**dtype**: the element type shared by all values in an array or column, such as `int64`, `float32`, `bool` or `str`.
:::


Most NumPy (and ML) bugs are **shape bugs**. Print `.shape` constantly.

```py title="shape, ndim, dtype, size"
X = np.array([[1.5, 2.0, 3.1],  #> A 2-D array written over two lines: first row...
              [4.0, 5.2, 6.3]])  #> ...second row. Two rows of three floats.
print("shape:", X.shape)      #> (rows, columns) = (2, 3). For ML: (n_samples, n_features).
print("ndim :", X.ndim)       #> Number of dimensions (axes): 2.
print("size :", X.size)       #> Total number of elements: 6.
print("dtype:", X.dtype)      #> The element type shared by every value: float64.
print(np.array([1, 2]).dtype, np.array([True, False]).dtype, np.array(["a", "bc"]).dtype)   #> int64, bool, Unicode string of max length 2.
```

### dtypes and casting {I}

```py title="Choosing and converting element types"
x = np.array([1, 2, 3], dtype=np.float32)     #> Set the dtype explicitly. float32 halves memory vs float64; deep learning uses it by default.
print(x.dtype, x.nbytes, "bytes")  #> 3 values x 4 bytes = 12 bytes.
y = np.array([1.7, -2.9, 3.5]).astype(int)    #> `astype` returns a converted COPY. Float -> int truncates toward zero.
print(y)  #> Display the value of `y`.
mixed = np.array([1, 2.5, "3"])               #> Mixed types are upcast to a common type, here strings!
print(mixed, mixed.dtype)                     #> A classic source of "why can't I do maths on my column?"
small = np.array([250, 251], dtype=np.uint8)  #> uint8: 0..255, the standard type for image pixels.
print(small + 10)                             #> Integer overflow WRAPS AROUND silently: 260 -> 4.
```

:::bug Image maths gives weird numbers
Pixel arrays are often `uint8`, and arithmetic on them silently wraps around past 255. Convert first: `img = img.astype(np.float32) / 255.0`.
:::

## Reshaping {B}

```py title="reshape, flatten, transpose, new axes"
a = np.arange(12)                       #> 0..11, shape (12,).
B = a.reshape(3, 4)                     #> Reinterpret as 3 rows x 4 columns. Total size must match (3*4 = 12).
print(B)  #> Display the value of `B`.
print(a.reshape(2, -1).shape)           #> -1 means "work this dimension out for me": (2, 6).
print(B.T.shape)                        #> `.T` transposes: rows become columns, giving (4, 3).
print(B.ravel())                        #> Flatten back to 1-D (returns a view when possible).
v = np.array([1, 2, 3])  #> A 1-D array of shape (3,).
print(v.shape, v[:, np.newaxis].shape, v[np.newaxis, :].shape)   #> Add an axis: column vector (3,1) or row vector (1,3).
print(v.reshape(-1, 1).shape)           #> The same column vector: the shape scikit-learn wants for ONE feature.
```

```py error title="Sizes must match"
np.arange(10).reshape(3, 4)      #> 10 elements cannot fill a 3x4 (=12) grid.
```

:::bug scikit-learn: `Expected 2D array, got 1D array instead`
Models expect `X` with shape `(n_samples, n_features)`. A single feature stored as shape `(n,)` must become `(n, 1)`: use `x.reshape(-1, 1)`, or select it from a DataFrame as `df[["col"]]` (double brackets).
:::

## Indexing and slicing {B}

```py title="Indexing 1-D and 2-D arrays"
A = np.arange(1, 13).reshape(3, 4)      #> [[1..4], [5..8], [9..12]]
print(A)  #> Display the value of `A`.
print(A[0, 2])                          #> Row 0, column 2 -> 3. (Use one bracket with a comma, not A[0][2].)
print(A[1])                             #> A whole row: [5 6 7 8].
print(A[:, 1])                          #> A whole column: every row (:), column 1 -> [2 6 10].
print(A[0:2, 1:3])                      #> Rows 0-1, columns 1-2: a 2x2 block.
print(A[-1, ::-1])                      #> Last row, reversed.
print(A[:, [0, 3]])                     #> *Fancy indexing*: a list of column indices picks columns 0 and 3.
```

### Boolean masks: the most useful trick in NumPy {B}

:::def
**Boolean mask**: an array of `True`/`False` values of the same shape as the data, used to select the positions where it is `True`.

**Fancy indexing**: selecting elements with a list or array of indices, such as `a[[0, 3, 5]]`. It always returns a copy.
:::


A comparison on an array produces an array of True/False. Using that as an index selects only the True positions.

```py title="Filtering with boolean masks"
temps = np.array([18.5, 22.1, 30.4, 15.0, 27.8, 33.2])  #> Six daily temperatures.
mask = temps > 25                       #> Element-wise comparison -> boolean array.
print(mask)  #> Display the value of `mask`.
print(temps[mask])                      #> Keep only the values where the mask is True.
print(temps[(temps > 20) & (temps < 30)])   #> Combine with & (and), | (or), ~ (not). Parentheses are REQUIRED.
print(mask.sum(), mask.mean())          #> True counts as 1: the number and the fraction of hot days.
temps[temps < 16] = 16                  #> Assign through a mask: clip low values in place.
print(temps)  #> Display the value of `temps`.
print(np.where(temps > 25, "hot", "ok"))  #> `np.where(cond, a, b)`: element-wise if/else.
print(np.flatnonzero(temps > 25))       #> The INDICES where the condition holds.
```

:::warn Use `&` and `|`, not `and` / `or`, on arrays
`(a > 1) and (a < 5)` raises `ValueError: The truth value of an array with more than one element is ambiguous`. Python's `and` wants a single True/False. Use the element-wise operators `&`, `|`, `~` and wrap each comparison in parentheses.
:::

## Views vs copies {I}

:::def
**View**: a new array object that looks at the same memory as another array. Changing a view changes the original.
:::


Basic slicing returns a **view**: a window onto the same memory. Changing the view changes the original. Fancy indexing and boolean masks return **copies**.

```py title="When does modifying a slice change the original?"
a = np.arange(6)  #> [0 1 2 3 4 5]
s = a[1:4]                  #> A view: no data copied.
s[0] = 100                  #> Writing into the view...
print(a)                    #> ...changes `a` as well.
c = a[1:4].copy()           #> Explicit copy: independent data.
c[0] = -1  #> Change the copy.
print(a, c)  #> `a` is unaffected by changes to `c`.
f = a[[0, 1]]               #> Fancy indexing always makes a copy.
f[0] = 999  #> Change the fancy-indexed copy; `a` is untouched.
print(a[0], np.shares_memory(a, s), np.shares_memory(a, f))   #> `shares_memory` tells you whether two arrays overlap.
```

## Combining and splitting arrays {I}

```py title="Stacking and splitting"
x = np.array([[1, 2], [3, 4]])  #> A 2x2 array.
y = np.array([[5, 6]])  #> A 1x2 array (one row).
print(np.concatenate([x, y], axis=0))          #> Join along rows (axis 0). Other dimensions must match.
print(np.vstack([x, y]))                       #> Same as above: vertical stack.
print(np.hstack([x, x]))                       #> Side by side (axis 1).
print(np.column_stack([[1, 2, 3], [4, 5, 6]])) #> Turn 1-D arrays into columns of a 2-D array: building a feature matrix.
print(np.stack([x, x]).shape)                  #> `stack` adds a NEW axis: two (2,2) arrays -> (2, 2, 2).
top, bottom = np.split(np.arange(8), 2)         #> Split into equal parts.
print(top, bottom)  #> [0 1 2 3] and [4 5 6 7].
```

## Understanding axes {B}

:::def
**Axis**: one dimension of an array. In a 2-D array, `axis=0` runs down the rows and `axis=1` runs across the columns. Aggregating along an axis collapses it.
:::


`axis=0` means "operate down the rows" (collapsing them, so you get one result per column). `axis=1` means "operate across columns" (one result per row).

```py title="Aggregating along an axis"
scores = np.array([[70, 80, 90],     #> 3 students (rows) x 3 exams (columns).
                   [60, 65, 70],  #> Student 2.
                   [95, 85, 75]])  #> Student 3; the closing brackets end the array.
print(scores.sum())                  #> No axis: everything collapses to one number.
print(scores.mean(axis=0))           #> Mean of each COLUMN (each exam), averaged over students.
print(scores.mean(axis=1))           #> Mean of each ROW (each student), averaged over exams.
print(scores.max(axis=1), scores.argmax(axis=1))   #> Best score per student and WHICH exam it was in.
print(scores.mean(axis=0, keepdims=True).shape)    #> keepdims keeps the collapsed axis with length 1: (1, 3). Handy for broadcasting.
```

```py figonly caption="axis=0 collapses rows (one value per column); axis=1 collapses columns (one value per row)."
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
data = np.array([[70, 80, 90], [60, 65, 70], [95, 85, 75]])
for ax, axis, title, col in [(axes[0], 0, "scores.mean(axis=0)", "#7b3ff2"), (axes[1], 1, "scores.mean(axis=1)", "#e0245e")]:
    ax.set_xlim(-0.6, 4.2 if axis == 1 else 2.6); ax.set_ylim(-1.7 if axis == 0 else -0.6, 2.6); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", family="monospace")
    means = data.mean(axis=axis)
    for i in range(3):
        for j in range(3):
            k = j if axis == 0 else i
            ax.add_patch(plt.Rectangle((j - 0.45, 2 - i - 0.45), 0.9, 0.9, color=col, alpha=0.10 + 0.12 * k, ec="white"))
            ax.text(j, 2 - i, data[i, j], ha="center", va="center", fontsize=12)
    if axis == 0:
        for j in range(3):
            ax.annotate("", xy=(j, -0.75), xytext=(j, -0.5), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.add_patch(plt.Rectangle((j - 0.45, -1.55), 0.9, 0.75, color=col))
            ax.text(j, -1.18, f"{means[j]:.1f}", ha="center", va="center", color="white", fontweight="bold")
    else:
        for i in range(3):
            ax.annotate("", xy=(2.95, 2 - i), xytext=(2.5, 2 - i), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.add_patch(plt.Rectangle((3.0, 2 - i - 0.38), 1.0, 0.76, color=col))
            ax.text(3.5, 2 - i, f"{means[i]:.1f}", ha="center", va="center", color="white", fontweight="bold")
plt.tight_layout(); plt.show()
```

:::ex Try it yourself
1. Create a 5x5 array of the numbers 1–25 and extract its centre 3x3 block.
2. From `x = np.random.default_rng(0).normal(size=1000)`, compute the fraction of values more than 2 standard deviations from the mean (expect about 0.05).
3. Reshape `np.arange(24)` into shape `(2, 3, 4)` and compute the sum along each axis. Predict each result's shape first.
:::
