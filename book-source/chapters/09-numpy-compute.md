# Computing with NumPy: vectorization, broadcasting, statistics, linear algebra

With arrays in hand, this chapter covers the operations that do the real work: element-wise maths, broadcasting arrays of different shapes, statistics, random numbers and linear algebra.

## Universal functions (ufuncs) {B}

:::def
**Ufunc (universal function)**: a NumPy function that operates element by element on whole arrays, such as `np.sqrt`, `np.exp` and `np.log`.
:::


Ufuncs apply a function to every element in compiled code.

```py title="Element-wise maths"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
x = np.array([1.0, 4.0, 9.0, 16.0])  #> Perfect squares, so the square roots are whole numbers.
print(x + 1, x * 2, x ** 0.5)                 #> Arithmetic operators work element by element.
print(np.sqrt(x), np.log(x), np.exp([0, 1]))  #> Square root, natural log, exponential.
print(np.round(np.log10(x), 3))               #> log base 10, rounded to 3 decimals.
a, b = np.array([1, 5, 3]), np.array([4, 2, 3])  #> Two arrays of the same shape.
print(a + b, a * b, a > b, np.maximum(a, b))  #> Two arrays of the same shape combine pairwise.
print(np.clip(np.array([-2, 0.5, 3]), 0, 1))  #> Limit values to the range [0, 1].
print(np.abs([-3, 3]), np.sign([-3, 0, 3]))  #> Absolute values; sign gives -1, 0 or +1.
```

:::warn `*` is not matrix multiplication
For arrays, `A * B` multiplies element by element. Matrix multiplication is `A @ B` (or `np.matmul`, `np.dot`).
:::

## Broadcasting {I}

:::def
**Broadcasting**: NumPy's rule for combining arrays of different shapes: compare shapes from the right; dimensions must be equal or 1, and size-1 dimensions are stretched to match.
:::


**Broadcasting** lets NumPy combine arrays of different shapes without copying data. Compare the shapes from the **right**; two dimensions are compatible if they are **equal or one of them is 1**. A dimension of 1 is stretched to match.

```py title="Broadcasting in action"
X = np.array([[1, 2, 3],  #> First row of a (2, 3) array...
              [4, 5, 6]])                 #> Shape (2, 3).
print(X + 10)                             #> A scalar broadcasts to every element.
row = np.array([10, 20, 30])              #> Shape (3,), treated as (1, 3): stretched down the 2 rows.
print(X + row)  #> Each row gets 10, 20, 30 added.
col = np.array([[100], [200]])            #> Shape (2, 1): stretched across the 3 columns.
print(X + col)  #> Row 1 gets 100 added, row 2 gets 200.
mu = X.mean(axis=0)                       #> Column means, shape (3,).
sd = X.std(axis=0)                        #> Column standard deviations, shape (3,).
print((X - mu) / sd)                      #> Standardise every column in one line: (2,3) op (3,) -> (2,3).
```

```py figonly caption="Broadcasting a (2, 3) array with a (3,) row and a (2, 1) column. Faded cells are virtual copies: NumPy never actually duplicates the data."
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 2.9))
def grid(ax, x0, vals, color, faded=None, shape=None):
    rows, cols = vals.shape
    for i in range(rows):
        for j in range(cols):
            alpha = 0.35 if (faded is not None and faded[i, j]) else 1
            ax.add_patch(plt.Rectangle((x0 + j, rows - 1 - i), 0.92, 0.92, color=color, alpha=alpha))
            ax.text(x0 + j + 0.46, rows - 1 - i + 0.46, vals[i, j], ha="center", va="center", color="white", fontweight="bold", fontsize=11)
X = np.array([[1, 2, 3], [4, 5, 6]])
for ax, other, label in [(axes[0], np.array([[10, 20, 30]]), "(2,3) + (3,)"), (axes[1], np.array([[100], [200]]), "(2,3) + (2,1)")]:
    ax.set_xlim(-0.3, 13.5); ax.set_ylim(-0.6, 2.4); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title(label, loc="left", family="monospace", fontweight="bold")
    grid(ax, 0, X, "#7b3ff2")
    ax.text(3.4, 0.95, "+", fontsize=20, ha="center", va="center")
    big = np.broadcast_to(other, (2, 3))
    faded = np.ones((2, 3), bool)
    if other.shape == (1, 3): faded[0, :] = False
    else: faded[:, 0] = False
    grid(ax, 4.0, big, "#e0245e", faded)
    ax.text(7.5, 0.95, "=", fontsize=20, ha="center", va="center")
    grid(ax, 8.2, X + big, "#0f9b62")
plt.tight_layout(); plt.show()
```

```py error title="Incompatible shapes"
X = np.ones((2, 3))  #> A (2, 3) array of ones.
v = np.array([1, 2])        #> Shape (2,) is compared with X's LAST dimension (3): 2 != 3 and neither is 1.
X + v                       #> Fix: v[:, np.newaxis] gives shape (2, 1), which broadcasts across columns.
```

:::bug `operands could not be broadcast together with shapes (a,b) (c,)`
Write both shapes one above the other, right-aligned. Every column must be equal or contain a 1. Usually the fix is `arr.reshape(-1, 1)` or `arr[:, None]` to turn a 1-D array into a column.
:::

### Broadcasting power move: pairwise distances {A}

```py title="All pairwise Euclidean distances, no loops"
P = np.array([[0, 0], [3, 4], [6, 8]])       #> 3 points in 2-D, shape (3, 2).
diff = P[:, None, :] - P[None, :, :]         #> (3,1,2) - (1,3,2) broadcasts to (3,3,2): every pair's difference.
D = np.sqrt((diff ** 2).sum(axis=-1))        #> Square, sum over the coordinate axis, square root -> (3,3) distance matrix.
print(D)                                     #> This is the heart of k-nearest-neighbours.
```

## Statistics and aggregation {B}

:::def
**Mean, median, standard deviation**: the mean is the average; the median is the middle value when sorted; the standard deviation (SD) measures how spread out values are around the mean.

**Percentile / quartile**: the value below which a given percentage of observations fall. The quartiles are the 25th, 50th (median) and 75th percentiles.

**Correlation**: a number from −1 to 1 measuring how strongly two variables move together linearly (Pearson's r). 0 means no linear relationship.
:::


```py title="Descriptive statistics"
rng = np.random.default_rng(seed=0)           #> The modern random generator. Always pass a seed for reproducibility.
x = rng.normal(loc=50, scale=10, size=1000)   #> 1000 draws from a normal distribution, mean 50, sd 10.
print(f"mean={x.mean():.2f}  median={np.median(x):.2f}  std={x.std(ddof=1):.2f}")   #> ddof=1 -> sample std (divides by n-1).
print(f"min={x.min():.1f} max={x.max():.1f} range={np.ptp(x):.1f}")                 #> ptp = "peak to peak" = max - min.
print("quartiles:", np.percentile(x, [25, 50, 75]).round(2))  #> 25th, 50th (median) and 75th percentiles.
print("cumsum:", np.cumsum([1, 2, 3, 4]))  #> Running total: [1 3 6 10].
y = 2 * x + rng.normal(0, 5, size=1000)       #> A variable correlated with x.
print("corr matrix:\n", np.corrcoef(x, y).round(3))   #> Pearson correlation matrix of the two variables.
```

:::warn `np.std` uses ddof=0 by default; pandas uses ddof=1
NumPy's default divides by *n* (population SD); pandas' `.std()` divides by *n−1* (sample SD). Small samples give noticeably different answers. Be explicit: `np.std(x, ddof=1)`.
:::

## Sorting, searching, unique values {B}

```py title="Order and counts"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])  #> An unsorted array with a repeated value.
print(np.sort(a))                                 #> Returns a sorted copy.
print(np.argsort(a))                              #> The INDICES that would sort the array: use them to reorder other arrays the same way.
print(a[np.argsort(a)[::-1][:3]])                 #> Top-3 values via argsort reversed.
vals, counts = np.unique(a, return_counts=True)   #> Unique values and how often each occurs: class balance at a glance.
print(vals, counts)  #> Each distinct value and its count.
print(np.searchsorted([10, 20, 30], 25))          #> Where 25 would be inserted to keep order: index 2 (binning).
print(np.argmax(a), np.argmin(a))  #> Positions of the largest and the smallest value.
```

## Missing values: NaN {I}

:::def
**NaN (not a number)**: a special float value marking a missing or undefined result. Any arithmetic with NaN gives NaN, and NaN is not equal to anything, not even itself.
:::


```py title="Detecting and handling NaN"
d = np.array([1.0, np.nan, 3.0, np.nan, 5.0])     #> np.nan marks a missing float value.
print(d.mean())                                   #> Any NaN poisons ordinary aggregates: result is nan.
print(np.nanmean(d), np.nansum(d), np.nanmax(d))  #> The nan-functions ignore missing values.
print(np.isnan(d), np.isnan(d).sum())             #> Detect missing values and count them.
print(d[~np.isnan(d)])                            #> Drop them using a negated mask.
print(np.nan_to_num(d, nan=0.0))                  #> Replace them with a fixed value.
print(np.nan == np.nan)                           #> NaN is not equal to anything, not even itself! Always use np.isnan().
```

## Random numbers {B}

```py title="The Generator API (use this, not np.random.seed)"
rng = np.random.default_rng(42)                  #> One generator object, seeded once, used everywhere.
print(rng.random(3))                             #> Uniform floats in [0, 1).
print(rng.integers(0, 10, size=5))               #> Integers in [0, 10): the high end is excluded.
print(rng.normal(0, 1, size=(2, 3)).round(2))    #> Standard normal values in a 2x3 array.
print(rng.choice(["A", "B", "C"], size=5, p=[0.6, 0.3, 0.1]))   #> Weighted random choice.
idx = rng.permutation(10)                        #> A shuffled copy of 0..9: the basis of random train/test splits.
print(idx)  #> Display the value of `idx`.
```

:::tip Old vs new random API
Older tutorials use `np.random.seed(0)` and `np.random.rand(...)`. That still works, but NumPy recommends `rng = np.random.default_rng(seed)` and calling methods on `rng`. It is faster, statistically better, and avoids hidden global state.
:::

## Linear algebra {I}

:::def
**Matrix multiplication**: combining two matrices by taking dot products of rows of the first with columns of the second; written `A @ B`. Requires `A.shape[1] == B.shape[0]`.

**Eigenvalue / eigenvector**: for a square matrix A, an eigenvector v keeps its direction when multiplied by A: Av = λv, where λ is the eigenvalue. The basis of PCA.
:::


```py title="Matrix operations ML is built on"
A = np.array([[2.0, 1.0],  #> A 2x2 symmetric matrix: first row...
              [1.0, 3.0]])  #> ...second row.
b = np.array([3.0, 5.0])  #> Right-hand side of the system A x = b.
print(A @ b)                                 #> Matrix-vector product.
print(A @ A)                                 #> Matrix-matrix product (rows of the first times columns of the second).
print(np.linalg.det(A))                      #> Determinant: about 5 (non-zero, so A is invertible).
x = np.linalg.solve(A, b)                    #> Solve Ax = b. More accurate and faster than computing inv(A) @ b.
print(x, np.allclose(A @ x, b))              #> Verify the solution; `allclose` compares floats with a tolerance.
vals, vecs = np.linalg.eigh(A)               #> Eigen-decomposition of a symmetric matrix (the maths behind PCA).
print(vals.round(4))  #> The two eigenvalues.
print(np.linalg.norm([3, 4]))                #> Euclidean length of a vector: 5.0.
U, S, Vt = np.linalg.svd(np.arange(6).reshape(2, 3).astype(float))   #> Singular value decomposition.
print(S.round(4))  #> The singular values, largest first.
```

```py error title="Inner dimensions must agree"
A = np.ones((3, 2))  #> Shape (3, 2).
B = np.ones((3, 2))  #> Also (3, 2): the inner dimensions (2 and 3) don't match.
A @ B                       #> (3,2) @ (3,2): inner dims 2 and 3 differ. A @ B.T works: (3,2) @ (2,3) -> (3,3).
```

:::ex Try it yourself
1. Min-max scale each column of a random `(100, 4)` matrix to [0, 1] using broadcasting.
2. Using `np.unique(..., return_counts=True)`, report the class proportions of `rng.choice([0, 1], 500, p=[.8, .2])`.
3. Solve the system 3x + 2y = 7, x − y = −1 with `np.linalg.solve`.
:::
