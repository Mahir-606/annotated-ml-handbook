# Unsupervised learning: clustering and dimensionality reduction

Without a target variable, the questions change: are there natural groups in the data? Can its many dimensions be summarised by a few? Unsupervised methods answer these questions, and they are also essential tools for exploring data before supervised modelling.

## k-means clustering {B}

:::def
**Clustering**: grouping observations so that those in the same group (cluster) are more similar to each other than to those in other groups.

**k-means**: a clustering algorithm that places k centres and alternately assigns points to the nearest centre and moves each centre to the mean of its points.

**Centroid**: the centre (mean position) of a cluster.
:::


k-means places *k* centres and assigns every point to its nearest centre, then moves each centre to the mean of its points, repeating until nothing changes. It finds compact, roughly spherical clusters of similar size.

```py title="Clustering synthetic blobs" caption="k-means recovers the four groups. The crosses are the learned cluster centres (cluster_centers_)."
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.
from sklearn.datasets import make_blobs  #> Import `make_blobs` from scikit-learn's `datasets` module.
from sklearn.cluster import KMeans  #> Import `KMeans` from scikit-learn's `cluster` module.
from sklearn.preprocessing import StandardScaler  #> Import `StandardScaler` from scikit-learn's `preprocessing` module.

X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.9, random_state=7)   #> y_true is only used to check our work later.
Xs = StandardScaler().fit_transform(X)                    #> Scale first: k-means uses Euclidean distance.
km = KMeans(n_clusters=4, n_init=10, random_state=0)      #> n_init=10: run 10 random starts and keep the best (lowest inertia).
labels = km.fit_predict(Xs)                               #> Fit and return the cluster label of each point.
print("inertia (within-cluster sum of squares):", round(km.inertia_, 2))  #> Lower inertia = tighter clusters.
print("cluster sizes:", np.bincount(labels))  #> Points per cluster.
fig, ax = plt.subplots(figsize=(6, 4.4))  #> One panel.
ax.scatter(Xs[:, 0], Xs[:, 1], c=labels, cmap="tab10", s=12, alpha=0.8)  #> Points coloured by cluster label.
ax.scatter(*km.cluster_centers_.T, marker="X", s=220, c="black", edgecolor="white")   #> `*arr.T` unpacks the two coordinate columns into x and y.
ax.set(title="k-means with k = 4", xlabel="feature 1 (scaled)", ylabel="feature 2 (scaled)")  #> Labels.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

### Choosing k: elbow and silhouette {I}

:::def
**Inertia**: the sum of squared distances from each point to its cluster centre. Lower means tighter clusters.

**Silhouette score**: for each point, how much closer it is to its own cluster than to the next nearest one, averaged. It runs from −1 to 1; higher is better.
:::


```py title="Two diagnostics side by side" caption="Inertia always decreases with k; look for the 'elbow' where gains flatten. The silhouette score (−1 to 1, higher is better) peaks at the best-separated clustering, here at k = 4."
from sklearn.metrics import silhouette_score  #> Import `silhouette_score` from scikit-learn's `metrics` module.

ks = range(2, 10)  #> Candidate values of k.
inertias, sils = [], []  #> Lists for both diagnostics.
for k in ks:  #> Loop over `ks`, naming each item `k`.
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)  #> Fit k-means with this k.
    inertias.append(m.inertia_)  #> Record inertia.
    sils.append(silhouette_score(Xs, m.labels_))          #> How much closer each point is to its own cluster than to the next one, averaged.
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))  #> Two panels.
axes[0].plot(ks, inertias, marker="o"); axes[0].set(title="Elbow method", xlabel="k", ylabel="inertia")  #> Inertia vs k.
axes[1].plot(ks, sils, marker="o", color="#e0245e"); axes[1].set(title="Silhouette score", xlabel="k", ylabel="mean silhouette")  #> Silhouette vs k.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
print("best k by silhouette:", list(ks)[int(np.argmax(sils))])  #> The k with the highest silhouette.
```

:::warn Clusters always "exist"
k-means will happily split pure noise into *k* groups. Validate clusters: check stability across random seeds and subsamples, inspect what distinguishes them (profile the features per cluster), and ask whether they mean anything in your domain.
:::

## Other clustering algorithms {I}

```py title="When k-means fails: non-spherical shapes" caption="On two interleaved moons, k-means cuts straight across the shapes; DBSCAN (density-based) and single-linkage agglomerative clustering follow them. DBSCAN labels any outlier −1 (it would be drawn grey; there are none in this clean example)."
from sklearn.datasets import make_moons  #> Import `make_moons` from scikit-learn's `datasets` module.
from sklearn.cluster import DBSCAN, AgglomerativeClustering  #> Import `DBSCAN` and `AgglomerativeClustering` from scikit-learn's `cluster` module.
from sklearn.mixture import GaussianMixture  #> Import `GaussianMixture` from scikit-learn's `mixture` module.

Xm, _ = make_moons(n_samples=400, noise=0.07, random_state=0)  #> Two interleaved half-moons (labels discarded with `_`).
Xm = StandardScaler().fit_transform(Xm)  #> Scale.
algos = {  #> Name -> clustering algorithm.
    "KMeans (k=2)": KMeans(n_clusters=2, n_init=10, random_state=0),  #> k-means with 2 clusters.
    "GaussianMixture (2)": GaussianMixture(n_components=2, random_state=0),            #> Soft, elliptical clusters with probabilities.
    "DBSCAN (eps=0.3)": DBSCAN(eps=0.3, min_samples=5),                                #> eps = neighbourhood radius; no need to choose k.
    "Agglomerative (single)": AgglomerativeClustering(n_clusters=2, linkage="single"), #> Merges nearest clusters bottom-up.
}  #> Close the dictionary opened above.
fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))  #> Four panels.
for ax, (name, algo) in zip(axes, algos.items()):  #> Loop over `zip(axes, algos.items())`, naming each item `ax, (name, algo)`.
    lab = algo.fit_predict(Xm)  #> Cluster labels from this algorithm.
    colors = np.where(lab == -1, "lightgrey", np.array(["#1f4fff", "#e0245e", "#0f9b62", "#ff6a00"])[lab % 4])  #> Grey for noise (-1), a palette colour otherwise.
    ax.scatter(Xm[:, 0], Xm[:, 1], c=colors, s=10)  #> Plot the points.
    ax.set_title(name, fontsize=11); ax.set_xticks([]); ax.set_yticks([])  #> Title; hide ticks (the units are meaningless).
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

| Algorithm | Finds | Need to set | Notes |
|---|---|---|---|
| `KMeans` | spherical, similar-sized clusters | `n_clusters` | fast; scale features; `MiniBatchKMeans` for big data |
| `GaussianMixture` | elliptical clusters, soft membership | `n_components` | choose by BIC/AIC (`.bic(X)`) |
| `DBSCAN` / `HDBSCAN` | arbitrary shapes, labels noise | `eps`, `min_samples` (HDBSCAN: `min_cluster_size`) | no k needed; struggles with varying density (HDBSCAN copes better) |
| `AgglomerativeClustering` | hierarchy of clusters | `n_clusters` or `distance_threshold`, `linkage` | dendrograms via SciPy |

### Evaluating clusters when labels exist {I}

```py title="External and internal validation"
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  #> Import `adjusted_rand_score` and `normalized_mutual_info_score` from scikit-learn's `metrics` module.
print("ARI:", round(adjusted_rand_score(y_true, labels), 3))          #> Agreement with true groups, corrected for chance: 1 = perfect, ~0 = random.
print("NMI:", round(normalized_mutual_info_score(y_true, labels), 3))  #> Normalised mutual information: shared information between labelings, 0 to 1.
print("silhouette:", round(silhouette_score(Xs, labels), 3))          #> Internal metric: needs no labels.
```

## Principal component analysis (PCA) {I}

:::def
**Dimensionality reduction**: representing data with fewer features while keeping as much useful information as possible.

**PCA (principal component analysis)**: finding new, uncorrelated axes (principal components) that capture the most variance, ordered from most to least.

**Explained variance ratio**: the share of the total variance captured by each principal component.
:::


PCA finds new axes (principal components) that are uncorrelated linear combinations of the original features, ordered by how much variance they capture. Keeping the first few gives a compact summary of the data.

```py title="PCA on the breast-cancer features" caption="Left: the first two components separate benign and malignant tumours well, even though PCA never saw the labels. Right: cumulative explained variance tells you how many components you need."
from sklearn.datasets import load_breast_cancer  #> Import `load_breast_cancer` from scikit-learn's `datasets` module.
from sklearn.decomposition import PCA  #> Import `PCA` from scikit-learn's `decomposition` module.
from sklearn.pipeline import make_pipeline  #> Import `make_pipeline` from scikit-learn's `pipeline` module.

bc = load_breast_cancer()  #> Load the dataset (a Bunch with .data, .target, .feature_names).
pca_pipe = make_pipeline(StandardScaler(), PCA())               #> ALWAYS standardise before PCA, or large-unit features dominate.
Z = pca_pipe.fit_transform(bc.data)                              #> Z: the data in principal-component coordinates.
pca = pca_pipe.named_steps["pca"]  #> Reach the fitted PCA step.
print("variance explained by PC1, PC2:", pca.explained_variance_ratio_[:2].round(3))  #> Share of total variance captured by each component.
print("components for 95% variance:", np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)  #> Smallest number of components whose cumulative share reaches 95%.

fig, axes = plt.subplots(1, 2, figsize=(11, 4))  #> Two panels.
for cls, name, c in [(0, "malignant", "#e0245e"), (1, "benign", "#1f4fff")]:  #> Loop over `[(0, "malignant", "#e0245e"), (1, "benign", "#1f4fff")]`, naming each item `cls, name, c`.
    axes[0].scatter(Z[bc.target == cls, 0], Z[bc.target == cls, 1], s=10, alpha=0.6, label=name, color=c)  #> PC1 vs PC2 for one class.
axes[0].set(xlabel="PC1", ylabel="PC2", title="Data projected onto 2 components")  #> Labels.
axes[0].legend()  #> Legend.
axes[1].plot(range(1, 31), np.cumsum(pca.explained_variance_ratio_), marker="o", ms=3)  #> Cumulative explained variance.
axes[1].axhline(0.95, ls="--", color="grey"); axes[1].text(14, 0.9, "95%", color="grey")  #> Mark the 95% line.
axes[1].set(xlabel="number of components", ylabel="cumulative explained variance", title="Scree (cumulative)")  #> Labels.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

```py title="What does each component mean? (loadings)"
loadings = pd.DataFrame(pca.components_[:2].T, index=bc.feature_names, columns=["PC1", "PC2"])   #> Weight of each original feature in each component.
print(loadings["PC1"].abs().sort_values(ascending=False).head(5).round(3))   #> Features that drive PC1 the most.
```

```py title="PCA as a preprocessing step"
from sklearn.linear_model import LogisticRegression  #> Import `LogisticRegression` from scikit-learn's `linear_model` module.
from sklearn.model_selection import cross_val_score  #> Import `cross_val_score` from scikit-learn's `model_selection` module.
pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95), LogisticRegression(max_iter=1000))   #> A float n_components keeps enough PCs for 95% of the variance.
print("CV accuracy with PCA:", cross_val_score(pipe, bc.data, bc.target, cv=5).mean().round(4))  #> Accuracy with the compressed features.
```

## t-SNE and UMAP for visualisation {A}

:::def
**t-SNE / UMAP**: non-linear methods that map high-dimensional data to 2-D or 3-D for visualisation, preserving local neighbourhoods.
:::


PCA is linear. **t-SNE** (and the popular external package **UMAP**) produce non-linear 2-D maps that preserve local neighbourhoods. They are excellent for *looking* at high-dimensional data, such as images, embeddings or single-cell data.

```py title="t-SNE on 8x8 handwritten digits" caption="t-SNE groups the 64-pixel images by digit without seeing the labels (colours are added only for display)."
from sklearn.datasets import load_digits  #> Import `load_digits` from scikit-learn's `datasets` module.
from sklearn.manifold import TSNE  #> Import `TSNE` from scikit-learn's `manifold` module.

digits = load_digits()                                                      #> 1797 images of 8x8 pixels, flattened to 64 features.
emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(digits.data)   #> perplexity ~ effective number of neighbours.
fig, ax = plt.subplots(figsize=(6.5, 5.2))  #> One panel.
sc = ax.scatter(emb[:, 0], emb[:, 1], c=digits.target, cmap="tab10", s=6)  #> Each image as a dot, coloured by its true digit.
for d in range(10):  #> Loop over `range(10)`, naming each item `d`.
    cx, cy = np.median(emb[digits.target == d], axis=0)  #> Centre of this digit's points.
    ax.text(cx, cy, str(d), fontsize=14, fontweight="bold", ha="center", va="center",  #> Write the digit at that centre...
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=0.85))  #> ...inside a white circle.
ax.set(title="t-SNE of the digits dataset", xticks=[], yticks=[])  #> Title; no ticks.
ax.grid(False)  #> Turn off the grid lines on this panel.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

:::warn Don't over-read t-SNE plots
Distances *between* clusters and cluster *sizes* in t-SNE/UMAP maps are not meaningful, and the layout changes with perplexity and seed. Use them to explore, never as evidence by themselves; don't cluster on t-SNE coordinates or feed them to a model.
:::

## Anomaly detection {I}

:::def
**Anomaly detection**: identifying observations that differ markedly from the majority, such as fraud, faults or rare events.
:::


```py title="Isolation Forest flags unusual samples"
from sklearn.ensemble import IsolationForest  #> Import `IsolationForest` from scikit-learn's `ensemble` module.
rng = np.random.default_rng(0)  #> Seeded generator.
normal = rng.normal(0, 1, (300, 2))  #> 300 ordinary points.
odd = rng.uniform(-6, 6, (15, 2))                         #> 15 scattered anomalies.
Xa = np.vstack([normal, odd])  #> All 315 points together; the anomalies are the last 15.
iso = IsolationForest(contamination=0.05, random_state=0).fit(Xa)   #> contamination = expected share of anomalies.
flag = iso.predict(Xa)                                    #> +1 = normal, -1 = anomaly.
print("flagged:", (flag == -1).sum(), "| of the true anomalies caught:", (flag[-15:] == -1).sum(), "/ 15")  #> How many were flagged, and how many of the 15 planted anomalies.
```

:::ex Try it yourself
1. Cluster the insurance customers (scaled `age`, `bmi`, `charges`) with k-means and profile each cluster with `groupby(labels).mean()`.
2. Run PCA on the wine dataset and plot the first two components coloured by cultivar.
3. Compare t-SNE maps of the digits with perplexity 5, 30 and 100.
:::
