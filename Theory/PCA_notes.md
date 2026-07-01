# Principal Component Analysis (PCA)

> Dimensionality reduction technique used to compress high-dimensional data into fewer dimensions while retaining maximum variance (information).

---

## 1. Why PCA?

- **Curse of dimensionality**: too many features → sparse data, overfitting, slow training.
- **Visualization**: humans can only see 2D/3D — PCA lets you project 50D data down to 2D/3D for plotting.
- **Multicollinearity**: correlated features get combined into independent components.
- **Speed**: fewer features = faster downstream models (especially distance-based ones like K-Means, KNN).
- **Noise reduction**: low-variance directions are often noise; dropping them can clean up data.

**Core idea**: Find new axes (directions) in the data such that projecting the data onto these axes preserves as much variance as possible, and the axes are orthogonal (uncorrelated) to each other.

---

## 2. Intuition (No Math First)

Imagine a scatter plot of height vs weight — the points form a diagonal cloud, not a circle. That diagonal direction is the direction of **maximum spread**. PCA finds that direction (Principal Component 1), then finds the next best direction *perpendicular* to it (PC2), and so on.

- **PC1** = direction of maximum variance in the data.
- **PC2** = direction of maximum remaining variance, orthogonal to PC1.
- Each subsequent PC captures less variance than the previous one.

If your data lies almost entirely along PC1 and PC2, you can drop the rest with minimal information loss.

---

## 3. The Math

### Step-by-step algorithm

1. **Standardize the data** (mean = 0, variance = 1 per feature).
   - Critical: PCA is variance-sensitive. Features on larger scales dominate if not standardized.
   ```
   X_scaled = (X - mean(X)) / std(X)
   ```

2. **Compute the covariance matrix** of the standardized data.
   ```
   Cov(X) = (1 / (n-1)) * X_scaled.T @ X_scaled
   ```
   - This is a (features × features) matrix showing how each pair of features varies together.

3. **Compute eigenvectors and eigenvalues** of the covariance matrix.
   - Eigenvectors → directions of the new axes (principal components).
   - Eigenvalues → amount of variance explained by each corresponding eigenvector.

4. **Sort eigenvectors by eigenvalue**, descending. The top eigenvector is PC1, next is PC2, etc.

5. **Select top-k eigenvectors** to form a projection matrix `W` (features × k).

6. **Project the data**:
   ```
   X_pca = X_scaled @ W
   ```

### Alternative: SVD (what sklearn actually uses internally)

Instead of eigendecomposition of the covariance matrix (numerically less stable), PCA in practice uses **Singular Value Decomposition**:

```
X_scaled = U * S * V.T
```

- `V.T` rows = principal components (same as eigenvectors of covariance matrix).
- `S` (singular values) relate to eigenvalues: `eigenvalue_i = (S_i^2) / (n-1)`.
- SVD is preferred because it's numerically more stable and avoids explicitly computing the covariance matrix.

---

## 4. Explained Variance Ratio

Tells you what fraction of total variance each PC captures.

```
explained_variance_ratio_i = eigenvalue_i / sum(all eigenvalues)
```

Use a **cumulative explained variance plot (scree plot)** to decide how many components to keep — commonly keep enough PCs to retain 90–95% of variance.

---

## 5. Python Implementation (scikit-learn)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit PCA (keep all components first to inspect variance)
pca = PCA()
pca.fit(X_scaled)

# 3. Scree plot — decide number of components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()

# 4. Refit with chosen number of components (e.g., 95% variance)
pca_final = PCA(n_components=0.95)  # sklearn accepts a float = variance threshold
X_pca = pca_final.fit_transform(X_scaled)

print("Original shape:", X_scaled.shape)
print("Reduced shape:", X_pca.shape)
print("Components kept:", pca_final.n_components_)
```

### Manual (from scratch, NumPy only)

```python
import numpy as np

def pca_from_scratch(X, k):
    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    # Covariance matrix
    cov_matrix = np.cov(X_scaled.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top-k
    W = eigenvectors[:, :k]

    # Project
    X_pca = X_scaled @ W

    explained_variance_ratio = eigenvalues[:k] / eigenvalues.sum()
    return X_pca, explained_variance_ratio

X_reduced, var_ratio = pca_from_scratch(X, k=2)
print("Explained variance ratio:", var_ratio)
```

> Note: `np.linalg.eigh` (not `eig`) is used because the covariance matrix is symmetric — `eigh` is faster and numerically stable for symmetric matrices.

---

## 6. Key Properties

| Property | Detail |
|---|---|
| Unsupervised | Doesn't use target labels `y` |
| Linear | Only captures linear relationships between features (use Kernel PCA / t-SNE / UMAP for nonlinear structure) |
| Orthogonal components | PCs are always uncorrelated with each other |
| Variance-based | Assumes high variance = high information (not always true — variance ≠ importance) |
| Sensitive to scale | Must standardize features first |
| Reversible (lossy) | Can reconstruct approximate original data via `pca.inverse_transform()` |

---

## 7. When to Use PCA

✅ Good for:
- Preprocessing before distance-based models (K-Means, KNN, SVM) to fight the curse of dimensionality.
- Visualizing high-dimensional clusters in 2D/3D.
- Removing multicollinearity before linear regression.
- Compressing data / speeding up training.
- Noise reduction (dropping low-variance components).

❌ Avoid / be careful when:
- Features are already few and interpretable — PCA destroys interpretability (components are linear combinations, not original features).
- Relationships are strongly nonlinear — consider **t-SNE, UMAP, or Kernel PCA**.
- You need feature importance tied to *original* features for explainability (e.g., regulated domains).
- Data isn't well-standardized or has significant outliers (PCA is sensitive to outliers since it's variance-based).

---

## 8. PCA vs Other Techniques

| Technique | Type | Preserves | Use Case |
|---|---|---|---|
| PCA | Linear | Global variance | General dimensionality reduction |
| t-SNE | Nonlinear | Local structure/clusters | Visualization only, not for modeling downstream |
| UMAP | Nonlinear | Local + some global structure | Visualization, faster than t-SNE |
| LDA (Linear Discriminant Analysis) | Linear, supervised | Class separability | When you have labels and want max class separation |
| Autoencoders | Nonlinear (neural) | Learned compressed representation | Large datasets, deep learning pipelines |

---

## 9. Common Interview Questions

1. **Why do we standardize before PCA?**
   Because PCA is variance-driven — features with larger numeric ranges would dominate the principal components otherwise.

2. **What do eigenvalues represent in PCA?**
   The amount of variance captured along the corresponding eigenvector (principal component).

3. **Can PCA be used for supervised learning feature selection?**
   Not directly — PCA is unsupervised and ignores the target variable. Use LDA if you want label-aware reduction.

4. **What's the difference between PCA components and original features?**
   Original features have direct meaning (e.g., "height"). PCA components are linear combinations of all features and generally lose direct interpretability.

5. **How do you decide the number of components to keep?**
   Scree plot / cumulative explained variance — commonly the "elbow point" or a variance threshold like 95%.

6. **Does PCA remove multicollinearity?**
   Yes — since principal components are orthogonal (uncorrelated) by construction.

---

## 10. Quick Reference — sklearn API

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)          # keep 2 components
pca = PCA(n_components=0.95)       # keep enough components for 95% variance

X_pca = pca.fit_transform(X_scaled)
pca.explained_variance_ratio_      # variance explained per component
pca.components_                    # the eigenvectors (loadings)
pca.inverse_transform(X_pca)       # reconstruct approx original data
```

---

## 11. Relevance to Current Projects

- **Pokémon K-Means clustering**: PCA is a natural preprocessing step before K-Means — reduces stat features (HP, Attack, Defense, Sp.Atk, Sp.Def, Speed, etc.) to 2 components for both better clustering performance (K-Means struggles in high dimensions due to distance concentration) and for plotting clusters visually in 2D.
- General EDA workflow: after correlation analysis, PCA is a good next step to check if features are redundant/collapsible.

---

*Notes compiled for ML fundamentals revision — pairs well with the existing statistics revision notes (central tendency, dispersion, hypothesis testing).*
