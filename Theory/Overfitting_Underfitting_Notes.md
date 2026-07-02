# Overfitting & Underfitting

> The bias-variance tradeoff in practice — why models fail to generalize, and how to fix it.

---

## 1. Core Definitions

| Term | Definition |
|---|---|
| **Overfitting** | Model learns the training data *too well* — including noise and outliers — and fails to generalize to unseen data. Low train error, high test error. |
| **Underfitting** | Model is too simple to capture the underlying pattern in the data. High train error, high test error. |
| **Good fit** | Model captures the true underlying signal without memorizing noise. Low train error, low test error, small gap between them. |

**One-line intuition**: Underfitting = model hasn't learned enough. Overfitting = model has memorized instead of learned.

---

## 2. Bias-Variance Tradeoff

Total error of a model can be decomposed as:

```
Total Error = Bias² + Variance + Irreducible Error
```

- **Bias** = error from overly simplistic assumptions (model can't capture true pattern). High bias → underfitting.
- **Variance** = error from sensitivity to small fluctuations in training data (model changes a lot if trained on a different sample). High variance → overfitting.
- **Irreducible error** = noise inherent to the data itself — no model can remove this.

| | Bias | Variance |
|---|---|---|
| Underfitting | High | Low |
| Overfitting | Low | High |
| Good fit | Low | Low |

There's a tradeoff: reducing bias (more complex model) tends to increase variance, and vice versa. The goal is finding the **sweet spot** that minimizes total error.

---

## 3. How to Detect Overfitting vs Underfitting

### Train vs Test error comparison

| Scenario | Train Error | Test Error | Gap |
|---|---|---|---|
| Underfitting | High | High | Small |
| Overfitting | Low | High | Large |
| Good fit | Low | Low | Small |

### Learning curves (the real diagnostic tool)

Plot training error and validation error against training set size (or epochs/iterations):

```python
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    estimator=model, X=X, y=y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error'
)

train_mean = -train_scores.mean(axis=1)
val_mean = -val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label='Train error')
plt.plot(train_sizes, val_mean, label='Validation error')
plt.xlabel('Training set size')
plt.ylabel('Error')
plt.legend()
plt.show()
```

**How to read it:**
- **Underfitting**: both curves converge to a high error value and stay close together — more data won't help; the model itself is too weak.
- **Overfitting**: large persistent gap between train (low) and validation (high) error, even with more data — model is too complex or needs regularization.
- **Good fit**: both curves converge to a low error value with a small gap.

---

## 4. Causes

### Overfitting causes
- Model too complex relative to amount/complexity of data (e.g., deep decision tree, high-degree polynomial, huge neural net on small data).
- Too many features relative to number of samples (curse of dimensionality).
- Training for too long (too many epochs in neural nets — starts memorizing).
- Noisy or insufficient training data.
- No regularization.

### Underfitting causes
- Model too simple (e.g., linear model on nonlinear data).
- Too few features / poor feature engineering.
- Excessive regularization (penalizing complexity too hard).
- Not training long enough.
- High bias assumptions baked into the model choice.

---

## 5. Fixes

### Fixing Overfitting (reduce variance)

1. **Get more training data** — most effective if feasible; harder for model to memorize a larger, more diverse dataset.
2. **Regularization**:
   - L1 (Lasso) — pushes some coefficients to exactly zero → feature selection.
   - L2 (Ridge) — shrinks coefficients smoothly, discourages large weights.
   - Elastic Net — combination of L1 + L2.
   ```python
   from sklearn.linear_model import Ridge, Lasso
   ridge = Ridge(alpha=1.0)   # alpha = regularization strength
   lasso = Lasso(alpha=0.1)
   ```
3. **Reduce model complexity**:
   - Prune decision trees (`max_depth`, `min_samples_leaf`).
   - Reduce number of layers/neurons in a neural net.
   - Lower polynomial degree in polynomial regression.
4. **Cross-validation** — use k-fold CV to get a reliable estimate of generalization performance, not just a single train/test split.
5. **Early stopping** — stop training when validation error starts increasing (common in gradient boosting, neural nets).
6. **Dropout** (neural nets) — randomly deactivate neurons during training to prevent co-adaptation.
7. **Ensemble methods** — bagging (e.g., Random Forest) averages out variance across many high-variance models.
8. **Feature selection / dimensionality reduction** — remove irrelevant/redundant features (PCA, correlation filtering, feature importance from tree models).
9. **Data augmentation** — especially in images/text, artificially expand training diversity.

### Fixing Underfitting (reduce bias)

1. **Increase model complexity** — deeper trees, more layers/neurons, higher polynomial degree, switch to a more expressive algorithm (e.g., linear regression → gradient boosting).
2. **Add more/better features** — feature engineering, interaction terms, polynomial features.
3. **Reduce regularization strength** — lower `alpha` in Ridge/Lasso.
4. **Train longer** — more epochs/iterations if the model hasn't converged.
5. **Remove noise-inducing constraints** — e.g., don't over-prune trees.

---

## 6. Visual Intuition — Polynomial Regression Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

np.random.seed(42)
X = np.sort(np.random.rand(30, 1) * 6 - 3, axis=0)
y = 0.5 * X.ravel()**2 + X.ravel() + 2 + np.random.randn(30) * 0.5

degrees = [1, 4, 15]   # underfit, good fit, overfit
titles = ['Underfitting (degree=1)', 'Good Fit (degree=4)', 'Overfitting (degree=15)']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)

for ax, degree, title in zip(axes, degrees, titles):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X_test)
    ax.scatter(X, y, color='black', s=15, label='Data')
    ax.plot(X_test, y_pred, color='red', label=f'Degree {degree}')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()
```

- **Degree 1** (linear): straight line, can't capture the curve → underfit.
- **Degree 4**: smooth curve that matches the true quadratic-ish trend → good fit.
- **Degree 15**: wildly oscillating curve threading through every training point exactly, including noise → overfit.

---

## 7. Regularization Math (Quick Reference)

**Ridge (L2)**:
```
Loss = MSE + λ * Σ(w_i²)
```
Shrinks weights smoothly toward zero, never exactly zero. Good when most features are somewhat useful.

**Lasso (L1)**:
```
Loss = MSE + λ * Σ|w_i|
```
Can shrink some weights to exactly zero → automatic feature selection. Good when you suspect many features are irrelevant.

**Elastic Net**:
```
Loss = MSE + λ1 * Σ|w_i| + λ2 * Σ(w_i²)
```
Balances both — useful when features are correlated (Lasso alone can behave erratically with correlated features).

`λ` (or `alpha` in sklearn) controls regularization strength:
- `λ = 0` → no regularization (equivalent to plain linear regression).
- `λ → ∞` → all weights shrink toward zero → underfitting.

---

## 8. Algorithm-Specific Knobs

| Algorithm | Overfitting knobs (reduce complexity) | Underfitting knobs (increase complexity) |
|---|---|---|
| Decision Tree | `max_depth ↓`, `min_samples_leaf ↑`, pruning | `max_depth ↑` |
| Random Forest | `max_depth ↓`, more trees (usually helps, not a big overfit risk) | `n_estimators ↑`, `max_depth ↑` |
| Gradient Boosting (XGBoost) | `max_depth ↓`, `learning_rate ↓` + early stopping, `subsample < 1` | `n_estimators ↑`, `max_depth ↑`, `learning_rate ↑` |
| KNN | `k ↑` (smoother boundary) | `k ↓` (more flexible boundary) |
| Neural Network | Dropout, weight decay, early stopping, fewer layers | More layers/neurons, train longer |
| SVM | `C ↓` (softer margin) | `C ↑` (harder margin, more complex boundary) |

---

## 9. Common Interview Questions

1. **What's the difference between bias and variance?**
   Bias = error from wrong assumptions (model too simple). Variance = error from sensitivity to training data fluctuations (model too complex/sensitive).

2. **How do you detect overfitting without a test set?**
   Use k-fold cross-validation on the training set, or hold out a validation split — track train vs validation error gap.

3. **Why does more data help with overfitting but not underfitting?**
   Overfitting = model memorizing noise; more diverse data makes memorization harder and forces genuine pattern learning. Underfitting = model is fundamentally too weak; more data of the same kind won't add expressive power.

4. **What is early stopping and why does it help?**
   Stopping training once validation error starts rising (even as train error keeps falling) — prevents the model from continuing to memorize noise in later epochs/iterations.

5. **Why is cross-validation better than a single train/test split for detecting overfitting?**
   A single split gives one noisy estimate of generalization error; k-fold CV averages over multiple splits, giving a more reliable estimate and reducing the chance of a lucky/unlucky split hiding overfitting.

6. **Does a Random Forest overfit like a single Decision Tree?**
   Individual trees overfit (high variance), but averaging many decorrelated trees (bagging + feature randomness) cancels out variance — so Random Forests are much more robust to overfitting than a single deep tree.

---

## 10. Quick Diagnostic Checklist

```
Train error high, Test error high, small gap        → Underfitting → increase complexity
Train error low,  Test error high, large gap         → Overfitting  → regularize / simplify / more data
Train error low,  Test error low,  small gap          → Good fit     → ship it
```

---

## 11. Relevance to Current Projects

- **Legendary Pokémon Classifier (Random Forest/XGBoost)**: with a likely small/imbalanced dataset (few legendaries vs many normal Pokémon), overfitting risk is high — watch train vs CV accuracy gap closely, use `max_depth`, `min_samples_leaf`, and class-weighting rather than just raw accuracy.
- **K-Means clustering**: doesn't have train/test overfitting in the classic sense, but analogous issue is choosing too many clusters (k) and "overfitting" to noise in stat distributions — use the elbow method / silhouette score, same spirit as bias-variance tradeoff.
- General workflow: after model training, always plot learning curves before trusting a single accuracy number — ties directly into the hyperparameter tuning weak point already flagged.

---

*Notes compiled for ML fundamentals revision — pairs with PCA notes and statistics revision notes in the same repo.*
