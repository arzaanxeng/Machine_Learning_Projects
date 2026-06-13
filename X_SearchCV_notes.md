# Hyperparameter Tuning — GridSearchCV, RandomSearchCV, Optuna

> Notes by Arzaan 

---

## 1. GridSearchCV

Tries **every single combination** of hyperparameters in the grid.
For each combination it runs K-Fold cross-validation and picks the combo with the highest average CV score.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights'    : ['uniform', 'distance'],
    'metric'     : ['euclidean', 'manhattan']
}

gs = GridSearchCV(
    estimator  = KNeighborsClassifier(),
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 1
)
gs.fit(X_train, y_train)

print("Best params :", gs.best_params_)
print("Best CV acc :", round(gs.best_score_, 4))
print("Test acc    :", round(gs.score(X_test, y_test), 4))
```

**Total fits** = (no. of combos) × cv  
Example: 4×2×2 = 16 combos × 5 folds = **80 fits**

**When to use** → small grid, fast model

**Drawbacks**
- Exponential explosion — adding one new param multiplies the search space
- Wastes time on unimportant params
- No intelligence — treats all combos equally

---

## 2. Why best_cv_score vs test_acc can differ

| Situation | Meaning |
|---|---|
| test_acc < best_cv_score | Normal. Small gap = fine. Large gap = overfitting. |
| test_acc > best_cv_score | Normal. Final model trained on full X_train (more data than any single CV fold). |

---

## 3. RandomSearchCV

Instead of trying every combination, randomly samples `n_iter` combinations and evaluates only those.

```python
from sklearn.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(
    estimator           = KNeighborsClassifier(),
    param_distributions = param_grid,
    n_iter              = 20,
    cv                  = 5,
    scoring             = 'accuracy',
    random_state        = 42
)
rs.fit(X_train, y_train)
```

**When to use** → large grid, expensive model, time constraint

**Drawbacks**
- Not exhaustive — may miss the best combo entirely
- No memory — trial 15 has no idea what happened in trials 1–14
- Different runs give different results without `random_state`

---

## 4. Why "expensive model" matters

"Expensive" = takes a long time to train. Has nothing to do with money.

```
KNN on 500 rows        →  0.01 seconds   →  cheap
Decision Tree          →  0.1  seconds   →  cheap
Random Forest          →  5    seconds   →  moderate
XGBoost on 1M rows     →  3    minutes   →  expensive
Neural Network (GPU)   →  2    hours     →  very expensive
LLM fine-tuning        →  3    days      →  extremely expensive
```

GridSearch with 100 combos on a 2-hour model = 200 hours.
Bayesian optimization tries 20–30 smart combos = 40–60 hours, still finds near-optimal result.

---

## 5. BayesianSearchCV — Optuna

Builds a **probability model** of which hyperparameters are likely to perform well.
Uses that model to smartly pick the next combo.
Gets better with every trial — early trials explore, later trials zoom into promising regions.

```python
# pip install optuna

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
    weights     = trial.suggest_categorical('weights', ['uniform', 'distance'])
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("Best params:", study.best_params)
print("Best score :", study.best_value)
```

**When to use** → expensive model, large grid, need near-optimal result

**Drawbacks**
- Sequential by nature — each trial depends on the previous one, hard to fully parallelize
- Overkill for cheap/fast models — probability model overhead makes it slower than GridSearch on simple problems
- Harder to debug — non-trivial to explain why it picked certain params
- Extra dependency — not part of sklearn, needs separate install (`pip install optuna`)

---

## 6. The Two-Stage Approach (most practical)

```
Stage 1 → RandomSearchCV with wide grid
          get a rough idea of which region works

Stage 2 → Optuna with narrow grid around that region
          fine-tune intelligently
```

This saves compute and finds a better result than either method alone.

---

## 7. One-Line Mental Model

| Method | Behaviour | Use when |
|---|---|---|
| GridSearchCV | tries everything | small grid, fast model |
| RandomSearchCV | tries random sample | large grid, time pressure |
| Optuna (Bayesian) | tries smart sample | expensive model, need best result |

---

## 8. What to say in an Interview

> "The choice depends on model complexity, grid size, and compute budget.
> For a small grid or fast model I use GridSearchCV — simple and exhaustive.
> For a large grid with time constraints I use RandomSearchCV — faster but blind.
> For expensive models like XGBoost or Neural Networks I use Optuna — it learns
> from each trial and finds near-optimal params without trying everything.
> But Optuna has tradeoffs too — it's sequential, overkill for simple problems,
> harder to debug, and adds an external dependency.
> In practice I'd run RandomSearch first to find a promising region, then use
> Optuna to fine-tune within that region."

---
