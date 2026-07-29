# ML Pipeline + GridSearchCV Reference (post train_test_split)

Everything below assumes you already have:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 1. Full pipeline with mixed numeric + categorical columns

This is the version you'll use most often — real datasets rarely have only numeric columns.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Split columns by type
numeric_features = ['age', 'income']
categorical_features = ['city', 'gender']

# 2. Build a mini pipeline for each type
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 3. Combine both into a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. Chain preprocessing + model into one pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Fit on RAW training data — the pipeline handles all preprocessing internally
pipe.fit(X_train, y_train)

# 6. Predict — same preprocessing is automatically applied to test data
y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Why this matters:** you never manually call `.transform()` on X_test yourself. The whole point of wrapping preprocessing in the pipeline is that `pipe.fit()` and `pipe.predict()` apply the exact same transformation learned from training data — no leakage, no mismatch.

---

## 2. GridSearchCV on top of the pipeline

The one thing to memorize: **parameter names use `stepname__param`**, and this nests as deep as your pipeline does.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'preprocessor__num__imputer__strategy': ['mean', 'median']
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,      # use all CPU cores
    verbose=1
)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

best_model = grid_search.best_estimator_   # this is a full fitted pipeline
y_pred = best_model.predict(X_test)
```

**Reading the naming path:** `preprocessor__num__imputer__strategy` means:
`pipe` → step `preprocessor` (the ColumnTransformer) → its `num` transformer → its `imputer` step → the `strategy` param.

To inspect any step directly:
```python
pipe.named_steps['classifier']            # the RandomForestClassifier object
pipe.named_steps['preprocessor']           # the ColumnTransformer object
```

---

## 3. Simpler version — all-numeric data, quick prototyping

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
pipe.fit(X_train, y_train)
```
`make_pipeline` auto-generates lowercase step names from the class names (`standardscaler`, `randomforestclassifier`), so your GridSearchCV keys become `randomforestclassifier__n_estimators` etc. Prefer explicit `Pipeline(steps=[...])` when you need GridSearchCV — the custom names are easier to read in a param grid.

---

## 4. cross_val_score with a pipeline (no grid search, just evaluation)

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
print(scores.mean(), scores.std())
```

---

## 5. Common gotchas

- **Never** call `.fit_transform()` on the full dataset before splitting — that's data leakage. Always fit preprocessing only on `X_train` (the pipeline does this for you automatically).
- `OneHotEncoder(handle_unknown='ignore')` — without this, an unseen category in the test set crashes `.transform()`.
- For a bare `ColumnTransformer` (not inside a full pipeline with a model), use `.fit_transform(X_train)` then `.transform(X_test)` — two separate calls, never `fit_transform` on test data.
- `GridSearchCV` gets slow fast (params × cv folds × model training time). If your grid is large, swap in `RandomizedSearchCV` with `n_iter=20` or so.
- `grid_search.best_estimator_` is already a fitted pipeline — you can call `.predict()` on it directly, no need to refit.

---

## 6. Regression version (e.g. for something like your Laptop Price Predictor)

Same skeleton, just swap the metric and model:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)
print(r2_score(y_test, y_pred))
```
