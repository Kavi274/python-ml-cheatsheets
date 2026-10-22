# Scikit-learn Cheat Sheet

> Every major model, pipeline, and evaluation tool in one place.

---

## Import
```python
from sklearn import ...   # see sections below
```

---

## The Universal API

Every sklearn model follows the same pattern:

```python
model.fit(X_train, y_train)        # learn from data
model.predict(X_test)              # make predictions
model.predict_proba(X_test)        # class probabilities (classifiers)
model.score(X_test, y_test)        # default metric (accuracy or R²)
model.get_params()                 # view hyperparameters
```

---

## Train / Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,
    stratify=y            # preserve class distribution
)
```

---

## Classification Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Logistic Regression
LogisticRegression(C=1.0, max_iter=1000, multi_class="auto", solver="lbfgs")

# Decision Tree
DecisionTreeClassifier(max_depth=5, min_samples_split=10, criterion="gini")

# Random Forest
RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)

# Gradient Boosting
GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# SVM
SVC(C=1.0, kernel="rbf", probability=True)   # kernel: "linear", "rbf", "poly"

# KNN
KNeighborsClassifier(n_neighbors=5, metric="euclidean")
```

---

## Regression Models

```python
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    BayesianRidge, HuberRegressor
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Linear Regression
LinearRegression(fit_intercept=True)

# Ridge (L2 regularization)
Ridge(alpha=1.0)

# Lasso (L1 regularization — feature selection)
Lasso(alpha=0.1, max_iter=10000)

# ElasticNet (L1 + L2)
ElasticNet(alpha=1.0, l1_ratio=0.5)

# Random Forest Regressor
RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Gradient Boosting Regressor
GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
```

---

## Clustering

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X)
kmeans.labels_           # cluster label per point
kmeans.cluster_centers_  # centroids
kmeans.inertia_          # within-cluster sum of squares

# DBSCAN (density-based, no need to specify k)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)
dbscan.labels_           # -1 = noise

# Agglomerative
AgglomerativeClustering(n_clusters=5, linkage="ward")

# Elbow method for KMeans
inertias = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_ for k in range(1, 11)]
```

---

## Preprocessing

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer, Normalizer
)

# Scaling
StandardScaler()          # mean=0, std=1 — use for most models
MinMaxScaler()            # scales to [0, 1]
RobustScaler()            # uses median/IQR — good with outliers

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled = scaler.transform(X_test)         # only transform on test!

# Encoding
LabelEncoder()                                    # ordinal: 0, 1, 2 — for target
OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # creates dummies
OrdinalEncoder(categories=[["low","med","high"]]) # ordered categories

# Polynomial features
PolynomialFeatures(degree=2, include_bias=False)  # adds x², x₁x₂, etc.
```

---

## Pipelines

Pipelines prevent data leakage and make code clean.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Simple pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)

# Mixed numeric + categorical pipeline
num_cols = ["age", "income"]
cat_cols = ["city", "gender"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="drop")

full_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingClassifier())
])
full_pipe.fit(X_train, y_train)
```

---

## Cross-Validation

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate, KFold, 
    StratifiedKFold, GridSearchCV, RandomizedSearchCV
)

# Basic CV
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"{scores.mean():.3f} ± {scores.std():.3f}")

# Multiple metrics
results = cross_validate(model, X, y, cv=5,
    scoring=["accuracy", "f1", "roc_auc"],
    return_train_score=True)

# Stratified K-Fold (keeps class ratio in each fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_score(model, X, y, cv=skf)
```

---

## Hyperparameter Tuning

```python
# Grid Search (exhaustive)
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [3, 5, None],
    "model__min_samples_split": [2, 5, 10]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)

# Randomized Search (faster for large spaces)
from scipy.stats import randint, uniform

param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(2, 15),
    "learning_rate": uniform(0.01, 0.3)
}
rand = RandomizedSearchCV(model, param_dist, n_iter=50, cv=5, scoring="roc_auc", random_state=42)
rand.fit(X_train, y_train)
```

---

## Feature Importance & Selection

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, RFECV, SelectFromModel
)

# Tree-based feature importance
model = RandomForestClassifier().fit(X_train, y_train)
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

# Select top K features
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X_train, y_train)

# Recursive Feature Elimination
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
rfe.fit(X_train, y_train)
rfe.support_    # boolean mask of selected features
rfe.ranking_    # feature ranking (1 = selected)

# Model-based selection (Lasso)
sel = SelectFromModel(Lasso(alpha=0.01))
sel.fit(X_train, y_train)
X_selected = sel.transform(X_train)
```

---

## Saving & Loading Models

```python
import joblib

joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")

# Also works for full pipelines
joblib.dump(pipe, "pipeline.pkl")
```

---

## Common Scoring Strings

| Task | Scoring string |
|------|---------------|
| Binary classification | `"accuracy"`, `"f1"`, `"roc_auc"`, `"precision"`, `"recall"` |
| Multiclass | `"f1_macro"`, `"f1_weighted"`, `"accuracy"` |
| Regression | `"r2"`, `"neg_mean_squared_error"`, `"neg_mean_absolute_error"` |
| Clustering | `"adjusted_rand_score"`, `"silhouette_score"` |
