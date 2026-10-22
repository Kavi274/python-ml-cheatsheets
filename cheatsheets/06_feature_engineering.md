# Feature Engineering Cheat Sheet

> Good features beat complex models. This is where ML gains are really made.

---

## Numerical Features

```python
# Binning / discretization
df["age_group"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 100],
                          labels=["teen", "young", "adult", "senior"])
df["income_quantile"] = pd.qcut(df["income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

# Ratios and interactions
df["price_per_sqft"] = df["price"] / df["area"]
df["income_to_age"] = df["income"] / df["age"]
df["feature_product"] = df["a"] * df["b"]

# Log / power transforms (fix skewness)
df["log_salary"] = np.log1p(df["salary"])       # log1p handles 0
df["sqrt_visits"] = np.sqrt(df["visits"])
df["salary_sq"] = df["salary"] ** 2             # capture non-linearity

# Standardized deviation from group mean
df["salary_vs_dept"] = df["salary"] - df.groupby("dept")["salary"].transform("mean")

# Rank (useful for tree models + outlier robustness)
df["salary_rank"] = df["salary"].rank(pct=True)
```

---

## Categorical Features

```python
# Frequency / count encoding
freq = df["city"].value_counts()
df["city_freq"] = df["city"].map(freq)
df["city_pct"] = df["city"].map(freq / len(df))

# Target mean encoding (avoid leakage — use on train only!)
target_mean = df.groupby("city")["target"].mean()
df["city_target_enc"] = df["city"].map(target_mean)

# Cross features (interactions)
df["city_gender"] = df["city"] + "_" + df["gender"]
df["age_city"] = df["age_group"].astype(str) + "_" + df["city"]

# Groupby aggregations (powerful!)
aggs = df.groupby("user_id").agg(
    total_purchases=("amount", "sum"),
    avg_purchase=("amount", "mean"),
    num_transactions=("amount", "count"),
    max_purchase=("amount", "max"),
    days_active=("date", "nunique")
).reset_index()
df = df.merge(aggs, on="user_id", how="left")
```

---

## DateTime Features

```python
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek      # 0=Monday
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["quarter"] = df["date"].dt.quarter
df["hour"] = df["date"].dt.hour
df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

# Cyclical encoding (month 1 and 12 are close!)
import math
df["month_sin"] = np.sin(2 * math.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * math.pi * df["month"] / 12)
df["hour_sin"] = np.sin(2 * math.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * math.pi * df["hour"] / 24)

# Time since a reference point
ref_date = pd.Timestamp("2020-01-01")
df["days_since_ref"] = (df["date"] - ref_date).dt.days

# Lag features (for time series)
df = df.sort_values("date")
df["lag_1"] = df["sales"].shift(1)
df["lag_7"] = df["sales"].shift(7)
df["rolling_mean_7"] = df["sales"].rolling(7, min_periods=1).mean()
df["rolling_std_7"] = df["sales"].rolling(7, min_periods=1).std()
```

---

## Text Features

```python
# Basic
df["text_len"] = df["review"].str.len()
df["word_count"] = df["review"].str.split().str.len()
df["avg_word_len"] = df["review"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df["exclamation_count"] = df["review"].str.count("!")
df["uppercase_ratio"] = df["review"].apply(lambda x: sum(c.isupper() for c in str(x)) / max(len(str(x)), 1))

# TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2),
                         min_df=5, max_df=0.8, stop_words="english")
X_tfidf = tfidf.fit_transform(df["review"])

# Word embeddings (average)
# Use sentence-transformers for production-quality embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), batch_size=32, show_progress_bar=True)
```

---

## Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X[["a", "b", "c"]])
# Creates: a, b, c, a², b², c², ab, ac, bc

# interaction_only=True → only cross terms (no squares)
```

---

## Dimensionality Reduction

```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import umap  # pip install umap-learn

# PCA
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Find optimal number of components
pca_full = PCA().fit(X_scaled)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1
print(f"Components for 95% variance: {n_components}")

# t-SNE (visualization only, not for features)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_pca)   # apply PCA first for speed

# UMAP (faster and better than t-SNE for large data)
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
```

---

## Feature Selection

```python
# Variance threshold — remove near-zero variance features
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
X_filtered = sel.fit_transform(X)

# Correlation-based
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
df.drop(columns=to_drop, inplace=True)

# Mutual information
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
mi = mutual_info_classif(X_train, y_train, random_state=42)
mi_scores = pd.Series(mi, index=feature_names).sort_values(ascending=False)

# SHAP for feature importance
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

---

## Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["month"] = pd.to_datetime(X["date"]).dt.month
        X["dayofweek"] = pd.to_datetime(X["date"]).dt.dayofweek
        X["is_weekend"] = (X["dayofweek"] >= 5).astype(int)
        return X.drop(columns=["date"])

pipe = Pipeline([
    ("date_features", DateFeatures()),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])
```
