# Data Preprocessing Cheat Sheet

> Clean data is the most impactful part of any ML project.

---

## Imports
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
```

---

## The Preprocessing Checklist

1. Load and inspect data
2. Handle duplicates
3. Fix data types
4. Handle missing values
5. Handle outliers
6. Scale numerical features
7. Encode categorical features
8. Feature engineering (see next sheet)
9. Split into train/test

---

## Load & Quick Audit

```python
df = pd.read_csv("data.csv")

# Full audit
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe())
for col in df.select_dtypes("object").columns:
    print(f"{col}: {df[col].nunique()} unique — {df[col].value_counts().head(3).to_dict()}")
```

---

## Handling Duplicates

```python
df.duplicated().sum()                          # count
df[df.duplicated(keep=False)]                  # show all duplicates
df.drop_duplicates(inplace=True)               # drop duplicates
df.drop_duplicates(subset=["id"], keep="last") # based on specific columns
```

---

## Fixing Data Types

```python
# Cast types
df["age"] = df["age"].astype(int)
df["price"] = pd.to_numeric(df["price"], errors="coerce")   # bad → NaN
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df["category"] = df["category"].astype("category")

# Fix mixed types (e.g., "25K", "30K")
df["salary"] = df["salary"].str.replace("K", "").astype(float) * 1000
```

---

## Missing Value Strategies

```python
# Visualize missingness
import missingno as msno
msno.matrix(df)              # matrix view
msno.bar(df)                 # bar chart

# Drop
df.dropna(thresh=int(0.5 * len(df.columns)), axis=0)  # drop rows >50% missing
df.dropna(subset=["target"])                            # must have target

# Impute — numerical
imp_mean = SimpleImputer(strategy="mean")         # use for symmetric distributions
imp_median = SimpleImputer(strategy="median")     # use for skewed distributions / outliers
df[num_cols] = imp_median.fit_transform(df[num_cols])

# Impute — categorical
imp_mode = SimpleImputer(strategy="most_frequent")
df[cat_cols] = imp_mode.fit_transform(df[cat_cols])

# KNN Imputer (uses similar rows)
knn_imp = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(knn_imp.fit_transform(df), columns=df.columns)

# Add missingness indicator (sometimes informative)
df["age_was_missing"] = df["age"].isnull().astype(int)
df["age"].fillna(df["age"].median(), inplace=True)
```

---

## Outlier Detection & Treatment

```python
# IQR method
def iqr_bounds(series):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

lower, upper = iqr_bounds(df["salary"])
outliers = df[(df["salary"] < lower) | (df["salary"] > upper)]
print(f"{len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# Clip outliers
df["salary"] = df["salary"].clip(lower, upper)

# Z-score method
from scipy import stats
z = np.abs(stats.zscore(df[num_cols]))
df_clean = df[(z < 3).all(axis=1)]    # keep rows where all z-scores < 3

# Log transform (reduces right skew)
df["salary_log"] = np.log1p(df["salary"])    # log1p handles zeros

# Winsorization
from scipy.stats import mstats
df["salary"] = mstats.winsorize(df["salary"], limits=[0.05, 0.05])
```

---

## Scaling Numerical Features

| Scaler | When to use |
|--------|------------|
| `StandardScaler` | Most models (logistic regression, SVM, KNN, neural nets) |
| `MinMaxScaler` | When you need [0, 1] range (neural nets, image data) |
| `RobustScaler` | Data with significant outliers |
| None | Tree-based models (random forest, XGBoost) — don't need scaling |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])   # fit only on train!
X_test[num_cols] = scaler.transform(X_test[num_cols])

# PowerTransformer — makes data more Gaussian (good for linear models)
pt = PowerTransformer(method="yeo-johnson")   # handles negative values too
X_train[num_cols] = pt.fit_transform(X_train[num_cols])
```

---

## Encoding Categorical Variables

```python
# Label Encoding — ordinal categories or target
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
le = LabelEncoder()
df["target"] = le.fit_transform(df["target"])
le.classes_           # see the mapping

# Ordinal Encoding — ordered categories
oe = OrdinalEncoder(categories=[["low", "medium", "high"]])
df[["risk_level"]] = oe.fit_transform(df[["risk_level"]])

# One-Hot Encoding — nominal categories (no order)
df = pd.get_dummies(df, columns=["color", "city"], drop_first=True)

# sklearn OHE (use in pipelines)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")

# Target Encoding — for high-cardinality categories
from category_encoders import TargetEncoder
te = TargetEncoder()
df["city_encoded"] = te.fit_transform(df["city"], df["target"])

# Frequency Encoding — simple and effective
freq = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq)
```

---

## Handling Class Imbalance

```python
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

print(df["target"].value_counts(normalize=True))  # check imbalance

# SMOTE — synthetic oversampling of minority class
sm = SMOTE(sampling_strategy="auto", random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Undersampling
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

# Class weights in model (often better than resampling)
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), weights))

model = LogisticRegression(class_weight=class_weight_dict)
# or: RandomForestClassifier(class_weight="balanced")
```

---

## Full Preprocessing Pipeline Template

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes("object").columns.tolist()

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

full_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

full_pipeline.fit(X_train, y_train)
print(f"Test Accuracy: {full_pipeline.score(X_test, y_test):.4f}")
```
