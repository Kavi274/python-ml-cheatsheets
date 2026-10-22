---
title: Python ML Cheat Sheets
emoji: 🤖
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
---

# Python ML Cheat Sheets

[![Stars](https://img.shields.io/github/stars/Kavi274/python-ml-cheatsheets?style=social)](https://github.com/Kavi274/python-ml-cheatsheets/stargazers)
[![Forks](https://img.shields.io/github/forks/Kavi274/python-ml-cheatsheets?style=social)](https://github.com/Kavi274/python-ml-cheatsheets/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> A complete, beginner-friendly reference for Python Machine Learning — from NumPy basics to deep learning. No fluff, just code.

**If this saves you a Google search, please star it!**

---

## What's Inside

| # | Topic | What you'll find |
|---|-------|-----------------|
| 01 | [NumPy](cheatsheets/01_numpy.md) | Arrays, indexing, math ops, broadcasting |
| 02 | [Pandas](cheatsheets/02_pandas.md) | DataFrames, groupby, merge, time series |
| 03 | [Scikit-learn](cheatsheets/03_scikit_learn.md) | All major models, pipelines, cross-validation |
| 04 | [Matplotlib & Seaborn](cheatsheets/04_matplotlib_seaborn.md) | Every plot type with code |
| 05 | [Data Preprocessing](cheatsheets/05_data_preprocessing.md) | Scaling, encoding, imputation, outliers |
| 06 | [Feature Engineering](cheatsheets/06_feature_engineering.md) | Selection, creation, dimensionality reduction |
| 07 | [Model Evaluation](cheatsheets/07_model_evaluation.md) | All metrics, confusion matrix, ROC, SHAP |
| 08 | [Deep Learning (Keras)](cheatsheets/08_deep_learning_keras.md) | Layers, CNNs, RNNs, callbacks, transfer learning |
| 09 | [Statistics for ML](cheatsheets/09_statistics_for_ml.md) | Distributions, hypothesis tests, Bayesian basics |

---

## Quick Previews

### Load & Inspect Data (30 seconds)
```python
import pandas as pd

df = pd.read_csv("data.csv")
df.shape          # (rows, cols)
df.dtypes         # column types
df.isnull().sum() # missing values per column
df.describe()     # stats summary
df.head()
```

### Train a Model (60 seconds)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
```

### Scale + Encode + Model in a Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipe = Pipeline([("prep", preprocessor), ("model", RandomForestClassifier())])
pipe.fit(X_train, y_train)
```

---

## How to Use

**Browse on GitHub** — every cheat sheet is plain Markdown, no setup needed.

**Clone locally:**
```bash
git clone https://github.com/Kavi274/python-ml-cheatsheets.git
```

**Use as a reference while coding** — Ctrl+F works great on the markdown files.

---

## Who Is This For

- Students starting their ML journey
- Data scientists who want a fast reference
- Anyone tired of re-googling the same syntax

---

## Contributing

Found a mistake? Missing a topic? [Contributions are welcome!](CONTRIBUTING.md)

---

## License

MIT — free to use, share, and adapt.

---

*Made with love for the ML community. If it helped you, a star means a lot.*
