# Matplotlib & Seaborn Cheat Sheet

> Every plot you'll ever need for EDA and presentations.

---

## Imports
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style globally
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100
```

---

## Figure Anatomy

```python
fig, ax = plt.subplots(figsize=(10, 6))   # figure + axes
ax.plot(x, y)
ax.set_title("My Plot", fontsize=16, fontweight="bold")
ax.set_xlabel("X Axis", fontsize=12)
ax.set_ylabel("Y Axis", fontsize=12)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("plot.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Line Plots

```python
# Matplotlib
fig, ax = plt.subplots()
ax.plot(x, y, color="steelblue", linewidth=2, linestyle="--", marker="o", markersize=4, label="Train")
ax.plot(x, y2, color="coral", label="Test")
ax.legend()

# Seaborn
sns.lineplot(data=df, x="epoch", y="loss", hue="split", style="split")
```

---

## Scatter Plots

```python
# Matplotlib
ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap="viridis", edgecolors="none")
plt.colorbar()

# Seaborn
sns.scatterplot(data=df, x="feature1", y="target", hue="category", size="weight",
                palette="deep", sizes=(20, 200), alpha=0.7)

# With regression line
sns.regplot(data=df, x="x", y="y", scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
```

---

## Bar Charts

```python
# Matplotlib
ax.bar(categories, values, color="steelblue", edgecolor="white", width=0.6)
ax.barh(categories, values)  # horizontal

# Seaborn — single grouping
sns.barplot(data=df, x="category", y="value", estimator="mean", errorbar="sd", palette="Set2")

# Grouped bar chart
sns.barplot(data=df, x="month", y="sales", hue="product", palette="Set1")

# Count chart
sns.countplot(data=df, x="category", order=df["category"].value_counts().index)
```

---

## Histograms & Distributions

```python
# Matplotlib
ax.hist(data, bins=30, edgecolor="white", color="steelblue", density=True, alpha=0.7)

# Seaborn — histogram + KDE
sns.histplot(data=df, x="age", bins=30, kde=True, hue="gender", stat="density")

# KDE only
sns.kdeplot(data=df, x="salary", hue="department", fill=True, alpha=0.3)

# Multiple distributions
sns.displot(df, x="value", col="group", kind="hist", bins=20, aspect=0.8)
```

---

## Box & Violin Plots

```python
# Seaborn box plot
sns.boxplot(data=df, x="group", y="value", palette="Set2",
            flierprops={"marker": "o", "markersize": 3})

# With individual points
sns.boxplot(data=df, x="group", y="value")
sns.stripplot(data=df, x="group", y="value", color="black", alpha=0.3, jitter=True)

# Violin plot (shows full distribution)
sns.violinplot(data=df, x="group", y="value", inner="quart", palette="muted")

# Swarm plot (no overlap)
sns.swarmplot(data=df, x="group", y="value", size=3)
```

---

## Heatmaps

```python
# Correlation matrix
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle
sns.heatmap(corr, annot=True, fmt=".2f", mask=mask,
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5)
plt.title("Feature Correlations")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual")
```

---

## Pair Plots

```python
# All-vs-all scatter matrix
sns.pairplot(df, hue="target", diag_kind="kde",
             plot_kws={"alpha": 0.5}, diag_kws={"fill": True})

# Custom
g = sns.PairGrid(df, hue="species")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
```

---

## Feature Importance Plot

```python
importances = pd.Series(model.feature_importances_, index=feature_names)
importances.nlargest(15).sort_values().plot(kind="barh", figsize=(8, 8))
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
```

---

## Subplots

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(df.select_dtypes("number").columns[:6]):
    sns.histplot(df[col], ax=axes[i], kde=True, color="steelblue")
    axes[i].set_title(col)

plt.suptitle("Numerical Features Distribution", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
```

---

## Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

fig, ax = plt.subplots()
ax.plot(train_sizes, train_scores.mean(axis=1), label="Train")
ax.fill_between(train_sizes,
                train_scores.mean(1) - train_scores.std(1),
                train_scores.mean(1) + train_scores.std(1), alpha=0.2)
ax.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
ax.fill_between(train_sizes,
                val_scores.mean(1) - val_scores.std(1),
                val_scores.mean(1) + val_scores.std(1), alpha=0.2)
ax.set_xlabel("Training Size"); ax.set_ylabel("Score"); ax.legend()
```

---

## Plot Styles Reference

```python
# Seaborn themes
sns.set_theme(style="darkgrid")    # grid on dark
sns.set_theme(style="whitegrid")   # grid on white — good for presentations
sns.set_theme(style="white")       # no grid
sns.set_theme(style="ticks")       # ticks only

# Seaborn palettes
"deep", "muted", "pastel", "bright", "dark", "colorblind"
"viridis", "magma", "coolwarm", "RdYlGn"

# Matplotlib styles
plt.style.use("seaborn-v0_8-whitegrid")
plt.style.use("ggplot")
plt.style.use("fivethirtyeight")
```

---

## Quick EDA Template

```python
def quick_eda_plots(df):
    num_cols = df.select_dtypes("number").columns
    n = len(num_cols)

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], ax=axes[i, 0], kde=True)
        axes[i, 0].set_title(f"{col} — distribution")
        sns.boxplot(y=df[col], ax=axes[i, 1])
        axes[i, 1].set_title(f"{col} — boxplot")
    plt.tight_layout()
    plt.show()
```
