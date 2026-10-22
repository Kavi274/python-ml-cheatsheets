# Model Evaluation Cheat Sheet

> Picking the right metric matters as much as picking the right model.

---

## Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    matthews_corrcoef, cohen_kappa_score, log_loss
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]   # positive class probability

# Core metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_prob):.4f}")

# Full report
print(classification_report(y_test, y_pred, target_names=class_names))
```

---

## When to Use Which Classification Metric

| Metric | Use when |
|--------|---------|
| Accuracy | Balanced classes |
| Precision | Cost of false positive is high (spam, fraud alert) |
| Recall | Cost of false negative is high (cancer screening, fraud detection) |
| F1 | Imbalanced classes, balance precision & recall |
| ROC-AUC | Comparing models, threshold-independent |
| PR-AUC | Highly imbalanced (rare positives) |
| MCC | Best single metric for imbalanced binary classification |
| Log Loss | When you care about calibrated probabilities |

---

## Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout(); plt.show()

# Extract values
TN, FP, FN, TP = cm.ravel()
print(f"True Positives:  {TP}")
print(f"False Positives: {FP} (Type I error)")
print(f"False Negatives: {FN} (Type II error)")
print(f"True Negatives:  {TN}")
```

---

## ROC & Precision-Recall Curves

```python
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
ax1.plot([0, 1], [0, 1], color="gray", linestyle="--")
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve"); ax1.legend()

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
baseline = y_test.mean()

ax2.plot(recall, precision, color="coral", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
ax2.axhline(baseline, color="gray", linestyle="--", label=f"Baseline = {baseline:.3f}")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve"); ax2.legend()

plt.tight_layout(); plt.show()
```

---

## Optimal Threshold Selection

```python
# F1-based optimal threshold
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, y_prob >= t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_threshold:.2f}, F1: {max(f1_scores):.4f}")

# Youden's J (ROC-based)
fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
print(f"Youden's J threshold: {roc_thresh[best_idx]:.3f}")

# Apply threshold
y_pred_custom = (y_prob >= best_threshold).astype(int)
```

---

## Multiclass Metrics

```python
# averaging options: "macro", "weighted", "micro"
print(f"F1 Macro:    {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 Weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Multiclass ROC-AUC
from sklearn.metrics import roc_auc_score
y_prob_multi = model.predict_proba(X_test)
roc_auc_multi = roc_auc_score(y_test, y_prob_multi, multi_class="ovr", average="weighted")
```

---

## Regression Metrics

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100   # as %
wmape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100

print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R²:    {r2:.4f}")
print(f"MAPE:  {mape:.2f}%")
print(f"WMAPE: {wmape:.2f}%")

# Residual plot
residuals = y_test - y_pred
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Predicted")
axes[1].hist(residuals, bins=50)
axes[1].set_title("Residual Distribution")
plt.tight_layout(); plt.show()
```

---

## Cross-Validation Best Practices

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = cross_validate(
    model, X, y, cv=cv,
    scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"},
    return_train_score=True,
    n_jobs=-1
)

for metric in ["accuracy", "f1", "roc_auc"]:
    train_key = f"train_{metric}"
    test_key = f"test_{metric}"
    print(f"{metric}:")
    print(f"  Train: {results[train_key].mean():.3f} ± {results[train_key].std():.3f}")
    print(f"  Val:   {results[test_key].mean():.3f} ± {results[test_key].std():.3f}")
```

---

## Model Comparison

```python
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    results[name] = scores
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Box plot comparison
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(results.values(), labels=results.keys(), patch_artist=True)
ax.set_ylabel("ROC-AUC"); ax.set_title("Model Comparison (5-Fold CV)")
plt.xticks(rotation=20); plt.tight_layout(); plt.show()
```

---

## SHAP Values (Model Explainability)

```python
import shap

# Tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Beeswarm (better summary)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="beeswarm")

# Single prediction explanation
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=feature_names
))

# Dependence plot (one feature)
shap.dependence_plot("feature_name", shap_values, X_test, feature_names=feature_names)
```

---

## Calibration

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Check calibration
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)

plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
plt.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
plt.title("Calibration Curve"); plt.legend(); plt.show()

# Calibrate the model
calibrated = CalibratedClassifierCV(model, method="platt", cv=5)
calibrated.fit(X_train, y_train)
```
