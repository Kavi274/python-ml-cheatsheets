# Statistics for ML Cheat Sheet

> The math behind the models — probabilities, distributions, and hypothesis tests.

---

## Imports
```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Descriptive Statistics

```python
data = df["value"]

# Central tendency
data.mean()
data.median()
data.mode()[0]

# Spread
data.std()
data.var()
data.sem()                        # standard error of mean
stats.iqr(data)                   # interquartile range

# Shape
stats.skew(data)                  # skewness (0 = symmetric)
stats.kurtosis(data)              # kurtosis (0 = normal)

# Full summary
data.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
```

---

## Probability Distributions

### Normal (Gaussian)
```python
from scipy.stats import norm

norm.pdf(x, loc=0, scale=1)      # probability density
norm.cdf(x, loc=0, scale=1)      # cumulative probability P(X ≤ x)
norm.ppf(0.95, loc=0, scale=1)   # inverse CDF (95th percentile)
norm.rvs(loc=0, scale=1, size=1000)  # random samples

# 68-95-99.7 rule
print(norm.cdf(1) - norm.cdf(-1))   # ~0.6827 (1σ)
print(norm.cdf(2) - norm.cdf(-2))   # ~0.9545 (2σ)
print(norm.cdf(3) - norm.cdf(-3))   # ~0.9973 (3σ)
```

### Common Distributions
```python
from scipy.stats import (
    norm,         # continuous, symmetric
    t,            # like normal but heavier tails (small samples)
    chi2,         # goodness of fit, variance tests
    f,            # ANOVA, regression F-tests
    binom,        # discrete, n trials, p success
    poisson,      # discrete, rare events, count data
    uniform,      # continuous, all equally likely
    expon,        # time between events
    lognorm,      # log-normal (positive, right-skewed data)
    beta,         # probabilities, proportions [0,1]
    gamma,        # waiting times, insurance claims
)

# Quick reference
binom.pmf(k=3, n=10, p=0.5)     # P(X = 3) for Binomial(10, 0.5)
poisson.pmf(k=2, mu=1.5)        # P(X = 2) for Poisson(λ=1.5)
```

---

## Testing Normality

```python
# Shapiro-Wilk (best for n < 5000)
stat, p = stats.shapiro(data)
print(f"Shapiro-Wilk p={p:.4f} — {'Normal' if p > 0.05 else 'Not normal'}")

# D'Agostino's K²
stat, p = stats.normaltest(data)

# Kolmogorov-Smirnov
stat, p = stats.kstest(data, "norm", args=(data.mean(), data.std()))

# Q-Q plot (visual)
fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(data, dist="norm", plot=ax)
ax.set_title("Q-Q Plot"); plt.show()
```

---

## Hypothesis Testing Framework

```
1. State H₀ (null) and H₁ (alternative)
2. Choose significance level α (usually 0.05)
3. Compute test statistic
4. Get p-value
5. Decision: if p < α → reject H₀
```

**p-value**: probability of seeing this data (or more extreme) if H₀ were true.  
A small p-value means the data is unlikely under H₀.

---

## Comparing Means (t-tests)

```python
# One-sample t-test: is the mean equal to a hypothesized value?
stat, p = stats.ttest_1samp(data, popmean=0)

# Two-sample independent t-test: are two group means equal?
stat, p = stats.ttest_ind(group1, group2, equal_var=False)  # Welch's (safer)

# Paired t-test: before vs after, same subjects
stat, p = stats.ttest_rel(before, after)

# Practical: effect size (Cohen's d)
def cohens_d(a, b):
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled_std

d = cohens_d(group1, group2)
# |d| < 0.2: trivial, 0.2–0.5: small, 0.5–0.8: medium, > 0.8: large
```

---

## Non-parametric Tests (when data isn't normal)

```python
# Mann-Whitney U (non-parametric two-sample test)
stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")

# Wilcoxon signed-rank (paired, non-parametric)
stat, p = stats.wilcoxon(before, after)

# Kruskal-Wallis (non-parametric ANOVA for 3+ groups)
stat, p = stats.kruskal(group1, group2, group3)

# Chi-squared test (for categorical variables)
contingency_table = pd.crosstab(df["gender"], df["purchase"])
stat, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"χ² = {stat:.3f}, p = {p:.4f}, dof = {dof}")
```

---

## Correlation

```python
# Pearson (linear, continuous, assumes normality)
r, p = stats.pearsonr(x, y)

# Spearman (monotonic, non-parametric)
rho, p = stats.spearmanr(x, y)

# Kendall's tau (non-parametric, handles ties)
tau, p = stats.kendalltau(x, y)

# Correlation matrix
corr = df.corr(method="pearson")    # or "spearman"

# Interpretation: |r| < 0.3 weak, 0.3–0.7 moderate, > 0.7 strong
```

---

## Confidence Intervals

```python
import scipy.stats as stats

# Mean CI using t-distribution (small samples)
n = len(data)
mean = data.mean()
se = stats.sem(data)                      # standard error
ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")

# Proportion CI (Clopper-Pearson exact)
from statsmodels.stats.proportion import proportion_confint
ci = proportion_confint(successes, n_trials, alpha=0.05, method="binom_test")

# Bootstrap CI (model-free)
def bootstrap_ci(data, func=np.mean, n_boot=10000, ci=95):
    boot_stats = [func(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    low = (100 - ci) / 2
    return np.percentile(boot_stats, [low, 100 - low])

lower, upper = bootstrap_ci(data)
```

---

## A/B Testing

```python
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower

# Required sample size
effect_size = proportion_effectsize(p1=0.10, p2=0.12)  # 10% → 12% conversion
analysis = NormalIndPower()
n = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative="two-sided")
print(f"Required n per group: {int(np.ceil(n))}")

# Run test after collecting data
control_conversions, control_n = 100, 1000
treatment_conversions, treatment_n = 130, 1000

stat, p = proportions_ztest(
    count=[treatment_conversions, control_conversions],
    nobs=[treatment_n, control_n],
    alternative="two-sided"
)
print(f"p-value: {p:.4f}")
print(f"Result: {'Significant' if p < 0.05 else 'Not significant'}")

# Lift
lift = (treatment_conversions / treatment_n) / (control_conversions / control_n) - 1
print(f"Lift: {lift:.1%}")
```

---

## Bayes' Theorem

```python
# P(A|B) = P(B|A) * P(A) / P(B)
# P(disease | positive test) = ?

p_disease = 0.001               # prior: 0.1% of population
p_positive_given_disease = 0.99  # sensitivity
p_positive_given_healthy = 0.05  # false positive rate

p_positive = (p_positive_given_disease * p_disease +
              p_positive_given_healthy * (1 - p_disease))

p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
print(f"P(disease | positive test) = {p_disease_given_positive:.2%}")
# ~1.94% — even a 99% accurate test has a low PPV for rare diseases
```

---

## Key Formulas

| Formula | Meaning |
|---------|---------|
| `P(A∩B) = P(A)·P(B|A)` | Multiplication rule |
| `P(A∪B) = P(A) + P(B) - P(A∩B)` | Addition rule |
| `Var(X) = E[X²] - E[X]²` | Variance |
| `Cov(X,Y) = E[XY] - E[X]E[Y]` | Covariance |
| `r = Cov(X,Y) / (σ_X · σ_Y)` | Pearson correlation |
| `SE = σ / √n` | Standard error of mean |
| `z = (x̄ - μ) / SE` | Z-score / test statistic |

---

## When to Use Which Test

| Situation | Test |
|-----------|------|
| Is mean = µ? (normal data) | One-sample t-test |
| Are 2 group means equal? (normal) | Independent t-test |
| Before/after same group? (normal) | Paired t-test |
| 3+ group means? (normal) | One-way ANOVA |
| 2 groups, not normal | Mann-Whitney U |
| 3+ groups, not normal | Kruskal-Wallis |
| Are 2 categorical variables independent? | Chi-squared |
| Is distribution X? | KS test / Shapiro-Wilk |
| Correlation? (non-normal) | Spearman / Kendall |
