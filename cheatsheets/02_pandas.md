# Pandas Cheat Sheet

> Pandas is your workhorse for loading, cleaning, and exploring data.

---

## Import
```python
import pandas as pd
import numpy as np
```

---

## Creating DataFrames

```python
# From dict
df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30], "score": [88.5, 92.1]})

# From CSV / Excel / JSON
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", index_col=0, parse_dates=["date"])
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_json("data.json")

# From NumPy array
df = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])

# Series
s = pd.Series([1, 2, 3], index=["a", "b", "c"], name="values")
```

---

## Inspecting Data

```python
df.shape           # (rows, cols)
df.dtypes          # column types
df.info()          # dtype + non-null counts
df.describe()      # count, mean, std, min, max, quartiles
df.head(10)        # first 10 rows
df.tail(5)
df.sample(5)       # random 5 rows
df.columns.tolist()
df.index
df.nunique()       # unique count per column
df["col"].value_counts()
```

---

## Selecting Data

```python
df["col"]               # single column → Series
df[["col1", "col2"]]    # multiple columns → DataFrame

# loc: label-based
df.loc[0]               # row by index label
df.loc[0:3, "col1":"col3"]
df.loc[df["age"] > 25]

# iloc: position-based
df.iloc[0]              # first row
df.iloc[:, 2]           # third column
df.iloc[0:3, 1:4]

# Conditions
df[df["score"] > 90]
df[(df["age"] > 25) & (df["score"] > 80)]
df[df["name"].isin(["Alice", "Charlie"])]
df[df["name"].str.contains("Al")]
```

---

## Adding & Removing Columns

```python
df["new_col"] = df["score"] * 2                       # new column
df["grade"] = df["score"].apply(lambda x: "A" if x >= 90 else "B")
df.insert(1, "rank", [1, 2, 3])                       # insert at position

df.drop("col", axis=1, inplace=True)                  # drop column
df.drop(["col1", "col2"], axis=1)
df.drop(0, axis=0)                                    # drop row by index
df.rename(columns={"old": "new"}, inplace=True)
```

---

## Handling Missing Values

```python
df.isnull().sum()               # count NaN per column
df.isnull().sum() / len(df)     # % missing

df.dropna()                     # drop rows with any NaN
df.dropna(subset=["col"])       # drop if NaN in specific column
df.dropna(thresh=3)             # keep rows with at least 3 non-NaN

df.fillna(0)                    # fill with constant
df.fillna(df.mean())            # fill with column mean
df["col"].fillna(df["col"].median(), inplace=True)
df.fillna(method="ffill")       # forward fill
df.fillna(method="bfill")       # backward fill
df.interpolate()                # linear interpolation
```

---

## Data Types

```python
df["col"].astype(int)
df["col"].astype(float)
df["col"].astype(str)
df["col"].astype("category")    # memory-efficient for low cardinality

pd.to_datetime(df["date"])
pd.to_numeric(df["col"], errors="coerce")   # coerce bad values to NaN
```

---

## Sorting

```python
df.sort_values("score", ascending=False)
df.sort_values(["age", "score"], ascending=[True, False])
df.sort_index()
df.nlargest(5, "score")         # top 5 rows by score
df.nsmallest(5, "score")
```

---

## Groupby

```python
df.groupby("city")["sales"].mean()
df.groupby("city").agg({"sales": "sum", "profit": "mean"})
df.groupby(["city", "month"]).agg({"sales": ["sum", "mean", "count"]})

# Custom aggregation
df.groupby("city")["sales"].agg(lambda x: x.max() - x.min())

# Transform (returns same-shape DataFrame)
df["sales_norm"] = df.groupby("city")["sales"].transform(lambda x: (x - x.mean()) / x.std())

# Filter groups
df.groupby("city").filter(lambda x: x["sales"].mean() > 1000)
```

---

## Merging & Joining

```python
# Merge (like SQL JOIN)
pd.merge(df1, df2, on="id")                     # inner join
pd.merge(df1, df2, on="id", how="left")
pd.merge(df1, df2, on="id", how="outer")
pd.merge(df1, df2, left_on="id", right_on="user_id")

# Concat (stack rows or columns)
pd.concat([df1, df2], axis=0, ignore_index=True)  # stack rows
pd.concat([df1, df2], axis=1)                     # stack columns

# Join (index-based)
df1.join(df2, how="left")
```

---

## Pivot Tables

```python
pd.pivot_table(df, values="sales", index="city", columns="month", aggfunc="sum", fill_value=0)

# Melt (wide → long)
pd.melt(df, id_vars=["id", "name"], value_vars=["jan", "feb", "mar"],
        var_name="month", value_name="sales")

# Pivot (long → wide)
df.pivot(index="id", columns="month", values="sales")
```

---

## String Operations

```python
df["name"].str.lower()
df["name"].str.upper()
df["name"].str.strip()
df["name"].str.replace("old", "new")
df["name"].str.split(",", expand=True)   # split into columns
df["name"].str.contains("pattern")
df["name"].str.startswith("A")
df["name"].str.len()
df["name"].str.extract(r"(\d+)")         # regex extract
```

---

## Date & Time

```python
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek   # 0=Monday
df["quarter"] = df["date"].dt.quarter
df["date"].dt.strftime("%Y-%m")

# Filter by date
df[df["date"] > "2023-01-01"]
df.set_index("date").loc["2023-01":"2023-06"]  # date range
```

---

## Apply & Map

```python
df["col"].apply(lambda x: x * 2)
df.apply(lambda row: row["a"] + row["b"], axis=1)  # row-wise
df.applymap(lambda x: round(x, 2))                # element-wise (deprecated → use map)

df["col"].map({"yes": 1, "no": 0})        # replace values via dict
df["col"].replace({"old": "new"})
```

---

## Export

```python
df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx", sheet_name="Sheet1", index=False)
df.to_json("output.json", orient="records")
df.to_parquet("output.parquet")   # fast, compressed
```

---

## Performance Tips

```python
# Use categorical dtype for repeated strings
df["city"] = df["city"].astype("category")

# Vectorized > apply > loop (always prefer vectorized)
df["total"] = df["price"] * df["qty"]       # good
df.apply(lambda r: r["price"] * r["qty"])   # slower

# Check memory
df.memory_usage(deep=True).sum() / 1e6      # MB
df.info(memory_usage="deep")
```
