# NumPy Cheat Sheet

> NumPy is the foundation of Python ML — arrays, math, and speed.

---

## Import
```python
import numpy as np
```

---

## Creating Arrays

```python
np.array([1, 2, 3])               # 1D array
np.array([[1, 2], [3, 4]])        # 2D array
np.zeros((3, 4))                  # all zeros
np.ones((2, 3))                   # all ones
np.full((2, 3), 7)                # fill with 7
np.eye(3)                         # identity matrix
np.random.rand(3, 4)              # uniform [0, 1)
np.random.randn(3, 4)             # standard normal
np.random.randint(0, 10, (3, 4))  # random ints
np.arange(0, 20, 2)               # [0, 2, 4, ..., 18]
np.linspace(0, 1, 5)              # [0.0, 0.25, 0.5, 0.75, 1.0]
```

---

## Array Properties

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.shape      # (2, 3)
a.ndim       # 2
a.size       # 6
a.dtype      # dtype('int64')
a.T          # transpose
```

---

## Reshaping

```python
a.reshape(3, 2)         # reshape to (3, 2)
a.flatten()             # 1D copy
a.ravel()               # 1D view (faster)
np.expand_dims(a, 0)    # add dimension at axis 0
a[np.newaxis, :]        # same as above
a.squeeze()             # remove size-1 dimensions
```

---

## Indexing & Slicing

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

a[0]          # first row: [1, 2, 3]
a[:, 1]       # second column: [2, 5, 8]
a[1, 2]       # element at row 1, col 2: 6
a[0:2, 1:3]   # submatrix [[2,3],[5,6]]

# Boolean indexing
a[a > 5]      # [6, 7, 8, 9]
a[a % 2 == 0] # even numbers

# Fancy indexing
a[[0, 2], :]  # rows 0 and 2
```

---

## Math Operations

```python
a + b          # element-wise add
a - b          # element-wise subtract
a * b          # element-wise multiply
a / b          # element-wise divide
a ** 2         # element-wise square
np.sqrt(a)     # square root
np.exp(a)      # e^a
np.log(a)      # natural log
np.abs(a)      # absolute value
np.floor(a)    # floor
np.ceil(a)     # ceiling
np.round(a, 2) # round to 2 decimal places
```

---

## Aggregations

```python
np.sum(a)              # total sum
np.sum(a, axis=0)      # sum along columns
np.sum(a, axis=1)      # sum along rows
np.mean(a)
np.median(a)
np.std(a)
np.var(a)
np.min(a), np.max(a)
np.argmin(a)           # index of minimum
np.argmax(a)           # index of maximum
np.cumsum(a)           # cumulative sum
np.prod(a)             # product of all elements
np.percentile(a, 75)   # 75th percentile
```

---

## Linear Algebra

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

np.dot(A, B)        # matrix multiplication
A @ B               # same, cleaner syntax
np.linalg.inv(A)    # inverse
np.linalg.det(A)    # determinant
np.linalg.eig(A)    # eigenvalues and eigenvectors
np.linalg.norm(A)   # Frobenius norm
np.linalg.solve(A, b)  # solve Ax = b
```

---

## Stacking & Splitting

```python
np.vstack([a, b])          # stack vertically (row-wise)
np.hstack([a, b])          # stack horizontally (col-wise)
np.concatenate([a, b], axis=0)
np.split(a, 3)             # split into 3 equal parts
np.vsplit(a, 2)            # vertical split
np.hsplit(a, 2)            # horizontal split
```

---

## Sorting

```python
np.sort(a)                  # sorted copy
np.sort(a, axis=0)          # sort along axis
np.argsort(a)               # indices that would sort
np.lexsort((b, a))          # sort by multiple keys
```

---

## Useful Utilities

```python
np.unique(a)                    # unique values
np.unique(a, return_counts=True)# unique + counts
np.where(a > 3, 1, 0)           # conditional replacement
np.clip(a, 0, 5)                # clip values to [0, 5]
np.nan                          # NaN constant
np.isnan(a)                     # boolean mask of NaNs
np.nanmean(a)                   # mean ignoring NaN
np.set_printoptions(precision=3, suppress=True)  # print formatting
```

---

## Broadcasting Rules

NumPy aligns shapes from the right. Dimensions of size 1 are stretched.

```python
# (3, 1) + (1, 4) → (3, 4)
a = np.array([[1], [2], [3]])   # shape (3, 1)
b = np.array([10, 20, 30, 40])  # shape (4,) → (1, 4)
a + b
# [[11, 21, 31, 41],
#  [12, 22, 32, 42],
#  [13, 23, 33, 43]]
```

---

## Random Seed (for reproducibility)

```python
np.random.seed(42)
rng = np.random.default_rng(42)  # preferred modern way
rng.integers(0, 10, size=(3, 3))
rng.normal(0, 1, size=(100,))
```
