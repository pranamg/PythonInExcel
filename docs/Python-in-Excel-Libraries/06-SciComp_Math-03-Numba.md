# Leveraging Numba in Python in Excel

Numba is a just-in-time (JIT) compiler that accelerates numerical Python code by translating it to fast machine code at runtime. With Python in Excel, you can use Numba to speed up custom functions and loops, making heavy computations much faster directly within your spreadsheets.

## 1. Setup and Imports

To use Numba, reserve the first worksheet for import statements:

```python
=PY(
import numba
from numba import jit, njit
)
```

This makes the Numba decorators available for all subsequent Python cells.

## 2. Accelerating Functions

- **JIT-compile a function:**

```python
=PY(
@jit(nopython=True)
def fast_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

result = fast_sum(xl("A1:A100"))
result
)
```

- **Use `njit` for even simpler syntax:**

```python
=PY(
@njit
def fast_square(arr):
    return [x*x for x in arr]

fast_square(xl("A1:A10"))
)
```

## 3. Vectorized Loops and Array Operations

Numba excels at accelerating explicit loops and custom math:

```python
=PY(
@njit
def moving_average(arr, window):
    result = []
    for i in range(len(arr) - window + 1):
        result.append(sum(arr[i:i+window]) / window)
    return result

moving_average(xl("A1:A100"), 5)
)
```

## 4. Best Practices

- **Imports Once**: Place all import statements on the first worksheet.
- **Use `nopython=True`**: For maximum speed, use `@jit(nopython=True)` or `@njit`.
- **Data Types**: Numba works best with NumPy arrays and native Python types.
- **Debugging**: Test your function without Numba first, then add the decorator for speed.

By leveraging Numba in Python in Excel, you can dramatically accelerate custom computations, making even large-scale analytics fast and efficient within your spreadsheets.
