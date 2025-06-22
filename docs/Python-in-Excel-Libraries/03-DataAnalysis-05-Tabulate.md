# Leveraging Tabulate in Python in Excel

Tabulate is a Python library for formatting tabular data as plain-text tables, HTML, LaTeX, and more. With Python in Excel, you can use Tabulate to create readable, well-formatted tables for reports, documentation, or display directly within your spreadsheets.

## 1. Setup and Imports

To use Tabulate, reserve the first worksheet for import statements:

```python
=PY(
from tabulate import tabulate
)
```

This makes the `tabulate` function available for all subsequent Python cells.

## 2. Formatting DataFrames and Lists

- **Format a pandas DataFrame:**

```python
=PY(
import pandas as pd
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
tab = tabulate(df, headers='keys', tablefmt='github')
tab
)
```

- **Format a list of lists:**

```python
=PY(
data = [["Alice", 24], ["Bob", 19]]
tab = tabulate(data, headers=["Name", "Age"], tablefmt='grid')
tab
)
```

## 3. Output Formats

- **Plain text:** `tablefmt='plain'`
- **Grid:** `tablefmt='grid'`
- **GitHub Markdown:** `tablefmt='github'`
- **HTML:** `tablefmt='html'`
- **LaTeX:** `tablefmt='latex'`

## 4. Best Practices

- **Imports Once**: Place all import statements on the first worksheet.
- **Data Preparation**: Clean and structure data before formatting.
- **Output Management**: Return formatted strings for display in Excel or export.
- **Performance**: For large tables, sample or summarize data to maintain readability.

By leveraging Tabulate in Python in Excel, you can create attractive, readable tables for reports and documentation, enhancing the presentation of your data within spreadsheets.
