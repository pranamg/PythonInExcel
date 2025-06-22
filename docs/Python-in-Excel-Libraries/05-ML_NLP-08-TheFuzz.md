# Leveraging TheFuzz in Python in Excel

TheFuzz (formerly known as FuzzyWuzzy) is a Python library for fuzzy string matching, enabling you to compare, score, and match text data even when there are typos or variations. With Python in Excel, you can use TheFuzz to perform approximate string matching and deduplication directly within your spreadsheets.

## 1. Setup and Imports

To use TheFuzz, reserve the first worksheet for import statements:

```python
=PY(
from thefuzz import fuzz, process
)
```

This makes the main TheFuzz functions available for all subsequent Python cells.

## 2. Fuzzy String Matching

- **Simple ratio comparison:**

```python
=PY(
score = fuzz.ratio('apple', 'appl')
score
)
```

- **Partial ratio:**

```python
=PY(
score = fuzz.partial_ratio('apple pie', 'pie')
score
)
```

- **Token sort ratio:**

```python
=PY(
score = fuzz.token_sort_ratio('new york', 'york new')
score
)
```

## 3. Extracting Best Matches

- **Find best match from a list:**

```python
=PY(
choices = xl("A1:A10")
match, score = process.extractOne('appl', choices)
(match, score)
)
```

- **Find top N matches:**

```python
=PY(
results = process.extract('appl', choices, limit=3)
results
)
```

## 4. Best Practices

- **Imports Once**: Place all import statements on the first worksheet.
- **Data Preparation**: Clean and standardize text before matching.
- **Output Management**: Return scores or tuples for review in Excel.
- **Performance**: For large lists, sample or preprocess data to maintain responsiveness.

By leveraging TheFuzz in Python in Excel, you can perform robust fuzzy matching and text deduplication, making data cleaning and comparison tasks easier within your spreadsheets.
