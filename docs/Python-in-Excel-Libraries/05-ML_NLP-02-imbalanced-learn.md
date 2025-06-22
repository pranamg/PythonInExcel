# Leveraging imbalanced-learn in Python in Excel

The **imbalanced-learn** library (imported as `imblearn`) offers resampling techniques to address class imbalance in machine-learning tasks directly within Excel’s Python integration. By combining Excel’s familiar interface with `imbalanced-learn`’s sampling methods, analysts can build balanced datasets for improved model performance without leaving their spreadsheets.

## 1. Setup and Imports

To ensure smooth usage, reserve the first worksheet for importing required packages. Enter the following Python formula:

```python
=PY(
import imblearn      # core library for resampling methods
import pandas as pd  # for DataFrame manipulation
from sklearn.model_selection import train_test_split
)
```

This import persists across the workbook, making `imblearn` and its functions available for all subsequent Python cells.

## 2. Loading and Splitting Data

Use Excel ranges or tables as pandas DataFrames, then split into training and test sets while preserving class proportions:

```python
=PY(
df = xl("DataTable[#All]", headers=True)
X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
)
```

Here, `stratify=y` maintains the original class distribution in both subsets.

## 3. Resampling Techniques

`imbalanced-learn` offers three main categories of methods:

### 3.1 Over-Sampling

Create synthetic minority samples with SMOTE:

```python
=PY(
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
pd.DataFrame({"Before":[y_train.value_counts()[1], y_train.value_counts()[0]],
              "After":[sum(y_res==1), sum(y_res==0)]})
)
```

### 3.2 Under-Sampling

Randomly remove majority samples:

```python
=PY(
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
pd.DataFrame({"Before":[y_train.value_counts()[1], y_train.value_counts()[0]],
              "After":[sum(y_res==1), sum(y_res==0)]})
)
```

### 3.3 Combined Methods

Use both over- and under-sampling:

```python
=PY(
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X_train, y_train)
pd.DataFrame({"Before":[y_train.value_counts()[1], y_train.value_counts()[0]],
              "After":[sum(y_res==1), sum(y_res==0)]})
)
```

## 4. Best Practices

- **Imports Once**: Place all import statements on the first worksheet to persist across the workbook.
- **Data Preparation**: Clean and preprocess data with pandas before resampling.
- **Stratified Splitting**: Always use `stratify=y` in `train_test_split` to maintain class proportions.
- **Output Management**: Return summary DataFrames to Excel for easy inspection of class balance before and after resampling.

By leveraging imbalanced-learn in Python in Excel, you can address class imbalance issues and improve the robustness of your machine learning models—all within the familiar spreadsheet environment.
