# Leveraging Scikit-Learn in Python in Excel

Scikit-learn is a cornerstone library for machine learning in Python, and with Python in Excel you can build, evaluate, and deploy models directly within your spreadsheets. Below is a structured guide covering setup, core workflows, and best practices.

## 1. Setup and Imports

To ensure Scikit-learn is available throughout your workbook, reserve the first worksheet for import statements:

```python
=PY(
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error
)
```

This imports key modules—data splitting, scaling, algorithms, and evaluation metrics—into your Python in Excel environment.

## 2. Loading Data

You can load built-in Scikit-learn datasets as pandas DataFrames and display them (“spill”) in Excel:

```python
=PY(
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X  # spills feature table into Excel
)
```

The `as_frame=True` parameter returns pandas DataFrames, making it easy to view and manipulate data directly in your worksheet.

## 3. Preprocessing

### 3.1 Train/Test Split

Divide data into training and testing sets:

```python
=PY(
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
)
```

This creates four DataFrames/Series for model training and evaluation.

### 3.2 Feature Scaling

Standardize features for algorithms that are sensitive to scale:

```python
=PY(
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
)
```

Scaled arrays can then be used with models requiring normalized inputs.

## 4. Model Training and Evaluation

### 4.1 Logistic Regression

```python
=PY(
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
report = classification_report(y_test, pred, output_dict=True)
import pandas as pd; pd.DataFrame(report).transpose()
)
```

This trains a classifier, makes predictions, and returns a transposed report as a table in Excel showing precision, recall, and F1-score.

### 4.2 Random Forest

```python
=PY(
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
)
```

You can spill both actual and predicted labels side by side for comparison in your sheet.

## 5. Regression Example

For numeric targets you can use metrics like mean squared error:

```python
=PY(
from sklearn.datasets import load_diabetes
data = load_diabetes(as_frame=True)
X_d, y_d = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.2, random_state=0)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_tr, y_tr)
mse = mean_squared_error(y_te, ridge.predict(X_te))
mse
)
```

This computes MSE and spills the numeric result into Excel for quick insight.

## 6. Hyperparameter Tuning

Use grid search directly in Excel to optimize parameters:

```python
=PY(
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50,100], 'max_depth': [None, 10]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)
grid.best_params_
)
```

This returns the best parameter combination for your model based on cross-validation.

## 7. Best Practices

- **Imports Once**: Place all import statements on the first worksheet to persist across the workbook.
- **Data Display**: Use `as_frame=True` when loading data to spill DataFrames for inspection.
- **Cell Organization**: Break workflows into sequential cells (preprocessing, training, evaluation) to follow Excel’s row-major execution order.
- **Performance**: For large datasets, sample or aggregate data before model training to keep response times low.
- **Output Management**: Return evaluation metrics as dictionaries or DataFrames for easy integration with native Excel features like charts and conditional formatting.

By embedding Scikit-learn workflows within Excel, you seamlessly integrate advanced modeling into familiar spreadsheets, eliminating context switching and accelerating analytic productivity.
