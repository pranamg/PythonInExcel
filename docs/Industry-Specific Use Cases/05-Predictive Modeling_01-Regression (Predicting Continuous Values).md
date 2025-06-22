This section introduces **Predictive Modeling**, an advanced analytical approach that leverages historical data to predict future outcomes or unknown values.

The first topic in this series is **1. Regression (Predicting Continuous Values)**.

Regression analysis is a powerful predictive modeling technique used to forecast continuous target variables (numbers that can take any value within a range), such as house prices, sales revenue, temperature, or test scores. This approach builds mathematical models that capture relationships between input features (independent variables) and a continuous target variable (dependent variable).

Based on [`piplist.txt`](./README.md) output, you should have `pandas` for data handling, `numpy` for numerical operations, and `scikit-learn` and `statsmodels` which are powerful libraries for building predictive models, including regression. `scikit-learn` is particularly well-suited for the typical machine learning workflow (data splitting, model training, prediction, evaluation). `seaborn` and `matplotlib` are available for visualization.

**Step 1: Generate Sample Data for Regression**

We'll create a dummy dataset with several numerical features and a continuous target variable (`Target_Score`) that has a somewhat linear relationship with the features, plus some random noise. We'll include missing values.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy data for Regression (Predicting Continuous Values)
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

num_records = 1000

data = {
    'ObservationID': range(1, num_records + 1),
    'Feature1_StudyHours': [random.uniform(1, 20) for _ in range(num_records)], # Feature 1
    'Feature2_PrepScore': [random.uniform(50, 95) for _ in range(num_records)], # Feature 2
    'Feature3_PriorAttempts': [random.randint(0, 5) for _ in range(num_records)], # Feature 3 (integer)
    'Feature4_Confidence': [random.uniform(0.5, 5.0) for _ in range(num_records)], # Feature 4 (rating scale)
    'RandomNoise': np.random.normal(0, 5, num_records) # Noise
}

df_reg_data = pd.DataFrame(data)

# Create the target variable as a linear combination of features + noise
# Target_Score = (Feature1 * 2) + (Feature2 * 0.5) + (Feature3 * -3) + (Feature4 * 10) + Noise + Constant
df_reg_data['Target_Score'] = (df_reg_data['Feature1_StudyHours'] * 2 +
                              df_reg_data['Feature2_PrepScore'] * 0.5 +
                              df_reg_data['Feature3_PriorAttempts'] * -3 +
                              df_reg_data['Feature4_Confidence'] * 10 +
                              df_reg_data['RandomNoise'] + 50).round(2) # Add a constant and round

# Ensure Target_Score is within a plausible range (e.g., 0-100)
df_reg_data['Target_Score'] = df_reg_data['Target_Score'].clip(0, 100)


# Introduce some missing values in features and target
for col in ['Feature1_StudyHours', 'Feature2_PrepScore', 'Feature4_Confidence', 'Target_Score']:
    missing_indices = random.sample(range(num_records), int(num_records * random.uniform(0.02, 0.07))) # 2-7% missing
    df_reg_data.loc[missing_indices, col] = np.nan

# Add a few outliers
outlier_indices = random.sample(range(num_records), 3)
df_reg_data.loc[outlier_indices, 'Target_Score'] = [150, -20, 110] # Values outside 0-100 range


# Drop the intermediate Noise column
df_reg_data = df_reg_data.drop(columns=['RandomNoise'])

# Shuffle rows
df_reg_data = df_reg_data.sample(frac=1, random_state=42).reset_index(drop=True)


df_reg_data # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_reg_data` with several numerical features and a continuous `Target_Score`.
*   The `Target_Score` is explicitly created to have a linear dependency on the features, plus some random noise, mimicking data suitable for linear regression.
*   Missing values (`np.nan`) and a few outliers are introduced in features and the target.
*   The result, `df_reg_data`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `RegressionData`.

**Step 2: Build, Evaluate, and Visualize a Regression Model**

Now, we'll load this dummy data, handle missing values, split the data, train a `LinearRegression` model, make predictions, evaluate its performance, and visualize the results.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"RegressionData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Build, evaluate, and visualize a Linear Regression model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # For splitting data
from sklearn.linear_model import LinearRegression # The regression model
from sklearn.impute import SimpleImputer # For handling missing values
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # For evaluation

# Load the data from Excel
# IMPORTANT: Replace "RegressionData" with the actual name of your Excel range or Table
df = xl("RegressionData[#All]", headers=True)

# Ensure numerical columns are numeric, coercing errors
numerical_cols = ['Feature1_StudyHours', 'Feature2_PrepScore', 'Feature3_PriorAttempts', 'Feature4_Confidence', 'Target_Score']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# --- Data Preparation ---
# Drop rows where the target variable is missing, as we can't train/test on those
df_cleaned = df.dropna(subset=['Target_Score']).copy()

# Separate features (X) and target (Y)
X = df_cleaned[['Feature1_StudyHours', 'Feature2_PrepScore', 'Feature3_PriorAttempts', 'Feature4_Confidence']]
Y = df_cleaned['Target_Score']

# Handle missing values in features (using median imputation)
# For simplicity in this example, we impute AFTER dropping rows with missing Y.
# Best practice is often to impute BEFORE splitting, or use a Pipeline, or impute after splitting
# (fit on train, transform train and test). We'll use simple imputation here.
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X) # fit_transform returns a numpy array


# Split data into training and testing sets
# train_size=0.8 means 80% for training, 20% for testing
X_train, X_test, Y_train, Y_test = train_test_split(X_imputed, Y, test_size=0.2, random_state=42) # random_state for reproducibility


# --- Model Training ---
# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)


# --- Prediction ---
# Make predictions on the test set
Y_pred = model.predict(X_test)


# --- Model Evaluation ---
# Calculate common regression metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse) # Root Mean Squared Error
r2 = r2_score(Y_test, Y_pred) # R-squared

# Get model coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Create a DataFrame for evaluation metrics
evaluation_metrics = pd.DataFrame({
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R2)'],
    'Value': [mae, mse, rmse, r2]
})

# Create a DataFrame for model coefficients (Map back to original feature names if possible)
feature_names = ['Feature1_StudyHours', 'Feature2_PrepScore', 'Feature3_PriorAttempts', 'Feature4_Confidence']
coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
coefficients_df.loc[len(coefficients_df)] = ['Intercept', intercept] # Add intercept


# --- Visualization ---

# Apply custom style guidelines
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False # Turn off default grid
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs

# Scatter plot of Actual vs Predicted values
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Use the actual test values and the predicted values
sns.scatterplot(x=Y_test, y=Y_pred, ax=ax1, color='#188ce5', alpha=0.6) # Blue

# Add a diagonal line representing perfect prediction
max_val = max(Y_test.max(), Y_pred.max())
min_val = min(Y_test.min(), Y_pred.min())
ax1.plot([min_val, max_val], [min_val, max_val], color='#ff4136', linestyle='--', linewidth=2) # Salmon dashed line

ax1.set_title('Actual vs. Predicted Target Score (Test Set)', fontsize=14, color='#1a1a24')
ax1.set_xlabel('Actual Target Score', fontsize=12, color='#1a1a24')
ax1.set_ylabel('Predicted Target Score', fontsize=12, color='#1a1a24')
sns.despine(ax=ax1, top=True, right=True)
ax1.grid(False)

plt.tight_layout()


# Optional: Scatter plot of Target vs ONE Feature with regression line
# (Choose the feature with the highest absolute coefficient or the one you want to highlight)
# Need to inverse transform one feature if we used a scaler before model training, but we only imputed here.
# Let's pick Feature1 (StudyHours) as it was designed to have a strong relationship.
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Get the original test data for Feature1 (before imputation/scaling)
# Need to map Y_test back to the original df_cleaned index to get the original X values for plotting
# A simpler approach for plotting is to just plot X_test[:, index of Feature1] vs Y_test
# But X_test is imputed, so the scatter points won't be original values with NaNs.
# Let's plot the original *cleaned* data for Feature1 vs Target_Score, AND the regression line based on the model.
# The regression line for ONE feature needs that feature's coefficient and the intercept.
# Find the coefficient for Feature1_StudyHours
feature1_coef = coefficients_df[coefficients_df['Feature'] == 'Feature1_StudyHours']['Coefficient'].iloc[0]

# Plot original data points (only rows without missing Target_Score)
sns.scatterplot(x=df_cleaned['Feature1_StudyHours'], y=df_cleaned['Target_Score'], ax=ax2, color='#750e5c', alpha=0.6, label='Data Points') # Purple

# Plot the regression line based on the model (using min/max of Feature1)
# The line formula for one feature is: Predicted_Y = Intercept + Coefficient_Feature1 * Feature1
feature1_min = df_cleaned['Feature1_StudyHours'].min()
feature1_max = df_cleaned['Feature1_StudyHours'].max()
x_line = np.array([feature1_min, feature1_max])
y_line = intercept + feature1_coef * x_line
ax2.plot(x_line, y_line, color='#ff6d00', linewidth=2, label='Regression Line') # Orange

ax2.set_title('Target Score vs. Study Hours with Regression Line', fontsize=14, color='#1a1a24')
ax2.set_xlabel('Study Hours (Feature 1)', fontsize=12, color='#1a1a24')
ax2.set_ylabel('Target Score', fontsize=12, color='#1a1a24')
ax2.legend()
sns.despine(ax=ax2, top=True, right=True)
ax2.grid(False)

plt.tight_layout()


# Output results
# Return a dictionary containing evaluation metrics, coefficients, and plots
output = {
    'Regression Evaluation Metrics': evaluation_metrics,
    'Model Coefficients and Intercept': coefficients_df,
    'Actual_vs_Predicted_Plot': fig1,
    'Feature1_vs_Target_Plot': fig2 # Optional plot showing one feature relationship
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy regression data. **Remember to replace `"RegressionData"`**.
*   We ensure numerical columns are correctly typed, coercing errors to `NaN`.
*   **Data Preparation:**
    *   Rows with missing `Target_Score` are removed because they cannot be used for training or evaluation.
    *   The data is split into features (X) and target (Y).
    *   Missing values in the features (X) are handled using `SimpleImputer` with a median strategy. **Note:** For production-level code, imputation should ideally be done within a pipeline or after splitting the data to avoid data leakage from the test set into the training set's imputation statistics. This simplified example imputes before splitting for ease of demonstration in a single cell.
    *   `train_test_split` divides the data into training (80%) and testing (20%) sets. The model is trained on the training data and evaluated on the unseen testing data to assess its generalization performance. `random_state` ensures the split is the same each time you run the code.
*   **Model Training:** A `LinearRegression` model from `sklearn.linear_model` is initialized and trained (`.fit()`) on the training data (`X_train`, `Y_train`).
*   **Prediction:** The trained model is used to predict `Target_Score` values for the test set features (`X_test`) using `.predict()`.
*   **Model Evaluation:** Standard metrics (`mean_absolute_error`, `mean_squared_error`, `r2_score`) are calculated by comparing the model's predictions (`Y_pred`) to the actual values from the test set (`Y_test`). These metrics quantify how well the model performed. MAE and RMSE are measures of prediction error in the original units of the target variable; R-squared indicates the proportion of the variance in the target variable that is predictable from the features.
*   Model coefficients and the intercept are extracted to show the linear equation the model learned (e.g., `Target_Score = intercept + coef1*Feature1 + coef2*Feature2 + ...`).
*   **Visualization:**
    *   `fig1`: A scatter plot compares the actual `Target_Score` values from the test set against the model's predicted values. Points close to the diagonal line indicate good predictions.
    *   `fig2`: An optional scatter plot shows the relationship between the `Target_Score` and one of the predictor features (`Feature1_StudyHours`). The regression line for this single feature relationship (derived from the multi-feature model's intercept and that feature's coefficient) is overlaid to visualize the model's learned linear trend for that specific feature.
*   **Custom Style:** Applied the specified style guidelines (font, colors - blue for actual vs predicted, salmon dashed for perfect line, purple for scatter points, orange for regression line, off-black for text/axes, axes, spines, grid).
*   We return a dictionary containing DataFrames for the evaluation metrics and coefficients, and the two plot figures.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames ('Regression Evaluation Metrics', 'Model Coefficients and Intercept') to spill them into your sheet.
*   For each plot figure object ('Actual_vs_Predicted_Plot', 'Feature1_vs_Target_Plot'), select "Picture in Cell" > "Create Reference" to see the plots.

This section demonstrates a complete regression analysis workflow, from data preparation through model evaluation. The next topic in the series is [Predictive Modeling - 2. Classification (Predicting Categorical Values)](./05-Predictive%20Modeling_02-Classification%20(Predicting%20Categorical%20Values).md), which explores techniques for predicting discrete categories rather than continuous values.

**Further Analysis:**
* **Advanced Regression Models:** Implementing Ridge, Lasso, and Elastic Net regression for regularization and feature selection
* **Non-linear Regression:** Using polynomial features and spline regression for modeling non-linear relationships
* **Ensemble Methods:** Applying Random Forests and Gradient Boosting for regression tasks
* **Cross-Validation Strategies:** Implementing time-based splitting and nested cross-validation for robust model evaluation
* **Feature Engineering:** Creating interaction terms, polynomial features, and custom transformers using scikit-learn pipelines