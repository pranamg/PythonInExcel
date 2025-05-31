Okay, let's tackle **Predictive Modeling - 2. Classification (Predicting Categorical Values)**.

Classification is used when your goal is to predict which category or class an observation belongs to. This is common in tasks like predicting customer churn (Churn/No Churn), identifying fraudulent transactions (Fraud/Not Fraud), or categorizing emails (Spam/Not Spam).

Based on your `piplist.txt`, you have `pandas`, `numpy`, `scikit-learn` (which provides a wide array of classification algorithms, data splitting tools, and evaluation metrics), `seaborn`, and `matplotlib`. This is exactly what we need for a typical classification workflow.

**Step 1: Generate Sample Data for Classification**

We'll create a dummy dataset representing customer features and a binary target variable indicating whether the customer churned or not (`IsChurn`). The features will have some relationship with the likelihood of churning. We'll include missing values.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy data for Classification (Predicting Churn)
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

num_records = 1500

data = {
    'CustomerID': [f'Cust{1000 + i}' for i in range(num_records)],
    'MonthlyCharges': [round(random.uniform(20, 120), 2) for _ in range(num_records)],
    'TotalCharges': [round(random.uniform(100, 8000), 2) for _ in range(num_records)],
    'ContractType': [random.choice(['Month-to-month', 'One year', 'Two year']) for _ in range(num_records)],
    'ServiceDuration_Months': [random.randint(1, 72) for _ in range(num_records)],
    'NumServiceCalls': [random.randint(0, 10) for _ in range(num_records)],
    'SupportScore': [random.uniform(1, 5) for _ in range(num_records)], # Higher score means better support
    'PaymentMethod': [random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']) for _ in range(num_records)],
}

df_clf_data = pd.DataFrame(data)

# Simulate 'IsChurn' based on some features
# Higher MonthlyCharges, Month-to-month contract, shorter duration, more service calls, lower support score increase churn probability
def determine_churn(row):
    churn_prob = 0.15 # Base churn probability

    if row['MonthlyCharges'] > 80:
        churn_prob += 0.10
    if row['ContractType'] == 'Month-to-month':
        churn_prob += 0.25
    if row['ServiceDuration_Months'] < 12:
        churn_prob += 0.15
    if row['NumServiceCalls'] > 3:
        churn_prob += 0.10 * (row['NumServiceCalls'] - 3) # Higher calls increase prob
    if row['SupportScore'] < 3:
         churn_prob += 0.15 * (3 - row['SupportScore']) # Lower score increases prob

    # Add some randomness
    churn_prob += np.random.normal(0, 0.1)

    # Ensure probability is between 0 and 1
    churn_prob = max(0, min(1, churn_prob))

    # Randomly assign churn based on probability
    return random.random() < churn_prob

# Apply the function to create the 'IsChurn' column (True/False)
df_clf_data['IsChurn'] = df_clf_data.apply(determine_churn, axis=1)

# Introduce some missing values
for col in ['MonthlyCharges', 'TotalCharges', 'ServiceDuration_Months', 'SupportScore', 'NumServiceCalls']:
    missing_indices = random.sample(range(num_records), int(num_records * random.uniform(0.02, 0.06))) # 2-6% missing
    df_clf_data.loc[missing_indices, col] = np.nan

# Introduce missing values in a categorical column
missing_cat_indices = random.sample(range(num_records), int(num_records * 0.04)) # 4% missing
df_clf_data.loc[missing_cat_indices, 'PaymentMethod'] = np.nan


# Shuffle rows
df_clf_data = df_clf_data.sample(frac=1, random_state=42).reset_index(drop=True)


df_clf_data # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_clf_data` with simulated customer data including numerical features (charges, duration, calls, support score) and categorical features (contract type, payment method), plus a binary target variable `IsChurn`.
*   The `IsChurn` target is generated with a probability that depends on the feature values, creating a realistic (though simplified) relationship.
*   Missing values (`np.nan`, `None`) are introduced in both numerical and categorical columns.
*   The result, `df_clf_data`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `ClassificationData`.

**Step 2: Build, Evaluate, and Visualize a Classification Model**

Now, we'll load this dummy data, handle missing values, encode categorical features, split the data, train a `LogisticRegression` model, make predictions, evaluate its performance using metrics and a confusion matrix, and visualize the confusion matrix.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"ClassificationData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Build, evaluate, and visualize a Classification model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # For splitting data
from sklearn.linear_model import LogisticRegression # A common classification model
from sklearn.impute import SimpleImputer # For handling missing values
from sklearn.preprocessing import OneHotEncoder, StandardScaler # For encoding and scaling
from sklearn.compose import ColumnTransformer # To apply different transformers to different columns
from sklearn.pipeline import Pipeline # To chain transformations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay # For evaluation and visualization


# Load the data from Excel
# IMPORTANT: Replace "ClassificationData" with the actual name of your Excel range or Table
df = xl("ClassificationData[#All]", headers=True)

# Ensure target column is boolean and features are appropriate types
# Coerce errors for numerical columns
numerical_cols = ['MonthlyCharges', 'TotalCharges', 'ServiceDuration_Months', 'NumServiceCalls', 'SupportScore']
categorical_cols = ['ContractType', 'PaymentMethod'] # Include payment method with NaNs
# CustomerID is an identifier, not a feature for the model

df['IsChurn'] = df['IsChurn'].astype(bool) # Ensure target is boolean
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for col in categorical_cols:
     # Ensure categorical columns are treated as strings for consistent imputation/encoding
     df[col] = df[col].astype(str).replace('nan', np.nan) # Convert 'nan' string from Excel to np.nan


# --- Data Preparation ---
# Drop rows where the target variable is missing (if any were generated, though unlikely for binary churn)
df_cleaned = df.dropna(subset=['IsChurn']).copy()

# Separate features (X) and target (Y)
X = df_cleaned[numerical_cols + categorical_cols]
Y = df_cleaned['IsChurn']

# Create preprocessing pipelines for different column types
# Numeric pipeline: Impute with median, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute with most frequent, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Use most_frequent for strings
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False for dense array
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # Keep other columns if any (like CustomerID if not dropped) - though we excluded it from X
)

# Create the full pipeline: preprocessing + model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear')) # Use liblinear for smaller datasets or L1 regularization
])


# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y) # Stratify to maintain class distribution


# --- Model Training ---
# Train the full pipeline (preprocessing + model)
model_pipeline.fit(X_train, Y_train)


# --- Prediction ---
# Make predictions on the test set
Y_pred = model_pipeline.predict(X_test)


# --- Model Evaluation ---
# Calculate common classification metrics
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred) # For positive class (True/Churn)
recall = recall_score(Y_test, Y_pred)     # For positive class (True/Churn)
f1 = f1_score(Y_test, Y_pred)           # Harmonic mean of precision and recall

# Calculate Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

# Create a DataFrame for evaluation metrics
evaluation_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Churn)', 'Recall (Churn)', 'F1-Score (Churn)'],
    'Value': [accuracy, precision, recall, f1]
})

# Create a DataFrame for the Confusion Matrix
cm_df = pd.DataFrame(cm, index=['Actual Not Churn', 'Actual Churn'], columns=['Predicted Not Churn', 'Predicted Churn'])


# --- Visualization ---

# Apply custom style guidelines
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False # Turn off default grid
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs

# Plot Confusion Matrix
fig1, ax1 = plt.subplots(figsize=(7, 6))
# Use ConfusionMatrixDisplay for easier plotting
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Churn', 'Churn'])
# Use a sequential color map, e.g., 'Blues' or pick from guidelines
# Let's use a guideline color for the heatmap, e.g. blue shades or green shades
# Blues colormap goes from light blue to dark blue
disp.plot(cmap='Blues', ax=ax1, values_format='d') # 'd' for integer formatting

ax1.set_title('Confusion Matrix', fontsize=14, color='#1a1a24')
ax1.set_xlabel('Predicted Label', fontsize=12, color='#1a1a24')
ax1.set_ylabel('Actual Label', fontsize=12, color='#1a1a24')
# Customize text color within the matrix squares if needed
for labels in ax1.texts:
    labels.set_color("#1a1a24") # Make text off-black

plt.tight_layout()


# Output results
# Return a dictionary containing evaluation metrics, confusion matrix, and plot
output = {
    'Classification Evaluation Metrics': evaluation_metrics,
    'Confusion Matrix': cm_df,
    'Confusion_Matrix_Plot': fig1
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy classification data. **Remember to replace `"ClassificationData"`**.
*   We ensure the target variable `IsChurn` is treated as boolean. Numerical columns are converted using `pd.to_numeric(errors='coerce')`, and categorical columns are explicitly converted to string/object type, ensuring `NaN` is correctly represented.
*   **Data Preparation:**
    *   Rows with missing `IsChurn` are dropped (though our data generation didn't create these).
    *   The data is split into features (X) and target (Y).
    *   We use `scikit-learn` Pipelines and `ColumnTransformer` for structured preprocessing:
        *   Numerical features are imputed with the median and then scaled using `StandardScaler`.
        *   Categorical features are imputed with the most frequent value and then one-hot encoded using `OneHotEncoder`. `handle_unknown='ignore'` helps if new categories appear in the test set. `sparse_output=False` gives a dense array.
    *   The `preprocessor` step is combined with the `LogisticRegression` model into a single `Pipeline`. This simplifies the workflow and helps prevent data leakage during training and prediction.
    *   `train_test_split` divides the data (features and target) into training (75%) and testing (25%) sets. `stratify=Y` is important for classification, ensuring the proportion of the target classes (Churn/Not Churn) is similar in both the training and testing sets.
*   **Model Training:** The `model_pipeline` (which includes both preprocessing and the classifier) is trained (`.fit()`) on the training data (`X_train`, `Y_train`). The pipeline automatically applies the preprocessing steps before training the classifier.
*   **Prediction:** The trained pipeline is used to make predictions (`.predict()`) on the raw test set features (`X_test`). The pipeline automatically applies the same preprocessing steps used during training to `X_test` before passing it to the classifier.
*   **Model Evaluation:**
    *   Common classification metrics (`accuracy`, `precision`, `recall`, `f1_score`) are calculated by comparing the model's predictions (`Y_pred`) to the actual test set values (`Y_test`).
    *   A `confusion_matrix` is calculated. This matrix shows the counts of True Positives (correctly predicted Churn), True Negatives (correctly predicted Not Churn), False Positives (incorrectly predicted Churn), and False Negatives (incorrectly predicted Not Churn).
*   **Visualization:**
    *   `fig1`: A heatmap of the confusion matrix is plotted using `sklearn.metrics.ConfusionMatrixDisplay`, which is the standard way to visualize classification performance in terms of correct and incorrect predictions for each class. `cmap='Blues'` uses a blue color scheme.
*   **Custom Style:** Applied the specified style guidelines (font, colors - using a blue color map for the heatmap, off-black for text/axes and within the matrix, axes, spines, grid - though heatmap usually doesn't need grid).
*   We return a dictionary containing DataFrames for the evaluation metrics and the confusion matrix counts, and the confusion matrix plot figure.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames ('Classification Evaluation Metrics', 'Confusion Matrix') to spill them into your sheet.
*   For the plot figure object ('Confusion_Matrix_Plot'), select "Picture in Cell" > "Create Reference" to see the plot.

This completes the basic classification analysis workflow. Would you like to proceed to the next use case: "Predictive Modeling - 3. Time Series Forecasting"?