The next topic in the statistical analysis series is **Statistical Analysis - 2. Inferential Statistics**.

Inferential statistics extends beyond descriptive analysis by enabling conclusions about larger populations based on sample data. This advanced statistical approach employs hypothesis testing to evaluate the significance of observed differences or relationships, and uses confidence intervals to estimate population parameters.

Based on [`piplist.txt`](./README.md) output, you have `pandas`, `numpy`, `scipy` (which includes modules for statistical tests), `statsmodels` (for regression and other models), `seaborn`, and `matplotlib`. This set of libraries is excellent for a wide range of inferential statistical tasks.

**Step 1: Generate Sample Data for Inferential Statistics**

We'll create a dummy dataset that can be used to perform a t-test comparing a numerical variable between two groups, and to examine the relationship between two numerical variables using correlation and regression.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy data for Inferential Statistics
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

num_records_group_a = 150
num_records_group_b = 180
num_total_records = num_records_group_a + num_records_group_b

# Simulate a numerical metric (e.g., Test Score) with a slight difference between two groups
# Group A (e.g., Standard Method)
scores_a = np.random.normal(loc=75, scale=10, size=num_records_group_a) # Mean 75, Std Dev 10
group_a_data = pd.DataFrame({
    'SubjectID': [f'A{i}' for i in range(num_records_group_a)],
    'Group': 'Standard Method',
    'TestScore': scores_a.round(1),
    'StudyHours': (np.random.normal(loc=5, scale=2, size=num_records_group_a) + scores_a/20).round(1).clip(0, 20) # Simulate StudyHours somewhat correlated with Score
})

# Group B (e.g., New Method) - slightly higher mean score
scores_b = np.random.normal(loc=78, scale=10, size=num_records_group_b) # Mean 78, Std Dev 10
group_b_data = pd.DataFrame({
    'SubjectID': [f'B{i}' for i in range(num_records_group_b)],
    'Group': 'New Method',
    'TestScore': scores_b.round(1),
    'StudyHours': (np.random.normal(loc=6, scale=2.5, size=num_records_group_b) + scores_b/18).round(1).clip(0, 25) # Simulate StudyHours somewhat correlated with Score
})

# Combine dataframes
df_inferential = pd.concat([group_a_data, group_b_data], ignore_index=True)

# Add another numerical variable (e.g., Abstract Reasoning Score)
df_inferential['AbstractReasoning'] = np.random.normal(loc=50, scale=12, size=num_total_records).round(1).clip(10, 90)
# Make it slightly correlated with TestScore
df_inferential['TestScore'] = df_inferential['TestScore'] + df_inferential['AbstractReasoning']*0.1 + np.random.normal(0, 3, size=num_total_records)
df_inferential['TestScore'] = df_inferential['TestScore'].round(1).clip(0, 100) # Keep scores between 0-100


# Add some missing values
missing_indices_score = random.sample(range(num_total_records), int(num_total_records * 0.03))
df_inferential.loc[missing_indices_score, 'TestScore'] = np.nan

missing_indices_hours = random.sample(range(num_total_records), int(num_total_records * 0.05))
df_inferential.loc[missing_indices_hours, 'StudyHours'] = np.nan

missing_indices_reasoning = random.sample(range(num_total_records), int(num_total_records * 0.04))
df_inferential.loc[missing_indices_reasoning, 'AbstractReasoning'] = np.nan


# Shuffle rows
df_inferential = df_inferential.sample(frac=1, random_state=42).reset_index(drop=True)

df_inferential # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_inferential` simulating test results for subjects in two different groups ('Standard Method' and 'New Method').
*   It includes `SubjectID`, `Group`, `TestScore`, `StudyHours`, and `AbstractReasoning` score.
*   `TestScore` for the 'New Method' group is simulated with a slightly higher mean to potentially show a significant difference in the t-test.
*   `StudyHours` and `AbstractReasoning` are simulated with some correlation to `TestScore` for correlation/regression examples.
*   Missing values (`np.nan`) are introduced in numerical columns.
*   The result, `df_inferential`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `TestData`.

**Step 2: Perform Inferential Statistical Tests and Visualize Relationships**

Now, we'll load this dummy data and perform:
1.  An independent samples t-test to compare the `TestScore` means between the 'Standard Method' and 'New Method' groups.
2.  Calculate the correlation matrix between the numerical variables.
3.  Perform a simple linear regression to predict `TestScore` based on `StudyHours`.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"TestData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Perform inferential statistical tests and visualize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For t-test, correlation
import statsmodels.api as sm # For regression

# Load the data from Excel
# IMPORTANT: Replace "TestData" with the actual name of your Excel range or Table
df = xl("TestData[#All]", headers=True)

# Ensure appropriate data types, coercing errors for numerical columns
df['TestScore'] = pd.to_numeric(df['TestScore'], errors='coerce')
df['StudyHours'] = pd.to_numeric(df['StudyHours'], errors='coerce')
df['AbstractReasoning'] = pd.to_numeric(df['AbstractReasoning'], errors='coerce')
df['Group'] = df['Group'].astype(str)


# --- 1. Independent Samples T-test ---
# Compare TestScore mean between 'Standard Method' and 'New Method' groups

# Separate the data by group, dropping rows with missing TestScore for the test
group_a_scores = df[df['Group'] == 'Standard Method']['TestScore'].dropna()
group_b_scores = df[df['Group'] == 'New Method']['TestScore'].dropna()

# Check if both groups have enough data (at least 2 samples each)
if len(group_a_scores) > 1 and len(group_b_scores) > 1:
    # Perform independent samples t-test (assuming equal variance is default in stats.ttest_ind)
    # For unequal variance (Welch's t-test), use equal_var=False
    t_stat, p_value_ttest = stats.ttest_ind(group_a_scores, group_b_scores, equal_var=True) # You might check variance equality first, but let's use basic t-test

    ttest_result = pd.DataFrame({
        'Metric': ['T-Statistic', 'P-Value (T-test)', 'Mean (Standard Method)', 'Mean (New Method)', 'Std Dev (Standard)', 'Std Dev (New)'],
        'Value': [t_stat, p_value_ttest, group_a_scores.mean(), group_b_scores.mean(), group_a_scores.std(), group_b_scores.std()]
    })
    # Add significance interpretation
    significance_ttest = 'Statistically Significant (p < 0.05)' if p_value_ttest < 0.05 else 'Not Statistically Significant (p >= 0.05)'
    ttest_result = pd.concat([ttest_result, pd.DataFrame({'Metric':['Significance (alpha=0.05)'], 'Value':[significance_ttest]})], ignore_index=True)
else:
    ttest_result = pd.DataFrame({'Result': ["Not enough valid data in one or both groups to perform T-test."]})


# --- 2. Correlation Analysis ---
# Calculate pairwise correlation between numerical columns
# Drop rows with ANY missing values across the selected columns for clean correlation calculation
numerical_cols = ['TestScore', 'StudyHours', 'AbstractReasoning']
correlation_matrix = df[numerical_cols].dropna().corr()


# --- 3. Simple Linear Regression ---
# Predict TestScore (dependent variable Y) using StudyHours (independent variable X)
# Drop rows with missing values in either variable
df_reg = df[['TestScore', 'StudyHours']].dropna()

if len(df_reg) > 1:
    # Define Y and X
    Y = df_reg['TestScore']
    X = df_reg['StudyHours']

    # Add a constant to the independent variable for the intercept (statsmodels convention)
    X = sm.add_constant(X)

    # Fit the OLS (Ordinary Least Squares) model
    model = sm.OLS(Y, X).fit()

    # Get the regression summary
    # statsmodels summary object doesn't convert easily to a simple DataFrame for Excel.
    # Extract key parts: coefficients, p-values, R-squared
    regression_summary_df = model.summary2().tables[1] # Table of coefficients, std err, t, P>|t|, intervals
    regression_r2 = model.rsquared
    regression_adj_r2 = model.rsquared_adj

    # Include R-squared and Adjusted R-squared in a separate df or add rows
    regression_summary_df.loc['R-squared'] = [regression_r2, np.nan, np.nan, np.nan, np.nan, np.nan]
    regression_summary_df.loc['Adj. R-squared'] = [regression_adj_r2, np.nan, np.nan, np.nan, np.nan, np.nan]

    # Rename P>|t| column for clarity in Excel
    regression_summary_df = regression_summary_df.rename(columns={'P>|t|': 'P_Value'})


else:
     regression_summary_df = pd.DataFrame({'Result': ["Not enough valid data to perform Regression."]})


# --- Visualization ---

# Apply custom style guidelines
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False # Turn off default grid
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs


# 1. Box Plot of Test Scores by Group (for T-test visualization)
fig1, ax1 = plt.subplots(figsize=(8, 6))
# Use the original DataFrame, seaborn handles grouping and plots valid data
sns.boxplot(x='Group', y='TestScore', hue='Group', legend=False, data=df, ax=ax1, palette=['#188ce5', '#2db757']) # Blue, Green


ax1.set_title('Test Score Distribution by Method', fontsize=14, color='#1a1a24')
ax1.set_xlabel('Method', fontsize=12, color='#1a1a24')
ax1.set_ylabel('Test Score', fontsize=12, color='#1a1a24')
sns.despine(ax=ax1, top=True, right=True)
ax1.grid(False)

plt.tight_layout()


# 2. Scatter Plot of Test Score vs Study Hours (for Regression visualization)
fig2, ax2 = plt.subplots(figsize=(8, 6))
# Use the data used for regression (NaNs removed)
sns.scatterplot(x='StudyHours', y='TestScore', data=df_reg, ax=ax2, color='#750e5c', alpha=0.6) # Purple

# Add the regression line
if len(df_reg) > 1:
    # Predict values using the fitted model for the line
    # Ensure the X for prediction includes the constant and is sorted for a smooth line
    X_pred = pd.DataFrame({'StudyHours': np.linspace(df_reg['StudyHours'].min(), df_reg['StudyHours'].max(), 100)})
    X_pred = sm.add_constant(X_pred, has_constant='add')
    Y_pred = model.predict(X_pred)
    ax2.plot(X_pred['StudyHours'], Y_pred, color='#ff6d00', linewidth=2) # Orange regression line


ax2.set_title('Test Score vs. Study Hours with Regression Line', fontsize=14, color='#1a1a24')
ax2.set_xlabel('Study Hours', fontsize=12, color='#1a1a24')
ax2.set_ylabel('Test Score', fontsize=12, color='#1a1a24')
sns.despine(ax=ax2, top=True, right=True)
ax2.grid(False)

plt.tight_layout()


# Output results
# Return a dictionary containing the results DataFrames and plots
output = {
    'Independent T-test Result (Score by Group)': ttest_result,
    'Correlation Matrix (Numeric Columns)': correlation_matrix,
    'Linear Regression Summary (Score vs Hours)': regression_summary_df,
    'Test_Score_Box_Plot_by_Group': fig1,
    'Score_vs_Hours_Scatter_Plot': fig2,
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy test data. **Remember to replace `"TestData"`**.
*   We ensure columns have appropriate data types, coercing errors to `NaN` for numerical columns.
*   **T-test:** We filter the `TestScore` data for each group, drop missing values for the test (`dropna()`), and use `scipy.stats.ttest_ind()` to perform an independent samples t-test. The result includes the t-statistic and the p-value, which tells you if the difference in means between the groups is statistically significant.
*   **Correlation:** We select the numerical columns, drop rows with any missing values across these columns, and use the `.corr()` method to calculate the pairwise Pearson correlation matrix.
*   **Linear Regression:** We select `TestScore` (Y) and `StudyHours` (X), drop rows with missing values in either, use `statsmodels.api.add_constant()` to include an intercept term, fit an Ordinary Least Squares (OLS) model using `sm.OLS().fit()`, and extract key parts of the summary into a DataFrame.
*   **Visualization:**
    *   `fig1`: A box plot shows the distribution of `TestScore` for each `Group`. This visually complements the t-test result, showing means, quartiles, and potential outliers within each group.
    *   `fig2`: A scatter plot shows the relationship between `StudyHours` and `TestScore`. The fitted regression line is overlaid to visualize the linear trend captured by the regression model.
*   **Custom Style:** Applied the specified style guidelines (font, colors - specific palette for box plot, purple for scatter points, orange for regression line, off-black for text/axes, axes, spines, grid).
*   We return a dictionary containing the DataFrames summarizing the t-test result, the correlation matrix, the regression summary, and the two plot figures.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames ('Independent T-test Result (Score by Group)', 'Correlation Matrix (Numeric Columns)', 'Linear Regression Summary (Score vs Hours)') to spill them into your sheet.
*   For each plot figure object ('Test_Score_Box_Plot_by_Group', 'Score_vs_Hours_Scatter_Plot'), select "Picture in Cell" > "Create Reference" to see the plots.

These examples demonstrate key inferential statistical techniques including hypothesis testing (t-test), relationship measurement (correlation), and predictive modeling (regression). The next topic in the series is ["Statistical Analysis - 3. Time Series Analysis"](./04-Statistical%20Analysis_03-Time%20Series%20Analysis.md), which focuses on analyzing and interpreting temporal patterns in data.

**Further Analysis:**
* **Advanced Hypothesis Testing:** Implementing ANOVA, MANOVA, and non-parametric tests like Mann-Whitney U and Kruskal-Wallis
* **Power Analysis:** Using statsmodels for sample size determination and statistical power calculations
* **Mixed Effects Models:** Building hierarchical/multilevel models for nested or repeated measures data
* **Multiple Testing Correction:** Implementing Bonferroni, FDR, and other methods to control familywise error rate
* **Bayesian Inference:** Using PyMC3 or Stan for Bayesian parameter estimation and hypothesis testing