The next major category in this series is **Data Cleaning & Preparation**, beginning with **1. Data Quality Assessment**.

Data Quality Assessment serves as the foundation for any reliable data analysis process. This essential step systematically identifies and evaluates data quality issues including missing values, duplicates, inconsistent data types, outliers, and structural errors in datasets.

Based on [`piplist.txt`](./README.md) output, we have `pandas` (essential for this), `numpy`, and visualization libraries (`matplotlib`, `seaborn`), all of which are perfect for performing data quality checks.

**Step 1: Generate Sample Data with Data Quality Issues**

We'll create a dummy dataset that intentionally includes common data quality problems like missing values, duplicate rows, mixed data types, potential outliers, and inconsistencies.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy data with various data quality issues
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

num_records = 1000

data = {
    'ID': range(1, num_records + 1),
    'Name': [fake.name() for _ in range(num_records)],
    'Age': [random.randint(18, 70) for _ in range(num_records)],
    'Salary': [round(random.uniform(30000, 120000), 2) for _ in range(num_records)],
    'Department': [random.choice(['Sales', 'Marketing', 'IT', 'Finance', 'HR', None, 'SALES']) for _ in range(num_records)], # Includes None and case inconsistency
    'EnrollmentDate': [fake.date_this_decade() if random.random() > 0.05 else None for _ in range(num_records)], # Missing dates
    'Rating': [random.choice([1, 2, 3, 4, 5, None, 'N/A']) for _ in range(num_records)], # Mixed types and missing
    'Email': [fake.email() for _ in range(num_records)],
    'City': [fake.city() if random.random() > 0.1 else np.nan for _ in range(num_records)], # Missing cities using np.nan
    'Comment': [fake.sentence() if random.random() > 0.2 else '' for _ in range(num_records)], # Empty strings

}

df_dirty = pd.DataFrame(data)

# Introduce some specific issues
# Add missing values
for col in ['Age', 'Salary', 'Department', 'EnrollmentDate']:
    missing_indices = random.sample(range(num_records), int(num_records * random.uniform(0.02, 0.1))) # 2-10% missing
    df_dirty.loc[missing_indices, col] = np.nan

# Add duplicate rows
duplicate_rows = df_dirty.sample(n=int(num_records * 0.05), replace=False) # Select 5% of rows
df_dirty = pd.concat([df_dirty, duplicate_rows], ignore_index=True)

# Add potential outliers in Salary
outlier_indices = random.sample(range(len(df_dirty)), int(len(df_dirty) * 0.01)) # 1% outliers
df_dirty.loc[outlier_indices, 'Salary'] = df_dirty['Salary'] * random.uniform(5, 10) # Multiply salary to create high outliers

# Add some negative Salary outliers (less common but possible data entry error)
negative_outlier_indices = random.sample(range(len(df_dirty)), int(len(df_dirty) * 0.005)) # 0.5% negative outliers
df_dirty.loc[negative_outlier_indices, 'Salary'] = df_dirty['Salary'] * -1 # Make salary negative

# Add mixed data type in a numeric column
# Introduce some strings into the Age column
string_age_indices = random.sample(range(len(df_dirty)), int(len(df_dirty) * 0.01))
df_dirty.loc[string_age_indices, 'Age'] = random.choice(['Twenty Five', 'Thirty', 'Forty Two', 'Unknown'])


# Shuffle rows to mix things up
df_dirty = df_dirty.sample(frac=1, random_state=42).reset_index(drop=True)

df_dirty # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_dirty` that simulates employee or customer records.
*   It uses `Faker` for realistic names, emails, cities, and sentences.
*   It intentionally injects various data quality issues:
    *   Randomly distributed missing values (`None`, `np.nan`).
    *   Case inconsistency in 'Department'.
    *   Mixed data types in 'Rating' (integers, strings, None) and 'Age' (integers, strings).
    *   Duplicate rows.
    *   Outliers (very high and negative values) in 'Salary'.
    *   Empty strings in 'Comment'.
*   The result, `df_dirty`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `DirtyData`.

**Step 2: Perform Data Quality Assessment and Summarize Findings**

Now, let's use Python to systematically identify the issues we just introduced (and any others present in your actual data).

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"DirtyData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Perform data quality assessment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dirty data from Excel
# IMPORTANT: Replace "DirtyData" with the actual name of your Excel range or Table
df = xl("DirtyData[#All]", headers=True)

# --- 1. Missing Value Analysis ---
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Missing Percentage (%)': missing_percentage
})
# Only show columns with missing values, sorted
missing_df_summary = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)


# --- 2. Duplicate Row Analysis ---
duplicate_rows_count = df.duplicated().sum()


# --- 3. Data Type Analysis ---
dtype_summary = df.dtypes.reset_index()
dtype_summary.columns = ['Column', 'DataType']


# --- 4. Unique Value Analysis (for potential inconsistencies or cardinality) ---
unique_value_counts = {}
for col in df.columns:
    # For large number of unique values, just count; for small, list top few
    if df[col].nunique() < 50: # Arbitrary threshold
         # Get value counts, including NaNs
         value_counts = df[col].value_counts(dropna=False)
         # Convert index to string for consistent output in Excel
         value_counts.index = value_counts.index.astype(str)
         unique_value_counts[col] = value_counts.reset_index().rename(columns={'index': 'Value', col: 'Count'})
    else:
        unique_value_counts[col] = pd.DataFrame({'Value': [f'{df[col].nunique()} unique values'], 'Count': [len(df)]})


# --- 5. Outlier Detection (Example for numeric column 'Salary') ---
# Convert 'Salary' to numeric, coercing errors to NaN
df['Salary_numeric'] = pd.to_numeric(df['Salary'], errors='coerce')

# Calculate IQR for 'Salary' (robust to outliers)
Q1 = df['Salary_numeric'].quantile(0.25)
Q3 = df['Salary_numeric'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers_iqr = df[(df['Salary_numeric'] < lower_bound) | (df['Salary_numeric'] > upper_bound)]

# --- 6. Mixed Data Type Detection (Example for 'Age', 'Rating') ---
# Check for non-numeric values in 'Age'
non_numeric_age = df[pd.to_numeric(df['Age'], errors='coerce').isna() & df['Age'].notna()] # Find non-numeric entries that are not NaN

# Check for non-standard values in 'Rating' (assuming standard is 1-5 integers)
# First, convert potential numeric strings to numeric
df['Rating_numeric'] = pd.to_numeric(df['Rating'], errors='coerce')
# Find entries that are NaN in numeric conversion but not NaN in original, or outside 1-5
non_standard_rating = df[(df['Rating_numeric'].isna() & df['Rating'].notna()) | (~df['Rating_numeric'].isna() & ~df['Rating_numeric'].isin([1, 2, 3, 4, 5]))]


# --- Visualization for Outlier Detection ---

# Apply custom style guidelines
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False # Turn off default grid
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs


# Box plot for Salary to visualize distribution and outliers
fig1, ax1 = plt.subplots(figsize=(8, 6))
# Use the original 'Salary' column which might have strings, let seaborn/pandas handle plotting numerics
# Or better, use the 'Salary_numeric' column we created, dropping NaNs for plotting
sns.boxplot(y=df['Salary_numeric'].dropna(), ax=ax1, color='#ff6d00') # Orange

ax1.set_title('Distribution and Outliers of Salary', fontsize=14, color='#1a1a24')
ax1.set_ylabel('Salary', fontsize=12, color='#1a1a24')
# Hide x-axis label and ticks as it's a single box plot
ax1.set_xticklabels([])
ax1.set_xlabel('')
sns.despine(ax=ax1, top=True, right=True, bottom=True, left=False) # Remove top, right, bottom spines

plt.tight_layout()


# Output results
# Combine summaries into a dictionary for return
output = {
    'Missing Values Summary': missing_df_summary,
    'Duplicate Rows Count': pd.DataFrame({'Metric': ['Total Duplicate Rows'], 'Count': [duplicate_rows_count]}), # Return as DataFrame
    'Data Types Summary': dtype_summary,
    'Unique Value Counts (Sample/Small N)': unique_value_counts, # Dictionary of DataFrames
    'Salary Outliers (IQR Method)': outliers_iqr,
    'Non-Numeric Age Entries': non_numeric_age,
    'Non-Standard Rating Entries': non_standard_rating,
    'Salary_Outlier_Box_Plot': fig1,
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy "dirty" data using `xl()`. **Remember to replace `"DirtyData"`**.
*   We calculate missing values per column using `isnull().sum()` and the percentage.
*   We count duplicate rows using `duplicated().sum()`.
*   We get a summary of detected data types using `dtypes`.
*   We analyze unique values for each column. If there are few unique values (< 50), we list the value counts; otherwise, we just report the total unique count. This helps spot variations like "Sales" vs "SALES" or unexpected string values.
*   We perform a basic outlier detection for the 'Salary' column using the Interquartile Range (IQR) method. We first ensure the column is numeric (`pd.to_numeric(errors='coerce')` is crucial here, turning errors into `NaN` so they don't break calculations).
*   We check for mixed data types by trying to convert columns to numeric and identifying rows where this conversion fails but the original value wasn't missing. We also specifically look for values outside the expected range for 'Rating'.
*   **Visualization:** A box plot is created for the `Salary_numeric` column. Box plots are excellent for visually showing the distribution, median, quartiles, and identifying potential outliers (points plotted individually beyond the "whiskers").
*   **Custom Style:** Applied the specified style guidelines (font, color for the box plot, axes, spines, grid, removed bottom axis ticks/label for clarity).
*   We return a dictionary containing various DataFrames summarizing the data quality findings (missing values, duplicates, dtypes, value counts, specific outlier/mixed type entries) and the box plot figure.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames ('Missing Values Summary', 'Duplicate Rows Count', 'Data Types Summary', 'Salary Outliers (IQR Method)', 'Non-Numeric Age Entries', 'Non-Standard Rating Entries'). Note that 'Unique Value Counts (Sample/Small N)' is a dictionary itself; you'll access its contents by referencing the cell (e.g., `=PY(A1["Department"])`) and then converting that cell to Excel Value.
*   For the plot figure object ('Salary_Outlier_Box_Plot'), select "Picture in Cell" > "Create Reference" to see the plot.

This comprehensive assessment provides a clear understanding of data quality issues present in the dataset. The next topic in the series is ["Data Cleaning & Preparation - 2. Data Transformation"](./03-Data%20Cleaning%20&%20Preparation_02-Data%20Transformation.md), which covers techniques for addressing the types of issues identified here, including handling missing values and fixing mixed data types.

**Further Analysis:**
* **Advanced Anomaly Detection:** Using isolation forests or autoencoders to detect complex anomalies in multivariate data
* **Data Quality Scoring:** Implementing a weighted scoring system based on completeness, accuracy, consistency, and timeliness
* **Pattern Recognition:** Using association rule mining to identify data quality patterns and dependencies
* **Time Series Quality:** Analyzing temporal consistency, seasonality violations, and detecting change points in time-based data
* **Cross-Validation Analysis:** Implementing k-fold validation to assess data quality metrics' stability across different subsets