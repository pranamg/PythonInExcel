**Reporting & Automation - 4. Conditional Formatting**

Based on [`piplist.txt`](./README.md) output, you should have `pandas`, `numpy`, and `matplotlib` excel at data analysis, they don't directly control Excel's formatting features. However, Python can identify which data points should be formatted based on complex logic or calculations.

This guide demonstrates the following workflow for conditional formatting with Python:
1. Load data into Python
2. Read formatting parameters from Excel cells
3. Apply logic to identify rows meeting formatting criteria
4. Add indicator columns (boolean flags) to mark qualifying rows
5. Output the enhanced data back to Excel
6. Use Excel's conditional formatting features to apply visual styles based on the indicator columns

This approach combines Python's analytical power with Excel's formatting capabilities to create visually informative reports.

We will generate dummy data and then write Python code that flags sales records where the `Amount` exceeds a threshold specified in an Excel cell.

**Step 1: Generate Sample Data for Conditional Formatting**

We'll create a simple sales dataset with columns including an `Amount` and a `Region`, suitable for highlighting values based on a threshold.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy sales data for Conditional Formatting example
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta

fake = Faker()

num_records = 1000

regions = ['North', 'South', 'East', 'West']

data = []
for i in range(num_records):
    transaction_date = fake.date_this_year()
    region = random.choice(regions)
    amount = round(random.uniform(50, 2500), 2) # Base amount range

    data.append([transaction_date, region, amount])

df_cf_data = pd.DataFrame(data, columns=['Date', 'Region', 'Amount'])

# Add some higher values and potential outliers
high_value_indices = random.sample(range(num_records), 50) # 50 relatively high values
df_cf_data.loc[high_value_indices, 'Amount'] = df_cf_data.loc[high_value_indices, 'Amount'] * random.uniform(1.5, 3.0)
outlier_indices = random.sample(range(num_records), 5) # 5 extreme outliers
df_cf_data.loc[outlier_indices, 'Amount'] = df_cf_data.loc[outlier_indices, 'Amount'] * random.uniform(10, 20)

# Ensure Date is datetime
df_cf_data['Date'] = pd.to_datetime(df_cf_data['Date'])

# Add some missing values
missing_indices = random.sample(range(num_records), int(num_records * 0.02)) # 2% missing
df_cf_data.loc[missing_indices, 'Amount'] = np.nan


# Shuffle rows
df_cf_data = df_cf_data.sample(frac=1, random_state=42).reset_index(drop=True)


df_cf_data # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_cf_data` with Date, Region, and Amount columns.
*   It includes a base range of amounts, plus some artificially higher values and outliers to make conditional formatting based on a threshold meaningful.
*   Missing values are introduced in the `Amount` column.
*   The result, `df_cf_data`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `ConditionalFormatData`.

**Step 2: Create an Excel Cell for the Threshold and Write Parameterized Python Code to Flag Data**

Now, you need to set up an Excel cell for your threshold value.

1.  **Choose an empty cell** in your Excel sheet (e.g., cell `A1`).
2.  **Enter a label** next to it (e.g., 'High Sales Threshold' in `B1`).
3.  **Enter your desired threshold value** in cell `A1`. For example, type `3000`.

Now, in a **new** Excel cell, enter `=PY` and paste the following code. This code will load the data and the threshold value from your Excel cell, and then add a flag column.

**IMPORTANT:** Replace `"ConditionalFormatData"` with the actual name of your Excel range/Table, and replace `"A1"` with the actual cell reference you used for your threshold parameter. Press **Ctrl+Enter**.

```python
# Flag sales amounts exceeding a threshold from an Excel cell
import pandas as pd
import numpy as np

# --- Step 1: Load Data ---
# IMPORTANT: Replace "ConditionalFormatData" with the actual name of your data source
df = xl("ConditionalFormatData[#All]", headers=True)

# Ensure numerical columns are numeric, coercing errors
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')


# --- Step 2: Load Parameter from Excel Cell ---
# IMPORTANT: Replace "A1" with your actual parameter cell reference
try:
    sales_threshold = xl("A1") # Load cell A1 as scalar value

    # Convert loaded parameter to appropriate type
    try:
        sales_threshold = float(sales_threshold)
        if pd.isna(sales_threshold): raise ValueError("Is NaN") # Treat NaN float as error
    except (ValueError, TypeError):
        sales_threshold = df['Amount'].quantile(0.9) if df['Amount'].count() > 0 else 0.0 # Default to 90th percentile or 0 if loading fails
        print(f"Warning: Cell A1 (High Sales Threshold) not a valid number. Defaulting to 90th percentile ({sales_threshold:,.2f}) or 0.")

except Exception as e:
    print(f"Error loading threshold from Excel cell: {e}. Please ensure cell reference is correct.")
    # Set a default threshold if loading fails completely
    sales_threshold = df['Amount'].quantile(0.9) if df['Amount'].count() > 0 else 0.0
    print(f"Defaulting threshold due to error: {sales_threshold:,.2f}")


# --- Step 3: Create Flag Column based on Threshold ---
# Create a new column 'Flag_HighSales'
# It will be True if Amount is NOT NaN AND Amount >= sales_threshold
# It will be False otherwise (including if Amount is NaN)
df['Flag_HighSales'] = df['Amount'].notna() & (df['Amount'] >= sales_threshold)


# --- Output Results ---
# Return the original data with the new flag column, and the threshold used
output = {
    'Data with High Sales Flag Head': df.head(), # Show head of the data
    'Data with High Sales Flag Info': df.info(verbose=True, memory_usage=False, buf=None), # Print info
    'High Sales Flag Counts': df['Flag_HighSales'].value_counts(dropna=False).reset_index().rename(columns={'index': 'Flag Value', 'Flag_HighSales': 'Count'}), # Count True/False/NaN
    'Threshold Used': pd.DataFrame({'Parameter': ['High Sales Threshold'], 'Value': [sales_threshold]}),
    'Full Data with High Sales Flag': df # Return the full DataFrame
}

output # Output the dictionary
```

**Explanation:**

*   We load the raw sales data (`df`) using `xl()`.
*   In **Step 2**, we use `xl("A1")` to read the value from your threshold cell (`A1`). Similar to the parameterization example, we include error handling and provide a default threshold (the 90th percentile of the existing data) if the cell value is missing or not a valid number.
*   In **Step 3**, we create a new column named `Flag_HighSales`. This column contains boolean values (`True` or `False`). `df['Amount'] >= sales_threshold` performs the comparison. `df['Amount'].notna()` ensures we only flag non-missing amounts. The `&` operator combines these conditions: a row is flagged as `True` only if the `Amount` is a valid number *and* is greater than or equal to the `sales_threshold`. Otherwise, the flag is `False`.
*   The final dictionary output includes the head and full version of the DataFrame with the new `Flag_HighSales` column, a count of True/False values in the flag column, and the threshold that was actually used.

**Viewing the Output:**

*   Click the Python cell containing this code, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) on the cell. This will spill a representation of the dictionary.
*   Extract the `Full Data with High Sales Flag` DataFrame using `=PY(CellRef["Full Data with High Sales Flag"])` and convert that cell to 'Excel Value' to see the entire dataset including the `Flag_HighSales` column.

**Applying Conditional Formatting in Excel:**

Once you have the `Full Data with High Sales Flag` spilled into your Excel sheet (let's say it's in columns D through H), you can apply conditional formatting:

1.  **Select the column(s)** you want to format (e.g., select column G which contains the 'Amount').
2.  Go to the **Home** tab in Excel, click **Conditional Formatting**, then **New Rule...**.
3.  Choose **"Use a formula to determine which cells to format"**.
4.  Enter a formula that checks the corresponding cell in the `Flag_HighSales` column. If your data starts in row 2, and the `Flag_HighSales` column is column H, the formula would be `=H2=TRUE`. Make sure to use absolute reference (`$H2`) if you want the rule to apply based on the flag column, but only apply the formatting to the selected column (G). If you want to highlight the *entire row* where the flag is TRUE, select the entire range (e.g., D2:H1000) and the formula would be `=$H2=TRUE`.
5.  Click the **Format...** button and choose your desired formatting (e.g., fill color, font color).
6.  Click **OK** twice.

Now, any row where the `Flag_HighSales` column is `TRUE` (meaning the Amount met your parameterized threshold) will have the specified conditional formatting applied in Excel.

**To update the formatting:** Change the threshold value in your Excel cell (`A1`) and **re-run the Python cell** (Ctrl+Enter). The `Flag_HighSales` column in your Python output will update, and because your Excel conditional formatting rule points to this column, the formatting on your data will also update automatically.

This integration of Python logic with Excel's formatting features creates dynamic, visually rich reports that automatically update when parameters change.

**Further Analysis:**

Here are some advanced conditional formatting techniques you could apply to this dataset:

1. **Complex Formatting Rules:**
   - Implement multi-condition formatting logic
   - Create dynamic threshold calculations
   - Design hierarchical formatting rules

2. **Statistical Formatting:**
   - Highlight statistical outliers
   - Format based on z-scores or percentiles
   - Implement moving average thresholds

3. **Time-based Formatting:**
   - Create rolling window comparisons
   - Highlight seasonal patterns
   - Format trend-based deviations

4. **Comparative Formatting:**
   - Implement peer group comparisons
   - Create benchmark-based formatting
   - Design year-over-year change indicators

5. **Advanced Visualization:**
   - Create custom color scales
   - Design icon sets based on metrics
   - Implement data bars with custom logic

The next topic in the series is [Reporting & Automation - User-Defined Functions (UDFs)](./07-Reporting%20%26%20Automation_05-User-Defined%20Functions%20(UDFs).md), which will show you how to create custom Excel functions using Python to extend Excel's capabilities with your own calculations and logic.