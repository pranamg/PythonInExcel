**Reporting & Automation - 3. Parameterization**

Parameterization enables dynamic control of Python code through Excel cell values. By reading parameters from Excel cells, you can filter data, customize plot titles, set thresholds, and more without modifying the Python code itself. When cell values change and the code is re-run, outputs update automatically, creating flexible and interactive reports.

Based on [`piplist.txt`](./README.md) output, you should have `pandas` and the `xl()` function to create parameterized analyses that respond to user inputs directly from Excel cells.

**Step 1: Generate Sample Data for Parameterization**

We'll reuse a simple sales dataset structure again, similar to the summary and reporting examples, as it provides good dimensions (Date, Region, Category, Amount) to parameterize.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy sales data for Parameterization
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta

fake = Faker()

num_records = 2500
start_date = date(2022, 1, 1)
end_date = date(2024, 6, 30) # Data period
dates = pd.date_range(start=start_date, end=end_date, freq='D') # Use periods to ensure correct number

regions = ['North', 'South', 'East', 'West', 'Central']
product_categories = ['Gadgets', 'Apparel', 'Home Appliances', 'Software', 'Accessories']

data = []
for i in range(num_records):
    transaction_date = random.choice(dates) # Pick random dates from the range
    region = random.choice(regions)
    category = random.choice(product_categories)
    amount = round(random.uniform(25, 7500), 2) # Wider range for amounts
    quantity = random.randint(1, 8)

    data.append([transaction_date, region, category, amount, quantity])

df_param_data = pd.DataFrame(data, columns=['TransactionDate', 'Region', 'ProductCategory', 'Amount', 'Quantity'])

# Add some missing values
for col in ['Amount', 'Region', 'ProductCategory']:
    missing_indices = random.sample(range(num_records), int(num_records * random.uniform(0.01, 0.04))) # 1-4% missing
    df_param_data.loc[missing_indices, col] = np.nan

# Ensure Date is datetime
df_param_data['TransactionDate'] = pd.to_datetime(df_param_data['TransactionDate'])

# Shuffle rows
df_param_data = df_param_data.sample(frac=1, random_state=42).reset_index(drop=True)

df_param_data # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_param_data` simulating sales transactions.
*   It includes Date, Region, Product Category, Amount, and Quantity.
*   Missing values are included.
*   The result, `df_param_data`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `ParameterizedSalesData`.

**Step 2: Create Excel Cells for Parameters and Write Parameterized Python Code**

Now, before running the Python code, you need to set up the Excel cells that will act as your parameters.

1.  **Choose some empty cells** in your Excel sheet (e.g., cells `A1`, `A2`, `A3`).
2.  **Enter labels** next to them for clarity (e.g., 'Target Region' in `B1`, 'Min Revenue' in `B2`, 'Start Month (YYYY-MM)' in `B3`).
3.  **Enter your desired parameter values** in the chosen cells (`A1`, `A2`, `A3`). For example:
    *   In `A1`, type `North`
    *   In `A2`, type `500`
    *   In `A3`, type `2023-07` (This will represent July 2023)

Now, in a **new** Excel cell, enter `=PY` and paste the following code. This code will load the data and the values from the specific Excel cells you just created, and use them to filter and summarize the data.

**IMPORTANT:** Replace `"ParameterizedSalesData"` with the actual name of your Excel range/Table, and replace `"A1"`, `"A2"`, `"A3"` with the actual cell references you used for your parameters. Press **Ctrl+Enter**.

```python
# Analyze sales data based on parameters from Excel cells
import pandas as pd
import numpy as np
from datetime import date, timedelta # Import timedelta if needed

# --- Step 1: Load Data ---
# IMPORTANT: Replace "ParameterizedSalesData" with the actual name of your data source
df = xl("ParameterizedSalesData[#All]", headers=True)

# Ensure appropriate data types
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['Region'] = df['Region'].astype(str).replace('nan', np.nan)
df['ProductCategory'] = df['ProductCategory'].astype(str).replace('nan', np.nan)


# --- Step 2: Load Parameters from Excel Cells ---
# IMPORTANT: Replace "A1", "A2", "A3" with your actual parameter cell references
try:
    target_region = xl("A1") # Load cell A1 as scalar value
    min_revenue = xl("A2")  # Load cell A2 as scalar value
    start_month_str = xl("A3") # Load cell A3 as scalar value

    # Convert loaded parameters to appropriate types if necessary
    # Handle potential errors if cell is empty or wrong type
    if not isinstance(target_region, str) or target_region == '':
         target_region = 'All Regions' # Default if not provided
         print(f"Warning: Cell A1 (Target Region) not found or empty. Defaulting to '{target_region}'.")

    try:
        min_revenue = float(min_revenue)
        if pd.isna(min_revenue): raise ValueError("Is NaN") # Treat NaN float as error
    except (ValueError, TypeError):
        min_revenue = 0.0 # Default if not a valid number
        print(f"Warning: Cell A2 (Min Revenue) not a valid number. Defaulting to {min_revenue}.")

    try:
        # Attempt to parse YYYY-MM into a datetime object
        start_date_param = pd.to_datetime(start_month_str + '-01', errors='coerce')
        if pd.isna(start_date_param): raise ValueError("Parsing failed")
    except (ValueError, TypeError):
        # Default to the earliest date in the data if parsing fails
        start_date_param = df['TransactionDate'].min()
        print(f"Warning: Cell A3 (Start Month YYYY-MM) not valid or parseable. Defaulting to earliest data date ({start_date_param.date()}).")


except Exception as e:
    print(f"Error loading parameters from Excel cells: {e}. Please ensure cell references are correct.")
    # Set default parameters if loading fails completely
    target_region = 'All Regions'
    min_revenue = 0.0
    start_date_param = df['TransactionDate'].min() if not df['TransactionDate'].empty else pd.to_datetime('1900-01-01') # Handle empty df case


# --- Step 3: Use Parameters to Filter Data ---
df_filtered = df.copy()

# Filter by Region (if not 'All Regions')
if target_region != 'All Regions':
    # Ensure comparison handles potential NaNs in Region column by checking notna() first
    df_filtered = df_filtered[df_filtered['Region'].notna() & (df_filtered['Region'] == target_region)]

# Filter by Minimum Revenue
df_filtered = df_filtered[df_filtered['Amount'].notna() & (df_filtered['Amount'] >= min_revenue)]

# Filter by Start Date
if pd.notna(start_date_param):
    df_filtered = df_filtered[df_filtered['TransactionDate'].notna() & (df_filtered['TransactionDate'] >= start_date_param)]


# --- Step 4: Generate Summary Based on Filtered Data ---
# Example: Total Revenue by Product Category for the filtered data
# Drop rows with missing Category/Amount in the filtered data before grouping
category_summary_filtered = df_filtered.dropna(subset=['ProductCategory', 'Amount']).groupby('ProductCategory')['Amount'].sum().reset_index()
category_summary_filtered = category_summary_filtered.rename(columns={'Amount': 'Total_Revenue_Filtered'})
# Sort for readability
category_summary_filtered = category_summary_filtered.sort_values('Total_Revenue_Filtered', ascending=False)

# Add a title reflecting the parameters
report_title = f"Total Revenue by Category for {target_region} (Min Revenue >= ${min_revenue:,.0f}) since {start_date_param.strftime('%Y-%m')}"


# --- Output Results ---
# Return a dictionary containing the filtered summary and parameter info
output = {
    'Report Title': pd.DataFrame({'Title': [report_title]}), # Return title as DataFrame for display
    'Filtered Sales Summary by Category': category_summary_filtered,
    'Parameters Used': pd.DataFrame({
        'Parameter': ['Target Region', 'Min Revenue', 'Start Month (YYYY-MM)'],
        'Value': [target_region, min_revenue, start_month_str],
        'Loaded Value (Internal)': [target_region, min_revenue, start_date_param] # Show internal loaded value for verification
    })
}

output # Output the dictionary
```

**Explanation:**

*   We load the main sales data (`df`) using `xl()`.
*   In **Step 2**, we use `xl("CellReference").scalar` to read the values from your specified Excel cells (`A1`, `A2`, `A3`). We wrap this in a `try...except` block and add checks to handle potential errors if the cells are empty, missing, or contain unexpected values (like non-numeric text in a number cell), providing default values and printing warnings.
    *   `xl("A1").scalar` gets the text from `A1` for the region.
    *   `xl("A2").scalar` gets the value from `A2` and we attempt to convert it to a float.
    *   `xl("A3").scalar` gets the text from `A3` (e.g., '2023-07') and we attempt to parse it into a datetime object using `pd.to_datetime`.
*   In **Step 3**, we use these loaded parameter variables (`target_region`, `min_revenue`, `start_date_param`) to filter the `df` DataFrame using standard pandas boolean indexing. We include `.notna()` checks to ensure filtering works correctly even if the data contains missing values in the columns being filtered.
*   In **Step 4**, we perform an aggregation (sum total revenue by product category) on the *filtered* DataFrame (`df_filtered`).
*   We also create a dynamic `report_title` string that includes the actual parameters used, demonstrating how you can incorporate parameters into text outputs.
*   The final dictionary output includes the filtered summary table and a table showing the parameters that were loaded from Excel.

**Viewing the Output:**

*   Click the Python cell containing this code, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) on the cell. This will spill a representation of the dictionary.
*   Now, extract the individual DataFrames using `=PY(CellRef["Key"])` and convert *those* cells to 'Excel Value' to see the title, filtered summary table, and the parameters table.
*   **To see the parameterization in action:** Change the values in your parameter cells (`A1`, `A2`, `A3`). For example, change `A1` to `South`, `A2` to `1000`, and `A3` to `2024-01`. Then, **re-run the Python cell** (click the cell and press Ctrl+Enter). The output table and title should update automatically based on the new filter criteria from your Excel cells.

These techniques demonstrate how to create dynamic, parameterized analyses and reports that respond to user inputs directly within Excel using Python.

**Further Analysis:**

Here are some advanced parameterization techniques you could apply to this dataset:

1. **Advanced Parameter Types:**
   - Implement multi-select parameters using delimited strings
   - Create date range selectors with validation
   - Design cascading parameter dependencies

2. **Parameter-Driven Analytics:**
   - Create dynamic aggregation level selection
   - Implement flexible metric calculations
   - Design user-defined grouping logic

3. **Visual Parameter Controls:**
   - Build dropdown lists for parameter selection
   - Create parameter validation rules
   - Implement dynamic parameter ranges

4. **Complex Parameter Logic:**
   - Design parameter-based filtering rules
   - Create parameter-driven calculations
   - Implement business logic conditions

5. **Parameter Management:**
   - Create parameter documentation systems
   - Implement parameter version control
   - Design parameter dependency tracking

The next topic in the series is [Reporting & Automation - Conditional Formatting](./07-Reporting%20%26%20Automation_04-Conditional%20Formatting.md), which shows how Python can work with Excel's built-in formatting features to create visually informative reports. While Python generates the data and determines formatting rules, Excel's native formatting capabilities handle the visual presentation.