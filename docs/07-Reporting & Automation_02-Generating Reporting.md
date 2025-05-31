**Reporting & Automation - 2. Generating Reports**

This guide demonstrates how to create comprehensive reports by combining multiple analyses, visualizations, and summaries into a cohesive output. Based on [`piplist.txt`](./README.md) output, you should have standard Python libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`) excel at analysis and component generation, the focus here is on organizing these elements effectively within Excel.

The reporting process involves three key steps:
1. Creating multiple analyses (summaries, calculations, visualizations) in a single code block
2. Organizing all results (DataFrames, figures, metrics) in a structured dictionary
3. Presenting components in Excel cells to create a clean, professional report layout

This approach maximizes Python's analytical capabilities while leveraging Excel's presentation features.

**Step 1: Generate Sample Data for Reporting**

We'll create a dummy sales dataset suitable for generating a performance report by different dimensions (time, region, product).

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy sales data for Reporting
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta

fake = Faker()

num_records = 3000
start_date = date(2022, 1, 1)
end_date = date(2024, 6, 15) # Data period covering a few years and recent date
dates = pd.date_range(start=start_date, end=end_date, freq='D')

regions = ['North', 'South', 'East', 'West', 'Central']
products = ['Laptop', 'Keyboard', 'Mouse', 'Monitor', 'Webcam', 'Headphones', 'Printer']

data = []
for i in range(num_records):
    transaction_date = random.choice(dates) # Pick random dates
    region = random.choice(regions)
    product = random.choice(products)
    quantity = random.randint(1, 5)
    price_per_item = round(random.uniform(20, 1500), 2)
    revenue = quantity * price_per_item

    data.append([transaction_date, region, product, quantity, revenue])

df_report_data = pd.DataFrame(data, columns=['Date', 'Region', 'Product', 'Quantity', 'Revenue'])

# Add some missing values
for col in ['Revenue', 'Region']:
    missing_indices = random.sample(range(num_records), int(num_records * random.uniform(0.02, 0.04))) # 2-4% missing
    df_report_data.loc[missing_indices, col] = np.nan

# Add a few outliers in Revenue
outlier_indices = random.sample(range(num_records), 5)
df_report_data.loc[outlier_indices, 'Revenue'] = df_report_data['Revenue'] * random.uniform(5, 10) # High outliers

# Ensure Date is datetime
df_report_data['Date'] = pd.to_datetime(df_report_data['Date'])


# Shuffle rows
df_report_data = df_report_data.sample(frac=1, random_state=42).reset_index(drop=True)

df_report_data # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_report_data` with simulated sales transactions including Date, Region, Product, Quantity, and Revenue.
*   It includes different categorical variables and numerical values suitable for aggregation and reporting.
*   Missing values and outliers are introduced.
*   The result, `df_report_data`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `ReportData`.

**Step 2: Generate Report Components (Summaries and Plots) within a Single Code Block**

Now, we'll load this dummy data, perform several aggregations and create plots, and collect them all into a single output dictionary to mimic a report.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"ReportData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Generate report components (summaries and plots)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta # Import timedelta

# Load the data from Excel
# IMPORTANT: Replace "ReportData" with the actual name of your Excel range or Table
df = xl("ReportData[#All]", headers=True)

# Ensure appropriate data types
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Region'] = df['Region'].astype(str).replace('nan', np.nan)
df['Product'] = df['Product'].astype(str).replace('nan', np.nan)


# --- Apply Custom Style Guidelines ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False
sns.set_theme(style="whitegrid")

# Define color palette from guidelines
plot_colors = ['#ffe600', '#188ce5', '#2db757', '#ff6d00', '#750e5c', '#ff4136', '#27acaa', '#1a1a24']


# --- Report Component 1: Key Performance Indicators (KPIs) ---
# Drop rows with missing Revenue for sum/mean, drop rows with missing Quantity for sum
total_revenue = df['Revenue'].dropna().sum()
average_revenue_per_transaction = df['Revenue'].dropna().mean()
total_quantity_sold = df['Quantity'].dropna().sum()
num_transactions = df['Revenue'].dropna().count() # Count non-missing revenue as transactions

kpi_summary = pd.DataFrame({
    'Metric': ['Total Revenue', 'Avg Revenue per Transaction', 'Total Quantity Sold', 'Number of Transactions'],
    'Value': [total_revenue, average_revenue_per_transaction, total_quantity_sold, num_transactions]
})

# Format currency and numbers for KPIs
kpi_summary['Value'] = kpi_summary['Value'].apply(lambda x: f'${x:,.2f}' if 'Revenue' in str(x) else f'{x:,.0f}') # Apply formatting based on metric name


# --- Report Component 2: Revenue by Region Table ---
# Group by Region and sum Revenue, drop rows with missing Region/Revenue before grouping
revenue_by_region_table = df.dropna(subset=['Region', 'Revenue']).groupby('Region')['Revenue'].sum().reset_index()
revenue_by_region_table = revenue_by_region_table.rename(columns={'Revenue': 'Total Revenue'})
revenue_by_region_table = revenue_by_region_table.sort_values('Total Revenue', ascending=False)
# Format currency
revenue_by_region_table['Total Revenue'] = revenue_by_region_table['Total Revenue'].apply(lambda x: f'${x:,.2f}')


# --- Report Component 3: Monthly Revenue Trend Table ---
# Drop rows with missing Date or Revenue before aggregation
monthly_revenue_table = df.dropna(subset=['Date', 'Revenue']).set_index('Date').resample('ME')['Revenue'].sum().reset_index()
monthly_revenue_table = monthly_revenue_table.rename(columns={'Revenue': 'Total_Revenue'})
# Rename the Date column to Month for clarity in the output table
monthly_revenue_table = monthly_revenue_table.rename(columns={'Date': 'Month'})
# Format Month column to YYYY-MM string
monthly_revenue_table['Month'] = monthly_revenue_table['Month'].dt.strftime('%Y-%m')
# Format currency
monthly_revenue_table['Total_Revenue'] = monthly_revenue_table['Total_Revenue'].apply(lambda x: f'${x:,.2f}')


# --- Report Component 4: Revenue by Region Bar Chart ---
# Use the aggregated data from Component 2
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x='Total Revenue', y='Region', hue='Region', legend=False, data=revenue_by_region_table.copy(), ax=ax1, palette=plot_colors[:len(revenue_by_region_table)], orient='h') # Use a copy to avoid modifying original for string formatting

ax1.set_title('Total Revenue by Region', fontsize=14, color='#1a1a24')
ax1.set_xlabel('Total Revenue', fontsize=12, color='#1a1a24')
ax1.set_ylabel('Region', fontsize=12, color='#1a1a24')
sns.despine(ax=ax1, top=True, right=True)
ax1.grid(False)

# Add data labels (Need to use original numeric data for labels before formatting)
# Re-calculate total revenue by region without string formatting for label values
revenue_by_region_numeric = df.dropna(subset=['Region', 'Revenue']).groupby('Region')['Revenue'].sum().reset_index()
revenue_by_region_numeric = revenue_by_region_numeric.rename(columns={'Revenue': 'Total Revenue'})
revenue_by_region_numeric = revenue_by_region_numeric.sort_values('Total Revenue', ascending=False)


for index, row in revenue_by_region_numeric.iterrows():
    # Format negative numbers as (1.0) (unlikely for revenue sum, but good practice)
    label_text = f'({abs(row["Total Revenue"]):,.0f})' if row["Total Revenue"] < 0 else f'{row["Total Revenue"]:,.0f}'
    ax1.text(row['Total Revenue'], index, f' ${label_text}', color='#1a1a24', va='center') # Add dollar sign

plt.tight_layout()


# --- Report Component 5: Monthly Revenue Trend Line Plot ---
# Use the aggregated data from Component 3, but need the original numeric values for plotting
# Re-calculate monthly revenue trend without string formatting for plotting
monthly_revenue_numeric = df.dropna(subset=['Date', 'Revenue']).set_index('Date').resample('ME')['Revenue'].sum()

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(monthly_revenue_numeric.index, monthly_revenue_numeric, marker='o', linestyle='-', color='#188ce5') # Blue

ax2.set_title('Monthly Revenue Trend', fontsize=14, color='#1a1a24')
ax2.set_xlabel('Date', fontsize=12, color='#1a1a24')
ax2.set_ylabel('Total Revenue', fontsize=12, color='#1a1a24')
sns.despine(ax=ax2, top=True, right=True)
ax2.grid(False)
fig2.autofmt_xdate()


plt.tight_layout()


# --- Output Results ---
# Collect all report components into a single dictionary
output = {
    'Report Title Note': pd.DataFrame({'Note': ["Sales Performance Report"]}),
    'Key Performance Indicators (KPIs)': kpi_summary,
    'Revenue by Region (Table)': revenue_by_region_table,
    'Monthly Revenue Trend (Table)': monthly_revenue_table,
    'Revenue by Region (Plot)': fig1,
    'Monthly Revenue Trend (Plot)': fig2,
    # Add a note about manual assembly in Excel
    'Assembly Note': pd.DataFrame({'Note': ["Please extract tables and plots from this output dictionary using '=PY(CellRef[\"Key\"])' and position them in your Excel sheet to create the report layout."]}),
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy sales data. **Remember to replace `"ReportData"`**.
*   We ensure columns have appropriate data types.
*   **Style Guidelines:** Matplotlib and Seaborn styles are set. A color palette is defined.
*   **KPIs:** We calculate overall metrics like total revenue, average transaction value, total quantity, and number of transactions. These are placed in a summary DataFrame and formatted for display.
*   **Revenue by Region Table:** We group the data by `Region` and sum the `Revenue`, creating a table sorted by total revenue. Revenue is formatted as currency strings.
*   **Monthly Revenue Trend Table:** We resample the time-indexed data by month and sum the `Revenue`, creating a table showing revenue per month. Date is formatted to 'YYYY-MM' string, and revenue is formatted as currency.
*   **Revenue by Region Bar Chart:** A horizontal bar chart is created using `seaborn.barplot()` showing total revenue per region. Data labels are added to the bars, formatted as currency.
*   **Monthly Revenue Trend Line Plot:** A line plot is created using `matplotlib.pyplot.plot()` showing the monthly total revenue over time.
*   **Output Dictionary:** All the generated DataFrames (KPIs, summary tables) and Matplotlib figure objects (plots) are collected into a single Python dictionary. We also include simple DataFrames with text notes, including instructions on how to use the output and acknowledging the need for manual assembly in Excel.
*   The dictionary is the final output of the Python cell.

**Viewing the Output:**

*   Click the Python cell containing this code, then click the Python icon/button next to the formula bar.
*   You will see a single Python object output, which is the dictionary.
*   To get the individual components into your Excel sheet, select "Excel Value" (**Ctrl+Shift+Alt+M**) on the cell. This will spill a representation of the dictionary.
*   Now, in *other* Excel cells, you can reference the cell containing the dictionary output and the key for the component you want. For example, if your dictionary output is in cell `A1`:
    *   In `A3`: `=PY(A1["Key Performance Indicators (KPIs)"])` -> Convert A3 to Excel Value (**Ctrl+Shift+Alt+M**) to see the KPI table.
    *   In `A10`: `=PY(A1["Revenue by Region (Plot)"])` -> Convert A10 to Picture in Cell ("Create Reference") to see the bar chart.
    *   Repeat for other keys ('Revenue by Region (Table)', 'Monthly Revenue Trend (Table)', 'Monthly Revenue Trend (Plot)', etc.) to extract all components into your worksheet.

These components can be arranged within your Excel sheet to create a professional sales performance report layout. This approach combines Python's analytical power with Excel's familiar grid system for presentation.

**Further Analysis:**

Here are some advanced reporting techniques you could apply to this dataset:

1. **Advanced Report Components:**
   - Create interactive dashboards with drill-down capabilities
   - Implement dynamic data filtering options
   - Add conditional formatting based on KPI thresholds

2. **Enhanced Visualizations:**
   - Create small multiples for trend comparison
   - Implement combination charts (e.g., bar + line)
   - Add dynamic annotations for key insights

3. **Automated Report Generation:**
   - Create report templates with placeholders
   - Implement scheduled report generation
   - Add email distribution capabilities

4. **Advanced Analytics Integration:**
   - Include predictive analytics results
   - Add statistical significance indicators
   - Implement what-if scenario analysis

5. **Custom Report Layouts:**
   - Create multi-page report structures
   - Implement custom branding elements
   - Design mobile-friendly report layouts

The next topic in the series is [Reporting & Automation - Parameterization](./07-Reporting%20%26%20Automation_03-Parameterization.md), which will teach you how to make your reports dynamic and reusable by accepting user inputs and customization options.