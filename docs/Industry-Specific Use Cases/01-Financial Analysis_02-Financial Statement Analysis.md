The next topic in the financial analysis series is **Financial Analysis - 2. Financial Statement Analysis**.

Financial statement analysis is a fundamental technique that examines a company's financial statements (like Income Statement, Balance Sheet) to evaluate its performance, financial health, and future prospects. This analysis typically includes calculating key ratios, analyzing trends, and benchmarking against industry standards.

Based on [`piplist.txt`](./README.md) output, you should have `pandas` for data manipulation, `numpy` for calculations, `statsmodels` for potential time series analysis (like trend analysis), and `seaborn`/`matplotlib`/`plotly` for visualization. This is well-supported.

**Step 1: Generate Sample Financial Statement Data**

We'll generate dummy annual financial data for a couple of fictional companies over several years.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy annual financial statement data
import pandas as pd
import numpy as np
from faker import Faker # Assuming Faker is available and can be used like this in Excel env

fake = Faker()

num_companies = 2
num_years = 7
start_year = 2018

data = []

for _ in range(num_companies):
    company_name = fake.company()
    # Ensure values are somewhat plausible for financial statements over time
    base_revenue = np.random.randint(100_000, 500_000) # Starting revenue
    base_assets = np.random.randint(50_000, 300_000) # Starting assets

    for year_offset in range(num_years):
        year = start_year + year_offset
        # Simulate growth/fluctuation year-over-year
        revenue = base_revenue * (1 + np.random.uniform(0.01, 0.15) * year_offset + np.random.uniform(-0.03, 0.05))
        cogs = revenue * np.random.uniform(0.4, 0.6) # Cost of Goods Sold
        gross_profit = revenue - cogs
        op_expenses = gross_profit * np.random.uniform(0.2, 0.35) # Operating Expenses
        # Simple tax assumption
        earnings_before_tax = gross_profit - op_expenses
        tax_rate = 0.25
        net_income = earnings_before_tax * (1 - tax_rate)
        # Ensure Net Income is not negative in dummy data for simplicity, or handle
        net_income = max(net_income, revenue * 0.01) # Ensure at least a small profit margin

        # Balance Sheet items - simplified
        assets = base_assets * (1 + np.random.uniform(0.02, 0.1) * year_offset + np.random.uniform(-0.02, 0.03))
        # Liabilities + Equity = Assets
        # Simulate liabilities as a portion of assets, with some fluctuation
        liabilities = assets * np.random.uniform(0.3, 0.6)
        equity = assets - liabilities
        # Ensure equity is positive
        equity = max(equity, assets * 0.1) # Ensure some minimum equity

        data.append([company_name, year, revenue, cogs, gross_profit, op_expenses, net_income, assets, liabilities, equity])

df_financials = pd.DataFrame(data, columns=['Company', 'Year', 'Revenue', 'Cost_of_Goods_Sold', 'Gross_Profit', 'Operating_Expenses', 'Net_Income', 'Total_Assets', 'Total_Liabilities', 'Total_Equity'])

# Ensure Year is integer and format currency-like columns
df_financials['Year'] = df_financials['Year'].astype(int)
# Round financial figures for cleaner display
financial_cols = ['Revenue', 'Cost_of_Goods_Sold', 'Gross_Profit', 'Operating_Expenses', 'Net_Income', 'Total_Assets', 'Total_Liabilities', 'Total_Equity']
for col in financial_cols:
    df_financials[col] = df_financials[col].round(2)


df_financials # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_financials` with financial data for multiple companies over several years.
*   It simulates basic income statement and balance sheet line items with some year-over-year growth and random fluctuations.
*   `Faker` is used to generate fake company names.
*   The result, `df_financials`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `FinancialData`.

**Step 2: Calculate Key Financial Ratios and Visualize Trends**

Now, let's calculate some common financial ratios like Gross Profit Margin, Net Profit Margin, and Debt-to-Equity, and visualize the trend of Revenue and Net Income for one company.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"FinancialData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Calculate financial ratios and visualize trends
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the financial data from Excel
# IMPORTANT: Replace "FinancialData" with the actual name of your Excel range or Table
df_financials = xl("FinancialData[#All]", headers=True)

# --- Ratio Calculation ---

# Handle potential division by zero or NaN values
# Adding a small epsilon or checking for zero denominator is good practice
epsilon = 1e-9

df_financials['Gross_Profit_Margin'] = (df_financials['Gross_Profit'] / (df_financials['Revenue'] + epsilon)) * 100
df_financials['Net_Profit_Margin'] = (df_financials['Net_Income'] / (df_financials['Revenue'] + epsilon)) * 100
df_financials['Debt_to_Equity'] = df_financials['Total_Liabilities'] / (df_financials['Total_Equity'] + epsilon)
df_financials['Return_on_Assets'] = (df_financials['Net_Income'] / (df_financials['Total_Assets'] + epsilon)) * 100


# Select relevant columns for output
df_ratios = df_financials[['Company', 'Year', 'Gross_Profit_Margin', 'Net_Profit_Margin', 'Debt_to_Equity', 'Return_on_Assets']]


# --- Trend Visualization ---

# Select one company for visualization (get the first unique company name)
company_to_plot = df_financials['Company'].unique()[0]
df_single_company = df_financials[df_financials['Company'] == company_to_plot].sort_values('Year')

# Set plot style
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid') # Starting point, will apply custom styles

# Apply custom style guidelines
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False # Turn off default grid

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Revenue and Net Income trends
ax.plot(df_single_company['Year'], df_single_company['Revenue'], marker='o', label='Revenue', color='#188ce5') # Blue
ax.plot(df_single_company['Year'], df_single_company['Net_Income'], marker='o', label='Net Income', color='#2db757') # Green

# Add data labels (Revenue and Net Income values)
for i, row in df_single_company.iterrows():
    ax.text(row['Year'], row['Revenue'], f'{row["Revenue"]:.0f}', ha='left', va='bottom', fontsize=9, color='#188ce5')
    ax.text(row['Year'], row['Net_Income'], f'{row["Net_Income"]:.0f}', ha='left', va='bottom', fontsize=9, color='#2db757')


# Formatting
ax.set_title(f'{company_to_plot} Financial Trend (Revenue & Net Income)', fontsize=14, color='#1a1a24')
ax.set_xlabel('Year', fontsize=12, color='#1a1a24')
ax.set_ylabel('Amount', fontsize=12, color='#1a1a24')
ax.legend()
ax.grid(False) # Explicitly turn off grid

# Customize spines
sns.despine(ax=ax, top=True, right=True, left=False, bottom=False) # Remove top and right spines

# Ensure years are treated as categories for plot
ax.set_xticks(df_single_company['Year'])
ax.set_xticklabels(df_single_company['Year'].astype(str))

# Use a concise number format for the Y-axis if values are large
from matplotlib.ticker import FuncFormatter
def millions_formatter(x, pos):
    return f'{x/1000:,.0f}K' if x >= 1000 else f'{x:,.0f}' # Adjust formatter based on scale

ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))


plt.tight_layout()

# Output both the ratios DataFrame and the plot
# Use a dictionary to return multiple outputs
output = {
    'Financial Ratios': df_ratios,
    'Revenue_NetIncome_Trend_Plot': fig
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy financial data. **Remember to replace `"FinancialData"`**.
*   We calculate four common financial ratios using `pandas`. We add a small epsilon to denominators to prevent division by zero errors if any financial metric happens to be exactly zero in the dummy data.
*   We select the calculated ratios along with Company and Year into a new DataFrame `df_ratios`.
*   We select data for the first company found in the dataset to create a trend plot.
*   We use `matplotlib.pyplot` and `seaborn` to create a line plot showing Revenue and Net Income over the years for that company.
*   **Custom Style:** I've applied the specified style guidelines: Arial 11pt font, specific colors (blue for Revenue, green for Net Income, off-black for text/axes), removed top/right spines, turned off the grid, added data labels, and formatted the y-axis.
*   We return a dictionary containing the `df_ratios` DataFrame and the plot figure object (`fig`).

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the `Financial Ratios` DataFrame to spill it into your sheet.
*   For the `Revenue_NetIncome_Trend_Plot` figure object, select "Picture in Cell" > "Create Reference" to see the plot.

**Further Analysis:**

Here are some advanced financial statement analysis techniques you could explore:

1. **Advanced Ratio Analysis:**
   - Implement DuPont analysis decomposition
   - Calculate efficiency ratios and turnover metrics
   - Add working capital analysis

2. **Comparative Analysis:**
   - Add industry benchmark comparisons
   - Implement peer group analytics
   - Create competitive positioning analysis

3. **Time Series Analysis:**
   - Add seasonal decomposition of financials
   - Implement trend analysis with confidence intervals
   - Create forward-looking projections

4. **Quality of Earnings Analysis:**
   - Add accrual ratio calculations
   - Implement earnings quality metrics
   - Create cash flow analysis tools

5. **Risk Assessment:**
   - Add Altman Z-score calculations
   - Implement bankruptcy prediction models
   - Create credit risk assessment metrics

The next topic in the series is [Financial Analysis - Investment Analysis](./01-Financial%20Analysis_03-Investment%20Analysis.md). Additional aspects of Financial Statement Analysis, such as benchmarking and time series forecasting of specific metrics, can provide deeper insights into company performance.