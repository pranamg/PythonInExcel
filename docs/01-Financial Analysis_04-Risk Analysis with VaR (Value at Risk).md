**Financial Analysis - Risk analysis with VaR (Value at Risk) using `scipy.stats`**.

Value at Risk (VaR) is a widely used metric to quantify the potential loss that could occur over a defined period for a given confidence level. For example, a 99% daily VaR of $1000 means that there is a 1% chance the portfolio could lose more than $1000 in a single day under normal market conditions.

Based on [`piplist.txt`](./README.md) output, you should have `scipy` (which includes `scipy.stats`), `pandas`, and `numpy`, so we can perform this calculation.

**Step 1: Generate Sample Daily Returns Data**

VaR is typically calculated on returns data. We'll generate dummy daily returns for a hypothetical portfolio.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy daily returns data for VaR calculation
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Define date range
start_date = date(2022, 1, 1)
end_date = date(2024, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='B') # 'B' frequency for business days

# Generate random daily returns (simulating portfolio returns)
# Mean return around 0.05% per day, standard deviation 1% per day
daily_returns = np.random.normal(loc=0.0005, scale=0.01, size=len(dates))

# Create DataFrame
df_returns = pd.DataFrame(daily_returns, index=dates, columns=['Portfolio_Daily_Return'])

df_returns # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_returns` with dummy daily percentage returns for a single portfolio over a few years.
*   The returns are simulated using a normal distribution with a small positive mean and a standard deviation typical for daily stock returns.
*   The index uses 'B' frequency to simulate business days.
*   The result will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `PortfolioReturns`.

**Step 2: Calculate Value at Risk (VaR)**

We will calculate VaR using two common methods:

1.  **Historical VaR:** This uses the percentile of the historical return distribution. Simple and doesn't assume a distribution shape.
2.  **Parametric VaR:** This assumes the returns follow a normal distribution and uses the mean and standard deviation with a Z-score from the normal distribution (`scipy.stats.norm.ppf`).

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"PortfolioReturns"` with the actual name of the Excel range/Table where your dummy returns data is. Press **Ctrl+Enter**.

```python
# Calculate Historical and Parametric VaR
import pandas as pd
import numpy as np
import scipy.stats as stats # For normal distribution percentile function

# Load the daily returns data from Excel
# IMPORTANT: Replace "PortfolioReturns" with the actual name of your Excel range or Table
df_returns_raw = xl("PortfolioReturns[#All]", headers=True)

# --- Data Preparation ---

# Make a copy and ensure the returns column is numeric
df_returns = df_returns_raw.copy()

# Assuming the returns are in the second column (index is first)
if len(df_returns.columns) < 2:
     raise ValueError("DataFrame should contain at least two columns: Date and Portfolio Return.")

# Identify the returns column name (assume it's the second column name from the generated data)
returns_col_name = df_returns.columns[1]

# Convert the returns column to numeric, coercing errors
df_returns[returns_col_name] = pd.to_numeric(df_returns[returns_col_name], errors='coerce')

# Drop any rows with NaN in the returns column after conversion
df_returns.dropna(subset=[returns_col_name], inplace=True)

# Check if any data remains
if df_returns.empty:
     raise ValueError("No valid numeric data found in the returns column after cleaning.")

# Extract the numeric returns series
portfolio_returns = df_returns[returns_col_name]


# --- VaR Calculation ---

confidence_level = 0.99 # e.g., 99% confidence
alpha = 1 - confidence_level # Significance level (e.g., 0.01 for 99%)

# 1. Historical VaR
# Calculate the percentile corresponding to the alpha level
# The alpha-th percentile of returns is the value such that alpha percent of returns fall below it.
# For VaR, we look at the loss side, so we find the 1-confidence_level percentile (e.g., 1st percentile for 99% VaR).
# A 1% percentile return of -0.02 means 1% of days lost 2% or more. VaR is reported as a positive loss.
historical_var_percentile = portfolio_returns.quantile(alpha)
historical_var = -historical_var_percentile # VaR is typically reported as a positive value

# 2. Parametric VaR (assuming Normal Distribution)
# Calculate the mean and standard deviation of the returns
mean_return = portfolio_returns.mean()
std_dev_return = portfolio_returns.std()

# Find the Z-score for the given confidence level using the Percent Point Function (inverse of CDF)
# For 99% confidence, we want the Z-score such that 1% of the area is in the left tail.
z_score = stats.norm.ppf(alpha, mean_return, std_dev_return)

# Parametric VaR calculation
parametric_var = -z_score # VaR is typically reported as a positive value

# --- Output ---

# Return the calculated VaR values in a dictionary
output = {
    'Confidence_Level': confidence_level,
    'Historical_VaR_Daily': historical_var,
    'Parametric_VaR_Daily_Normal_Assumption': parametric_var
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy daily returns data. **Remember to replace `"PortfolioReturns"`**.
*   We extract the numeric daily returns column.
*   We define the `confidence_level` (e.g., 0.99 for 99%).
*   **Historical VaR:** We calculate the value at the `alpha` (1 - confidence level) percentile of the historical returns data using `quantile()`. We negate the result because VaR is reported as a positive loss.
*   **Parametric VaR:** We calculate the mean and standard deviation of the returns. We then use `scipy.stats.norm.ppf(alpha, mean, std_dev)` to find the value (the Z-score scaled by mean and std dev) below which `alpha` percentage of returns are expected to fall, assuming a normal distribution. We negate this value for the VaR result.
*   We return a dictionary containing the confidence level and the calculated VaR values from both methods.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the dictionary. The individual key-value pairs will spill into adjacent cells, showing the calculated VaR figures.

**Further Analysis:**

Here are some advanced risk analysis techniques you could explore:

1. **Advanced VaR Methods:**
   - Implement Expected Shortfall (CVaR)
   - Add Monte Carlo VaR simulation
   - Create stressed VaR calculations

2. **Risk Decomposition:**
   - Implement component VaR analysis
   - Add factor-based risk decomposition
   - Create risk attribution analysis

3. **Scenario Analysis:**
   - Add historical scenario testing
   - Implement stress testing
   - Create sensitivity analysis tools

4. **Tail Risk Analysis:**
   - Calculate extreme value statistics
   - Implement copula-based dependency modeling
   - Create tail dependence analysis

5. **Dynamic Risk Measures:**
   - Add dynamic volatility models (GARCH)
   - Implement regime-switching models
   - Create time-varying correlation analysis

The next topic in the series is [Financial Analysis - Monte Carlo Simulations](./01-Financial%20Analysis_05-Monte%20Carlo%20Simulations.md), which explores using simulation techniques for financial projections using `numpy`.