This guide demonstrates how to perform financial analysis using Python in Excel, with examples and code that utilize *only* the libraries available in the `piplist.txt`.

The first category in this series is **Financial Analysis**.

### Financial Analysis - 1. Portfolio Optimization

Portfolio optimization is a critical financial analysis technique that analyzes the performance and risk of a collection of assets (a portfolio) to determine the optimal combination that maximizes return for a given level of risk, or minimizes risk for a given level of return.

Based on [`piplist.txt`](./README.md) output, you should have the necessary libraries like `pandas` for data handling, `numpy` for numerical operations, `matplotlib` and `seaborn` for visualization, and `scipy` which can be used for optimization tasks (though a full optimization routine is quite involved, we can demonstrate key steps like calculating returns, volatility, and covariance).

**Step 1: Generate Sample Data**

We'll create dummy historical daily stock price data for a few assets. We can use `pandas` and `numpy` for this. `Faker` isn't typically used for generating quantitative time series data like stock prices, so we'll rely on `numpy`'s random functions and `pandas` for date indexing.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy daily stock prices for portfolio optimization
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Define date range
start_date = date(2020, 1, 1)
end_date = date(2023, 12, 31)
dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# Define assets
assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Generate random price movements
# Start with a base price and apply daily random percentage changes
base_price = 100
daily_returns = np.random.randn(len(dates), len(assets)) * 0.005 # Simulate daily percentage changes
prices = np.zeros_like(daily_returns)
prices[0, :] = base_price # Set initial price

# Calculate cumulative prices based on daily returns
# This is a simplified simulation; real prices are more complex
for i in range(1, len(dates)):
    prices[i, :] = prices[i-1, :] * (1 + daily_returns[i, :])
    # Ensure prices don't go below 1 (simple floor)
    prices[i, :] = np.maximum(prices[i, :], 1.0)


# Create DataFrame
df_prices = pd.DataFrame(prices, index=pd.to_datetime(dates), columns=assets)

# Select only valid business days (optional but good practice for financial data)
# df_prices = df_prices[df_prices.index.dayofweek < 5]

df_prices # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_prices` where rows are dates and columns are asset tickers.
*   It simulates prices by starting at a base price and applying small random daily percentage changes.
*   The result, `df_prices`, will be spilled into your Excel sheet when you convert the cell output to 'Excel Value'. Let's assume this data is placed in a range or Table named `PortfolioPrices`.

**Step 2: Analyze Portfolio Performance (Calculate Returns, Volatility, Covariance)**

Now, let's calculate the daily returns, average annual returns, annual volatility, and the covariance matrix, which are fundamental inputs for portfolio optimization.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"PortfolioPrices"` with the actual name of the Excel range/Table where your dummy price data is. Press **Ctrl+Enter**.

```python
# Calculate daily returns, average annual returns, annual volatility, and covariance matrix
import pandas as pd
import numpy as np

# Load the price data from Excel
# IMPORTANT: Replace "PortfolioPrices" with the actual name of your Excel range or Table
df_prices = xl("PortfolioPrices[#All]", headers=True)

# Ensure the index is datetime if not already (xl might load dates as strings)
# Assuming the first column is the date/index
if not pd.api.types.is_datetime64_any_dtype(df_prices.index):
     # Assuming the first column is the date, try setting and converting index
     if isinstance(df_prices.iloc[:, 0].dtype, pd.DatetimeTZDtype):
         df_prices = df_prices.set_index(df_prices.columns[0])
     else:
        try:
            df_prices = df_prices.set_index(df_prices.columns[0])
            df_prices.index = pd.to_datetime(df_prices.index)
        except Exception as e:
            # If first column isn't date, assume dates are in the index already or handle as needed
            print(f"Could not automatically set and convert index to datetime. Ensure the first column is a date or handle index manually. Error: {e}")
            # Proceeding without explicit datetime index conversion, may cause issues
            pass


# Calculate daily returns
# Shift prices by 1 to get the previous day's price
daily_returns = df_prices.pct_change().dropna()

# Assume 252 trading days in a year for annualization
trading_days_per_year = 252

# Calculate average annual returns
average_annual_returns = daily_returns.mean() * trading_days_per_year

# Calculate annual volatility (standard deviation)
annual_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)

# Calculate covariance matrix (used for portfolio risk)
covariance_matrix = daily_returns.cov() * trading_days_per_year

# Store results in a dictionary to return multiple dataframes/series
results = {
    'Daily Returns Head': daily_returns.head(),
    'Average Annual Returns': average_annual_returns,
    'Annual Volatility': annual_volatility,
    'Annual Covariance Matrix': covariance_matrix
}

results # Output the dictionary containing results
```

**Explanation:**

*   We load the dummy price data using `xl()`. **Remember to replace `"PortfolioPrices"`**.
*   We calculate daily percentage changes (`pct_change()`). `dropna()` removes the first row which will be `NaN`.
*   We annualize the average daily returns and daily volatility assuming 252 trading days.
*   We calculate the annual covariance matrix.
*   We return a dictionary containing the head of the daily returns DataFrame, the average annual returns, the annual volatility, and the covariance matrix. You can access these individual items in Excel by referencing the cell containing the dictionary output and adding the key, e.g., `=PY(A1["Average Annual Returns"])` where `A1` is the cell with the dictionary output. You'd then convert *those* cells to 'Excel Value' to see the actual data.

**Viewing the Output:**

*   For the `df_prices` and the dataframes/series within the `results` dictionary (like 'Average Annual Returns', 'Annual Volatility', 'Annual Covariance Matrix'), click the Python cell, then click the Python icon/button next to the formula bar, and select "Excel Value" (**Ctrl+Shift+Alt+M**) to spill the data into your sheet.
*   The 'Daily Returns Head' will also be a DataFrame.

**Further Analysis:**

Here are some advanced portfolio optimization techniques you could explore:

1. **Efficient Frontier Analysis:**
   - Implement the full Markowitz optimization
   - Calculate the optimal portfolio weights
   - Plot the efficient frontier curve

2. **Risk-Adjusted Performance Metrics:**
   - Calculate Sharpe Ratio and Sortino Ratio
   - Implement Value at Risk (VaR) analysis
   - Add Maximum Drawdown calculations

3. **Advanced Portfolio Constraints:**
   - Add sector allocation constraints
   - Implement transaction cost modeling
   - Design rebalancing strategies

4. **Alternative Risk Measures:**
   - Calculate downside deviation
   - Implement CVaR/Expected Shortfall
   - Add tail risk analysis

5. **Portfolio Stress Testing:**
   - Implement Monte Carlo simulations
   - Add scenario analysis capabilities
   - Create risk factor sensitivity tests

**Next Steps:**

Calculating the full efficient frontier requires optimization techniques (typically using `scipy.optimize.minimize`). While `scipy` is available, setting up the optimization problem (defining objective function, constraints) is more involved than a simple code block. The calculations we've done (returns, volatility, covariance) are the essential first steps.

The next topic in the series is [Financial Analysis - Financial Statement Analysis](./01-Financial%20Analysis_02-Financial%20Statement%20Analysis.md), which builds upon these concepts by examining company-specific financial metrics.