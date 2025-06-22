**Financial Analysis - 3. Investment Analysis**.

This area focuses on evaluating individual assets or a group of assets. We'll look at returns, risk, correlation, and potentially some simple technical indicators.

Based on [`piplist.txt`](./README.md) output, you should have `pandas`, `numpy`, `seaborn`, `matplotlib`, `scipy.stats` which are great for these tasks. `pandas_ta` was mentioned in the use case list, but it's **not** present in your provided `piplist.txt`. Therefore, we cannot directly use `pandas_ta`. I will show you how to calculate a simple moving average manually using `pandas` instead, as this is a common technical indicator and achievable with available libraries.

**Step 1: Generate Sample Investment Price Data**

We can reuse or slightly modify the dummy price data generation from the Portfolio Optimization case. This time, let's generate data specifically for a few major tech stocks for clarity.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy daily stock prices for investment analysis
import pandas as pd
import numpy as np
from datetime import date, timedelta
# Assuming Faker is available and can be used like this in Excel env
# from faker import Faker # Not strictly needed for tickers, but good practice if generating company info

# Define date range
start_date = date(2021, 1, 1)
end_date = date(2024, 4, 30)
# Generate dates including weekends initially, then filter
all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# Define assets (real tickers for illustrative purposes)
assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']

# Generate random price movements
base_price = 150 # Starting price level
# Simulate daily percentage changes with some variance
daily_returns = np.random.normal(loc=0.0005, scale=0.008, size=(len(all_dates), len(assets))) # Mean slight positive drift

prices = np.zeros_like(daily_returns)
prices[0, :] = base_price * (1 + np.random.uniform(-0.05, 0.05, len(assets))) # Slightly varied starting prices

# Calculate cumulative prices
for i in range(1, len(all_dates)):
    prices[i, :] = prices[i-1, :] * (1 + daily_returns[i, :])
    # Ensure prices don't go below a reasonable level (simple floor)
    prices[i, :] = np.maximum(prices[i, :], base_price * 0.5)


# Create DataFrame with all dates
df_prices_all = pd.DataFrame(prices, index=pd.to_datetime(all_dates), columns=assets)

# Filter for typical trading days (Mon-Fri)
df_prices_trading_days = df_prices_all[df_prices_all.index.dayofweek < 5].copy()


# Rename the index column for Excel clarity if needed
# df_prices_trading_days.index.name = 'Date'
# df_prices_trading_days = df_prices_trading_days.reset_index()


df_prices_trading_days # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_prices_trading_days` with dummy daily stock prices for a few tech companies.
*   It simulates price movements with a slight upward trend and daily noise.
*   It filters the dates to include only Monday through Friday, more typical of stock market data.
*   The result will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `InvestmentPrices`.

**Step 2: Analyze Returns, Correlation, and Visualize**

Now, let's calculate daily returns, visualize their distribution, calculate and visualize the correlation matrix, and calculate and plot a simple moving average for one stock.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"InvestmentPrices"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Analyze investment data: returns, correlation, simple moving average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats # For distribution analysis

# Load the price data from Excel
# IMPORTANT: Replace "InvestmentPrices" with the actual name of your Excel range or Table
df_raw = xl("InvestmentPrices[#All]", headers=True)

# --- Data Preparation ---

# Make a copy to avoid modifying the original DataFrame loaded by xl() directly in place during manipulation
df_prices = df_raw.copy()

# Assume the first column is the date and remaining columns are numeric prices
# Identify the date column name
date_col_name = df_prices.columns[0]

# Set the date column as index and convert to datetime, dropping the original column
try:
    df_prices = df_prices.set_index(date_col_name, drop=True)
    df_prices.index = pd.to_datetime(df_prices.index)
except Exception as e:
    print(f"Error setting index or converting to datetime. Please ensure the first column is a date and data is clean. Error: {e}")
    # If this fails, output the raw data or a message and stop
    # For debugging, you might return df_raw here or a specific error indicator
    raise # Re-raise the error to make it visible in Excel error details

# Ensure all remaining columns (which should be prices) are numeric
# Coerce errors means non-numeric values will become NaN
for col in df_prices.columns:
    df_prices[col] = pd.to_numeric(df_prices[col], errors='coerce')

# Drop any rows that might have become all NaN after coercion (e.g., if an extra header row was included)
df_prices.dropna(how='all', inplace=True)

# Drop columns that could not be converted to numeric (e.g., if other non-price columns were included accidentally)
df_prices.dropna(axis=1, how='all', inplace=True)

# Check if any numeric columns remain
if df_prices.empty or len(df_prices.columns) < 2: # Need at least 2 assets for correlation matrix
     raise ValueError("Dataframe is empty or contains only non-numeric columns after cleaning. Cannot proceed with analysis.")


# Calculate daily returns - this should now work on numeric columns only
daily_returns = df_prices.pct_change().dropna()


# --- Analysis & Visualization ---

# 1. Visualize Returns Distribution for one asset (e.g., AAPL)
returns_dist_plot_fig = None
asset_for_dist_plot = 'AAPL'
if asset_for_dist_plot in daily_returns.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(daily_returns[asset_for_dist_plot], kde=True, color='#188ce5', ax=plt.gca()) # Blue
    plt.title(f'Distribution of {asset_for_dist_plot} Daily Returns', fontsize=14, color='#1a1a24')
    plt.xlabel('Daily Return', fontsize=12, color='#1a1a24')
    plt.ylabel('Frequency', fontsize=12, color='#1a1a24')
    plt.grid(False) # Explicitly turn off grid
    sns.despine(top=True, right=True) # Remove top and right spines
    plt.tight_layout()
    # Store plot figure
    returns_dist_plot_fig = plt.gcf() # Get current figure
    plt.close() # Close the figure to prevent displaying inline if not desired
else:
    print(f"Warning: Asset '{asset_for_dist_plot}' not found in data for distribution plot.")


# 2. Calculate and Visualize Correlation Matrix
# Ensure there are at least two columns for correlation calculation
correlation_matrix = None
correlation_plot_fig = None
if len(daily_returns.columns) >= 2:
    correlation_matrix = daily_returns.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5, ax=plt.gca())
    plt.title('Asset Daily Returns Correlation Matrix', fontsize=14, color='#1a1a24')
    plt.xticks(rotation=45, ha='right', fontsize=10, color='#1a1a24')
    plt.yticks(rotation=0, fontsize=10, color='#1a1a24')
    plt.tight_layout()
    # Store plot figure
    correlation_plot_fig = plt.gcf() # Get current figure
    plt.close() # Close the figure
else:
    print("Warning: Not enough columns (assets) to calculate correlation matrix.")


# 3. Calculate and Plot Simple Moving Average (SMA) for one asset (e.g., MSFT)
sma_plot_fig = None
sma_data_head = "MSFT data not available for SMA calculation."
stock_to_plot_sma = 'MSFT'
window_size = 20 # 20-day SMA

if stock_to_plot_sma in df_prices.columns:
    # Calculate SMA using pandas rolling window function
    df_single_stock = df_prices[[stock_to_plot_sma]].copy()
    df_single_stock['SMA'] = df_single_stock[stock_to_plot_sma].rolling(window=window_size).mean()
    sma_data_head = df_single_stock.head()

    plt.figure(figsize=(12, 6))
    # Plot original price
    plt.plot(df_single_stock.index, df_single_stock[stock_to_plot_sma], label=stock_to_plot_sma, color='#188ce5', alpha=0.8) # Blue
    # Plot SMA
    plt.plot(df_single_stock.index, df_single_stock['SMA'], label=f'{window_size}-Day SMA', color='#ff6d00', linestyle='--') # Orange

    # Apply custom style guidelines
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.edgecolor'] = '#1a1a24'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.grid'] = False # Turn off default grid

    plt.title(f'{stock_to_plot_sma} Price and {window_size}-Day SMA', fontsize=14, color='#1a1a24')
    plt.xlabel('Date', fontsize=12, color='#1a1a24')
    plt.ylabel('Price', fontsize=12, color='#1a1a24')
    plt.legend()
    plt.grid(False) # Explicitly turn off grid
    sns.despine(top=True, right=True) # Remove top and right spines
    plt.tight_layout()
    # Store plot figure
    sma_plot_fig = plt.gcf() # Get current figure
    plt.close() # Close the figure
else:
     print(f"Warning: Asset '{stock_to_plot_sma}' not found in data for SMA plot.")


# Store results in a dictionary to return multiple outputs
output = {
    'Daily Returns Head': daily_returns.head(),
    'Returns Distribution Plot (AAPL)': returns_dist_plot_fig, # Will be None if AAPL not found
    'Correlation Matrix': correlation_matrix, # Will be None if not enough columns
    'Correlation Matrix Plot': correlation_plot_fig, # Will be None if not enough columns
    f'{stock_to_plot_sma}_SMA_Data_Head': sma_data_head, # Will be string message if MSFT not found
    f'{stock_to_plot_sma}_SMA_Plot': sma_plot_fig # Will be None if MSFT not found
}

output # Output the dictionary containing results
```

**Explanation:**

*   We load the dummy price data using `xl()`. **Remember to replace `"InvestmentPrices"`**.
*   We calculate daily returns using `pct_change()`.
*   We generate a histogram with a Kernel Density Estimate (KDE) for the daily returns of one asset (AAPL) to visualize its distribution using `seaborn`.
*   We calculate the correlation matrix of daily returns for all assets using `corr()`.
*   We visualize the correlation matrix as a heatmap using `seaborn`, which is great for quickly seeing relationships between assets.
*   We calculate a 20-day Simple Moving Average (SMA) for one asset (MSFT) using the `rolling().mean()` function in `pandas`.
*   We plot the price trend alongside the SMA for MSFT.
*   **Custom Style:** Applied the specified style guidelines to the plots (font, colors, axes, spines, no grid).
*   We return a dictionary containing the head of the daily returns DataFrame, the correlation matrix DataFrame, and the three plot figures.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames (like `Daily Returns Head`, `Correlation Matrix`, `MSFT_SMA_Data_Head`).
*   For the plot figures (like `Returns Distribution Plot`, `Correlation Matrix Plot`, `MSFT_SMA_Plot`), select "Picture in Cell" > "Create Reference".

**Further Analysis:**

Here are some advanced investment analysis techniques you could explore:

1. **Technical Analysis:**
   - Implement additional technical indicators (RSI, MACD)
   - Add Bollinger Bands analysis
   - Create candlestick chart patterns

2. **Statistical Analysis:**
   - Calculate advanced risk metrics
   - Implement momentum indicators
   - Create volatility analysis tools

3. **Performance Analysis:**
   - Add risk-adjusted return metrics
   - Implement attribution analysis
   - Create performance persistence tests

4. **Market Analysis:**
   - Add sector rotation analysis
   - Implement market breadth indicators
   - Create sentiment analysis tools

5. **Portfolio Analysis:**
   - Calculate factor exposures
   - Implement style analysis
   - Create portfolio attribution tools

The next topic in the series is [Financial Analysis - Risk Analysis with VaR (Value at Risk)](./01-Financial%20Analysis_04-Risk%20Analysis%20with%20VaR%20(Value%20at%20Risk).md), which explores how to measure and manage investment risk using statistical techniques.