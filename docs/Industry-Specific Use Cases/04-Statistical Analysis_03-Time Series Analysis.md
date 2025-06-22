The final topic in the statistical analysis series is **Statistical Analysis - 3. Time Series Analysis**.

Time series analysis focuses on interpreting and understanding data points collected over time. This specialized analytical approach enables the identification of patterns, decomposition of underlying forces (such as seasonality and trends), and provides a foundation for forecasting future values.

Based on [`piplist.txt`](./README.md) output, you have `pandas` (essential for time series data structures and manipulations), `numpy`, `statsmodels` (which has dedicated time series tools), `seaborn`, and `matplotlib` (both great for visualizing time series). This is a strong setup for time series analysis.

**Step 1: Generate Sample Time Series Data**

We'll create a dummy dataset representing a daily time series metric (like sales or website visits) that exhibits a trend, seasonality (e.g., weekly patterns), and random noise.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy daily time series data
import pandas as pd
import numpy as np
from datetime import date, timedelta
import random

# Define date range (daily data for a few years)
start_date = date(2021, 1, 1)
end_date = date(2024, 5, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

n_points = len(dates)

# Simulate components:
# 1. Trend: Linear or slight curve
trend = np.linspace(100, 300, n_points) # Increasing trend

# 2. Seasonality: Weekly pattern (e.g., higher values on weekends)
# Day of the week: 0=Mon, 6=Sun
day_of_week = dates.dayofweek.values # Get the numpy array of day of the week
weekly_seasonality = np.sin(day_of_week * (2 * np.pi / 7)) * 20 # Sin wave for weekly pattern
# Boost values for weekend (Sat/Sun) - perform operation on numpy array
weekly_seasonality[day_of_week >= 5] += 15

# 3. Noise: Random fluctuations
noise = np.random.normal(0, 15, n_points) # Mean 0, Std Dev 15

# Combine components (additive model: Trend + Seasonality + Noise)
value = trend + weekly_seasonality + noise

# Ensure values are non-negative
value = np.maximum(10, value) # Floor at 10

# Create DataFrame
df_ts = pd.DataFrame({'Date': dates, 'Value': value.round(2)})

# Add some missing values randomly
missing_indices = random.sample(range(n_points), int(n_points * 0.03))
df_ts.loc[missing_indices, 'Value'] = np.nan


df_ts # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_ts` with a `Date` column and a `Value` column.
*   It simulates a time series by combining a linear trend, a weekly seasonal pattern (higher values on weekends), and random noise using `numpy`.
*   Missing values (`np.nan`) are randomly introduced in the `Value` column.
*   The result, `df_ts`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `TimeSeriesData`.

**Step 2: Analyze and Visualize Time Series Components and Rolling Statistics**

Now, we'll load this dummy data, perform time series decomposition to separate trend, seasonality, and residuals, calculate a rolling mean, and visualize the results.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"TimeSeriesData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Analyze and visualize time series data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose # For decomposition

# Load the data from Excel
# IMPORTANT: Replace "TimeSeriesData" with the actual name of your Excel range or Table
df = xl("TimeSeriesData[#All]", headers=True)

# Ensure 'Date' is a datetime column and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Drop rows where 'Value' is NaN for decomposition and plotting, but keep original for rolling mean calculation before dropping NaNs
df_clean = df.dropna(subset=['Value'])


# --- 1. Time Series Decomposition ---
# Decompose the time series into trend, seasonality, and residual
# Specify the period for seasonality (e.g., 7 for weekly data)
# Use model='additive' as the components were simulated additively
decomposition = seasonal_decompose(df_clean['Value'], model='additive', period=7) # period=7 for weekly seasonality

# Get the components as pandas Series
trend_component = decomposition.trend
seasonal_component = decomposition.seasonal
residual_component = decomposition.resid
observed_component = decomposition.observed


# --- 2. Rolling Statistics ---
# Calculate rolling mean (e.g., 30-day rolling average) on the original data (before dropping NaNs)
# Note: rolling().mean() automatically handles NaNs by default (requires at least min_periods non-NaNs)
rolling_mean_30d = df['Value'].rolling(window=30, min_periods=1).mean() # min_periods=1 allows calculation even with few points initially


# --- Visualization ---

# Apply custom style guidelines
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False # Turn off default grid
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs

# Plotting the decomposed components requires creating separate axes
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True) # 4 rows, 1 column, share x-axis

# Plot Observed (Original) data
axes[0].plot(observed_component.index, observed_component, label='Observed', color='#188ce5', linewidth=1) # Blue
axes[0].set_title('Observed Data', fontsize=14, color='#1a1a24')
axes[0].set_ylabel('Value', fontsize=12, color='#1a1a24')
axes[0].grid(False)
sns.despine(ax=axes[0], top=True, right=True)

# Plot Trend
axes[1].plot(trend_component.index, trend_component, label='Trend', color='#2db757', linewidth=1) # Green
axes[1].set_title('Trend Component', fontsize=14, color='#1a1a24')
axes[1].set_ylabel('Value', fontsize=12, color='#1a1a24')
axes[1].grid(False)
sns.despine(ax=axes[1], top=True, right=True)

# Plot Seasonality
axes[2].plot(seasonal_component.index, seasonal_component, label='Seasonality', color='#ff6d00', linewidth=1) # Orange
axes[2].set_title('Seasonal Component', fontsize=14, color='#1a1a24')
axes[2].set_ylabel('Value', fontsize=12, color='#1a1a24')
axes[2].grid(False)
sns.despine(ax=axes[2], top=True, right=True)

# Plot Residuals
axes[3].plot(residual_component.index, residual_component, label='Residuals', color='#750e5c', linewidth=1) # Purple
axes[3].set_title('Residual Component', fontsize=14, color='#1a1a24')
axes[3].set_xlabel('Date', fontsize=12, color='#1a1a24')
axes[3].set_ylabel('Value', fontsize=12, color='#1a1a24')
axes[3].grid(False)
sns.despine(ax=axes[3], top=True, right=True)

plt.tight_layout()
# Optional: Autoformat x-axis dates to prevent overlap
fig.autofmt_xdate()


# Plot Original Data and Rolling Mean
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Plot original data (can include NaNs here if rolling mean handles them)
ax2.plot(df.index, df['Value'], label='Original Data', color='#1a1a24', linewidth=0.8, alpha=0.6) # Off-black, slightly transparent

# Plot rolling mean
ax2.plot(rolling_mean_30d.index, rolling_mean_30d, label='30-Day Rolling Mean', color='#ff4136', linewidth=1.5) # Salmon

ax2.set_title('Time Series with 30-Day Rolling Mean', fontsize=14, color='#1a1a24')
ax2.set_xlabel('Date', fontsize=12, color='#1a1a24')
ax2.set_ylabel('Value', fontsize=12, color='#1a1a24')
ax2.legend()
sns.despine(ax=ax2, top=True, right=True)
ax2.grid(False)
fig2.autofmt_xdate()


plt.tight_layout()


# Output results
# Return a dictionary containing heads of component Series and plots
output = {
    'Trend Component Head': trend_component.head(),
    'Seasonal Component Head': seasonal_component.head(),
    'Residual Component Head': residual_component.head(),
    'Rolling 30-Day Mean Head': rolling_mean_30d.head(),
    'Time_Series_Decomposition_Plot': fig, # Plot with all components
    'Time_Series_Rolling_Mean_Plot': fig2, # Plot with original and rolling mean
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy time series data. **Remember to replace `"TimeSeriesData"`**.
*   We ensure the 'Date' column is a datetime object and set it as the DataFrame's index. Time series functions in `pandas` and `statsmodels` often work best with a DatetimeIndex.
*   We create a clean version of the DataFrame (`df_clean`) by dropping rows with `NaN` values in the 'Value' column, as `seasonal_decompose` requires a series without missing values.
*   **Time Series Decomposition:** `statsmodels.tsa.seasonal.seasonal_decompose()` is used to break down the time series into its additive components: Trend, Seasonality, and Residuals. We specify `period=7` assuming a weekly seasonality in our daily data.
*   **Rolling Statistics:** We calculate the 30-day rolling mean of the *original* 'Value' series using `rolling(window=30).mean()`. Pandas' rolling functions can handle NaNs, calculating the mean based on available non-NaN values within the window.
*   **Visualization:**
    *   `fig`: A single figure containing four subplots showing the original time series, and its separated trend, seasonal, and residual components from the decomposition. This is a standard way to visualize decomposition results.
    *   `fig2`: A plot showing the original time series alongside its 30-day rolling mean. The rolling mean helps to smooth out short-term fluctuations and visualize the underlying trend more clearly.
*   **Custom Style:** Applied the specified style guidelines (font, colors - different colors for each component plot and rolling mean, off-black for text/axes, axes, spines, grid, formatted x-axis for dates).
*   We return a dictionary containing the heads of the calculated Series (components and rolling mean) and the two plot figures.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames/Series ('Trend Component Head', 'Seasonal Component Head', 'Residual Component Head', 'Rolling 30-Day Mean Head') to spill them into your sheet.
*   For each plot figure object ('Time_Series_Decomposition_Plot', 'Time_Series_Rolling_Mean_Plot'), select "Picture in Cell" > "Create Reference" to see the plots.

This analysis provides fundamental insights into temporal patterns within the data. While this section focuses on decomposition and pattern identification, more advanced time series techniques, such as ARIMA forecasting models, are covered in the next major category: [**Predictive Modeling**](./05-Predictive%20Modeling_01-Regression%20(Predicting%20Continuous%20Values).md). The predictive modeling section builds upon these statistical foundations to develop models for forecasting future values and understanding complex temporal relationships.

**Further Analysis:**
* **Advanced Decomposition:** Using STL decomposition for complex seasonality patterns and handling missing values
* **Change Point Detection:** Implementing statistical tests to identify structural breaks and regime changes
* **Spectral Analysis:** Using Fourier transforms and periodograms to identify cyclical patterns
* **Cross-Correlation Analysis:** Analyzing lagged relationships between multiple time series
* **Wavelet Analysis:** Implementing continuous wavelet transforms for time-frequency decomposition