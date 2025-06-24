# Example Notebook - Basic Data Analysis

This notebook demonstrates basic data analysis capabilities using Python in Excel.

## Loading Data

First, let's create some sample data in Excel and load it using Python.

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample data
data = {
    'Date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='M'),
    'Sales': np.random.normal(1000, 100, 12),
    'Expenses': np.random.normal(800, 50, 12),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 12)
}

df = pd.DataFrame(data)
df
```

## Basic Analysis

Let's perform some basic analysis on our data.

```python
# Calculate summary statistics
summary = df.describe()
print("Summary Statistics:")
summary
```

## Visualization

Create some visualizations to better understand the data.

```python
# Set the style for better-looking plots
plt.style.use('seaborn')

# Create a figure with multiple subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Sales and Expenses over time
ax1.plot(df['Date'], df['Sales'], marker='o', label='Sales')
ax1.plot(df['Date'], df['Expenses'], marker='s', label='Expenses')
ax1.set_title('Sales and Expenses Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Amount')
ax1.legend()
ax1.grid(True)

# Plot 2: Sales by Region (Box Plot)
sns.boxplot(data=df, x='Region', y='Sales', ax=ax2)
ax2.set_title('Sales Distribution by Region')
ax2.set_xlabel('Region')
ax2.set_ylabel('Sales')

plt.tight_layout()
plt.show()
```

## Advanced Analysis

Calculate some business metrics.

```python
# Calculate profit and profit margin
df['Profit'] = df['Sales'] - df['Expenses']
df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100

# Group by region
region_analysis = df.groupby('Region').agg({
    'Sales': ['mean', 'std'],
    'Profit': ['mean', 'std'],
    'Profit_Margin': 'mean'
}).round(2)

print("Regional Performance Analysis:")
region_analysis
```
