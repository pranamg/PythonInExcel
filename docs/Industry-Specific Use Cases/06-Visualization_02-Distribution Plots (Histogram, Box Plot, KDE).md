**Visualization - 2. Distribution Plots (Histogram, Box Plot, KDE)**

Distribution plots provide essential insights into the shape, spread, and central tendency of numerical data, while helping identify potential outliers. This guide demonstrates how to create and interpret these statistical visualizations.

Based on [`piplist.txt`](./README.md) output, you should have `pandas`, `numpy`, `seaborn`, and `matplotlib` to create professional-quality distribution plots.

**Step 1: Generate Sample Data for Distribution Plots**

We'll create a dummy dataset with a few numerical columns that have different distributions (e.g., normal-ish, skewed, with outliers).

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy data for Distribution Plots
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

num_records = 800

# Simulate data for a numerical variable (e.g., Customer Value) with some skewness
# Use a log-normal distribution pattern for positive skew
customer_value = np.random.lognormal(mean=5.0, sigma=0.8, size=num_records)
# Scale and round for plausible customer value numbers
customer_value = (customer_value * 10).round(2)
# Ensure min value is not extremely low
customer_value = np.maximum(10, customer_value)


# Simulate data for another numerical variable (e.g., Waiting Time) with some outliers
waiting_time = np.random.exponential(scale=5, size=num_records) # Exponential distribution (skewed)
waiting_time = (waiting_time * 60).round(0) # Convert to minutes, round to integer
# Add a few extreme outliers
outlier_indices = random.sample(range(num_records), 5)
waiting_time[outlier_indices] = [500, 650, 700, 800, 1000]


# Simulate a third numerical variable (e.g., Process Time) more normally distributed
process_time = np.random.normal(loc=30, scale=8, size=num_records)
process_time = process_time.round(0).clip(5, 60) # Clip to realistic range (5-60 minutes)


data = {
    'RecordID': range(1, num_records + 1),
    'CustomerValue': customer_value,
    'WaitingTime_minutes': waiting_time,
    'ProcessTime_minutes': process_time
}

df_dist_data = pd.DataFrame(data)

# Add some missing values
for col in ['CustomerValue', 'WaitingTime_minutes', 'ProcessTime_minutes']:
     missing_indices = random.sample(range(num_records), int(num_records * random.uniform(0.02, 0.05))) # 2-5% missing
     df_dist_data.loc[missing_indices, col] = np.nan


df_dist_data # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_dist_data` with three numerical columns:
    *   `CustomerValue`: Designed to be positively skewed (tail extends to the right).
    *   `WaitingTime_minutes`: Designed with an exponential distribution (also skewed) and explicit outliers.
    *   `ProcessTime_minutes`: Designed to be more normally distributed within a range.
*   Missing values (`np.nan`) are added to all numerical columns.
*   The result, `df_dist_data`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `DistributionData`.

**Step 2: Create and Style Distribution Plots**

Now, we'll load this dummy data and create a histogram with KDE overlay, a box plot, and a violin plot to visualize the distributions of these numerical variables, applying the specified style guidelines.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"DistributionData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Create and style distribution plots (Histogram, Box Plot, Violin Plot)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from Excel
# IMPORTANT: Replace "DistributionData" with the actual name of your Excel range or Table
df = xl("DistributionData[#All]", headers=True)

# Ensure numerical columns are numeric, coercing errors
numerical_cols = ['CustomerValue', 'WaitingTime_minutes', 'ProcessTime_minutes']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# --- Apply Custom Style Guidelines ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False # Turn off default grid
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs


# --- 1. Histogram with KDE Overlay (CustomerValue) ---
# Drop NaNs for plotting
df_plot1 = df.dropna(subset=['CustomerValue'])

fig1, ax1 = plt.subplots(figsize=(10, 6))
# Use histplot with kde=True for histogram and KDE line
sns.histplot(data=df_plot1, x='CustomerValue', kde=True, ax=ax1, color='#188ce5', bins=30) # Blue

ax1.set_title('Distribution of Customer Value (Histogram + KDE)', fontsize=14, color='#1a1a24')
ax1.set_xlabel('Customer Value', fontsize=12, color='#1a1a24')
ax1.set_ylabel('Frequency', fontsize=12, color='#1a1a24')
sns.despine(ax=ax1, top=True, right=True)
ax1.grid(False)


plt.tight_layout()


# --- 2. Box Plot (WaitingTime_minutes) ---
# Drop NaNs for plotting
df_plot2 = df.dropna(subset=['WaitingTime_minutes'])

fig2, ax2 = plt.subplots(figsize=(8, 6))
# Box plots are good for showing quartiles and outliers
sns.boxplot(data=df_plot2, y='WaitingTime_minutes', ax=ax2, color='#ff6d00') # Orange

ax2.set_title('Distribution of Waiting Time (Box Plot)', fontsize=14, color='#1a1a24')
ax2.set_ylabel('Waiting Time (minutes)', fontsize=12, color='#1a1a24')
# Hide x-axis label and ticks as it's a single box plot
ax2.set_xticklabels([])
ax2.set_xlabel('')
sns.despine(ax=ax2, top=True, right=True, bottom=True, left=False) # Remove top, right, bottom spines

plt.tight_layout()


# --- 3. Violin Plot (ProcessTime_minutes) ---
# Drop NaNs for plotting
df_plot3 = df.dropna(subset=['ProcessTime_minutes'])

fig3, ax3 = plt.subplots(figsize=(8, 6))
# Violin plots show the density distribution (like KDE) along with box plot features
sns.violinplot(data=df_plot3, y='ProcessTime_minutes', ax=ax3, color='#2db757') # Green

ax3.set_title('Distribution of Process Time (Violin Plot)', fontsize=14, color='#1a1a24')
ax3.set_ylabel('Process Time (minutes)', fontsize=12, color='#1a1a24')
# Hide x-axis labels/ticks
ax3.set_xticklabels([])
ax3.set_xlabel('')
sns.despine(ax=ax3, top=True, right=True, bottom=True, left=False) # Remove top, right, bottom spines


plt.tight_layout()


# Output results
# Return a dictionary containing the plot figures
output = {
    'CustomerValue_Distribution_Plot': fig1, # Histogram + KDE
    'WaitingTime_Box_Plot': fig2,
    'ProcessTime_Violin_Plot': fig3,
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy data using `xl()`. **Remember to replace `"DistributionData"`**.
*   We ensure numerical columns are correctly typed, coercing errors to `NaN`.
*   **Style Guidelines:** We set Matplotlib's `rcParams` and use `seaborn.set_theme()` for styling.
*   **Histogram with KDE:**
    *   We filter out rows with missing `CustomerValue`.
    *   `seaborn.histplot()` is used with `kde=True` to plot the histogram (bars showing frequency in bins) and overlay a Kernel Density Estimate curve (a smoothed representation of the distribution). This is great for seeing the shape and modality (number of peaks).
    *   Applied blue color (`#188ce5`), title, labels, spines, and turned off the grid.
*   **Box Plot:**
    *   We filter out rows with missing `WaitingTime_minutes`.
    *   `seaborn.boxplot()` is used. Box plots are excellent for quickly visualizing the median (line in the box), quartiles (the box itself), the range of most data (whiskers), and potential outliers (individual points beyond the whiskers). They are particularly useful for identifying skewness and outliers.
    *   Applied orange color (`#ff6d00`), title, label, customized spine removal.
*   **Violin Plot:**
    *   We filter out rows with missing `ProcessTime_minutes`.
    *   `seaborn.violinplot()` is used. Violin plots combine the aspects of a box plot (showing median, quartiles) with a kernel density estimate, providing a richer view of the distribution shape than a simple box plot, especially for multi-modal distributions.
    *   Applied green color (`#2db757`), title, label, customized spine removal.
*   We return a dictionary containing the three Matplotlib figure objects.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   For each plot figure object ('CustomerValue_Distribution_Plot', 'WaitingTime_Box_Plot', 'ProcessTime_Violin_Plot'), select "Picture in Cell" > "Create Reference" to see the plots spilled into your worksheet.

These distribution plots provide complementary perspectives on your numerical data's characteristics, helping identify patterns, outliers, and underlying distributions.

**Further Analysis:**

Here are some advanced distribution analysis techniques you could apply to this dataset:

1. **Advanced Distribution Analysis:**
   - Fit theoretical distributions (e.g., normal, lognormal, gamma)
   - Perform goodness-of-fit tests
   - Create Q-Q plots for distribution comparison

2. **Multi-dimensional Distribution Analysis:**
   - Create conditional box plots by categories
   - Implement violin plots with nested categories
   - Generate ridge plots for comparing multiple distributions

3. **Outlier Analysis:**
   - Implement statistical outlier detection methods
   - Create distribution plots with highlighted outliers
   - Compare different outlier detection techniques

4. **Distribution Comparison:**
   - Add statistical test results to plots
   - Create side-by-side distribution comparisons
   - Implement before/after distribution analysis

5. **Advanced KDE Analysis:**
   - Use adaptive bandwidth selection
   - Implement bivariate KDE plots
   - Create weighted KDE plots

The next topic in the series is [Visualization - Relationship Plots (Scatter, Pair Plot, Heatmap)](./06-Visualization_03-Relationship%20Plots%20(Scatter,%20Pair%20Plot,%20Heatmap).md), which explores techniques for visualizing relationships between variables.