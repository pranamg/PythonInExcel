Okay, let's explore **Visualization - 3. Relationship Plots (Scatter, Pair Plot, Heatmap)**.

These visualizations are crucial for understanding how different variables in your dataset relate to each other.
*   **Scatter Plots:** Show the relationship between two numerical variables.
*   **Pair Plots:** Display scatter plots for every pair of numerical variables in a dataset, and often histograms or KDE plots on the diagonal to show individual variable distributions. Useful for quick exploration of many relationships.
*   **Heatmaps:** Commonly used to visualize correlation matrices, showing the strength and direction of pairwise relationships between numerical variables in a compact grid format.

Your `piplist.txt` includes `pandas`, `numpy`, `seaborn`, and `matplotlib`, providing all the necessary functions for these tasks.

**Step 1: Generate Sample Data for Relationship Plots**

We'll create a dummy dataset with several numerical columns that have varying degrees of correlation, and also a categorical column to potentially differentiate points in scatter/pair plots. We'll include missing values.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy data for Relationship Plots
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

num_records = 600

# Simulate numerical variables with some correlation
# Create a base set of uncorrelated normal variables
base_vars = np.random.randn(num_records, 4) # 4 variables

# Introduce correlation by creating linear combinations
# Var1: Base
# Var2: Related to Var1 + noise
# Var3: Related to Var1 and Var2 + noise
# Var4: Less related + noise

numerical_data = pd.DataFrame({
    'Feature_A': base_vars[:, 0] * 10 + 50, # Scale around 50
    'Feature_B': base_vars[:, 0] * 7 + base_vars[:, 1] * 5 + 30, # Correlated with A
    'Feature_C': base_vars[:, 0] * 4 + base_vars[:, 1] * 3 + base_vars[:, 2] * 8 + 70, # Correlated with A and B
    'Feature_D': base_vars[:, 3] * 10 + 20 # Less correlated
})
# Ensure values are somewhat positive and scale appropriately
numerical_data = (numerical_data * 5 + 50).clip(1, 200).round(2)


# Add a categorical variable
categories = ['Group 1', 'Group 2', 'Group 3']
group_data = random.choices(categories, weights=[0.4, 0.3, 0.3], k=num_records)
# Introduce some correlation between category and one feature (e.g., Feature_A)
# Add an offset to Feature_A for Group 3
for i in range(num_records):
    if group_data[i] == 'Group 3':
        numerical_data.loc[i, 'Feature_A'] = numerical_data.loc[i, 'Feature_A'] * random.uniform(1.2, 1.5)


# Create final DataFrame
df_rel_data = pd.DataFrame({
    'ObservationID': range(1, num_records + 1),
    'Group': group_data
}).join(numerical_data)


# Introduce some missing values
for col in ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']:
    missing_indices = random.sample(range(num_records), int(num_records * random.uniform(0.03, 0.08))) # 3-8% missing
    df_rel_data.loc[missing_indices, col] = np.nan

missing_group_indices = random.sample(range(num_records), int(num_records * 0.02))
df_rel_data.loc[missing_group_indices, 'Group'] = np.nan


# Shuffle rows
df_rel_data = df_rel_data.sample(frac=1, random_state=42).reset_index(drop=True)


df_rel_data # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_rel_data` with four numerical columns (`Feature_A` to `Feature_D`) designed to have varying positive correlations, and a categorical column (`Group`).
*   Some missing values are introduced in both numerical and categorical columns.
*   The result, `df_rel_data`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `RelationshipData`.

**Step 2: Create and Style Relationship Plots**

Now, we'll load this dummy data and create a scatter plot with grouping, a pair plot, and a correlation heatmap, applying the specified style guidelines.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"RelationshipData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Create and style relationship plots (Scatter, Pair Plot, Heatmap)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from Excel
# IMPORTANT: Replace "RelationshipData" with the actual name of your Excel range or Table
df = xl("RelationshipData[#All]", headers=True)

# Ensure numerical columns are numeric, coercing errors
numerical_cols = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Ensure Group is treated as string, converting potential NaN object to np.nan
df['Group'] = df['Group'].astype(str).replace('nan', np.nan)


# --- Apply Custom Style Guidelines ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
# For relationship plots, especially pairplot/heatmap, standard grid isn't typical axis grid
plt.rcParams['axes.grid'] = False
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs


# --- 1. Scatter Plot (e.g., Feature_A vs Feature_B, colored by Group) ---
# Drop NaNs for plotting relevant columns
df_scatter = df.dropna(subset=['Feature_A', 'Feature_B', 'Group'])

fig1, ax1 = plt.subplots(figsize=(8, 6))
# Use seaborn scatterplot, specifying hue for coloring by Group
# Use palette colors from guidelines: Yellow, Blue, Off-black
group_colors = ['#ffe600', '#188ce5', '#1a1a24'] # Yellow, Blue, Off-black

sns.scatterplot(data=df_scatter, x='Feature_A', y='Feature_B', hue='Group', ax=ax1, palette=group_colors[:len(df_scatter['Group'].unique())], alpha=0.7, s=50)

ax1.set_title('Feature_B vs. Feature_A (Colored by Group)', fontsize=14, color='#1a1a24')
ax1.set_xlabel('Feature_A', fontsize=12, color='#1a1a24')
ax1.set_ylabel('Feature_B', fontsize=12, color='#1a1a24')
sns.despine(ax=ax1, top=True, right=True)
# ax1.grid(False) # Redundant with rcParams


plt.tight_layout()


# --- 2. Pair Plot (Pairwise relationships between numerical features) ---
# Drop NaNs from numerical columns before creating the pair plot
df_pairplot = df[numerical_cols].dropna()

# PairGrid allows more control over plot styling than sns.pairplot directly
g = sns.PairGrid(df_pairplot, palette=group_colors) # Can add hue='Group' here too if desired and included in df_pairplot
g.map_upper(sns.scatterplot, s=10, alpha=0.5) # Scatter plots in upper triangle
g.map_lower(sns.kdeplot, cmap="Blues_d") # KDE plots in lower triangle
g.map_diag(sns.histplot, kde=True) # Histograms with KDE on the diagonal

# Apply titles and labels to individual axes if needed, but PairGrid handles titles automatically often
# Customize ticks/spines - can iterate through axes if granular control is needed
# For simplicity, we rely on the global style set earlier where possible


# pairplot object doesn't directly return a single figure, but a grid object.
# We can extract the underlying figure
fig2 = g.fig
fig2.suptitle('Pairwise Relationships between Numerical Features', y=1.02, fontsize=16, color='#1a1a24') # Add title above grid

# Adjust layout - might need manual adjustment for PairGrid titles/spacing
fig2.tight_layout()
# Remove the default grid lines from seaborn theme for pairplot axes
for ax in fig2.axes:
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True) # Despine individual axes


# --- 3. Correlation Heatmap ---
# Calculate the correlation matrix, dropping rows with NaNs first for consistency
correlation_matrix = df[numerical_cols].dropna().corr()

fig3, ax3 = plt.subplots(figsize=(8, 7))
# Use a diverging colormap like 'coolwarm' for correlations (-1 to +1)
# Or sequential if focusing on strength regardless of direction, but diverging is standard
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax3, cbar_kws={'label': 'Correlation Coefficient'})

ax3.set_title('Correlation Matrix of Numerical Features', fontsize=14, color='#1a1a24')
ax3.tick_params(axis='both', which='major', labelsize=10, color='#1a1a24') # Adjust label size and color
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right') # Rotate x-axis labels
ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)
sns.despine(ax=ax3, top=True, right=True, left=False, bottom=False) # Keep bottom/left for heatmap labels
ax3.grid(False) # Explicitly turn off grid


plt.tight_layout()


# Output results
# Return a dictionary containing the plot figures
output = {
    'FeatureA_vs_FeatureB_Scatter_Plot': fig1,
    'Pairwise_Relationships_Plot': fig2,
    'Correlation_Heatmap': fig3,
    'Correlation_Matrix_Values': correlation_matrix # Also return the matrix values as DataFrame
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy data using `xl()`. **Remember to replace `"RelationshipData"`**.
*   We ensure numerical columns are correctly typed and handle missing values in categorical columns explicitly.
*   **Style Guidelines:** We set Matplotlib's `rcParams` and `seaborn.set_theme()` for styling. Grid is turned off globally, then specific adjustments are made per plot.
*   **Scatter Plot:**
    *   We filter out rows with NaNs in the columns needed for this plot (`Feature_A`, `Feature_B`, `Group`).
    *   `seaborn.scatterplot()` is used, mapping `Feature_A` to x and `Feature_B` to y. `hue='Group'` colors the points based on the categorical variable, allowing visual inspection of whether the relationship differs across groups.
    *   Applied blue, yellow, and off-black colors from the guidelines for the different groups using `palette`. Title, labels, spines applied.
*   **Pair Plot:**
    *   We drop rows with NaNs in any of the selected numerical columns as `pairplot` often requires this.
    *   `seaborn.PairGrid()` is used to set up the grid structure for pairwise plots. We map different plot types to the upper triangle (scatter), lower triangle (KDE contours), and diagonal (histograms with KDE).
    *   Applied global style settings and manually adjusted plot title and removed grids/despined axes after creation. Pair plots are excellent for getting a quick overview of relationships and distributions for multiple numerical variables simultaneously.
*   **Correlation Heatmap:**
    *   We calculate the pairwise Pearson correlation matrix using `.corr()` on the numerical columns after dropping rows with NaNs.
    *   `seaborn.heatmap()` is used to visualize the matrix. The color intensity represents the correlation coefficient (-1 to +1). `annot=True` displays the correlation values on the heatmap. A diverging colormap (`coolwarm`) is used, which is standard for correlations (showing strong positive vs. strong negative).
    *   Applied title, adjusted tick label size/rotation for readability, and removed unnecessary spines.
*   We return a dictionary containing the three Matplotlib figure objects and the correlation matrix values as a DataFrame.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrame ('Correlation_Matrix_Values') to spill it into your sheet.
*   For each plot figure object ('FeatureA_vs_FeatureB_Scatter_Plot', 'Pairwise_Relationships_Plot', 'Correlation_Heatmap'), select "Picture in Cell" > "Create Reference" to see the plots spilled into your worksheet.

These plots provide powerful visual tools for exploring bivariate and multivariate relationships within your data.

Would you like to proceed to the next use case: "Visualization - 4. Composition Plots (Pie, Stacked Bar)"?