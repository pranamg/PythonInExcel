Okay, let's move on to the next visualization use case: **Visualization - 4. Composition Plots (Pie, Stacked Bar)**.

Composition plots show how different parts contribute to a whole.
*   **Pie Charts:** Best for showing the proportion of a single categorical variable when the number of categories is small.
*   **Stacked Bar Charts:** Useful for comparing the composition of different categories across different groups, or showing how proportions change over time or across groups.

Your `piplist.txt` includes `pandas` for data preparation and `matplotlib`/`seaborn` for plotting, which are ideal for these tasks.

**Step 1: Generate Sample Data for Composition Plots**

We'll create two dummy datasets:
1.  One with a categorical variable and associated values for a **Pie Chart** (e.g., market share by product line).
2.  One with two categorical variables and associated values for a **Stacked Bar Chart** (e.g., sales by product category per region).

In a new Excel cell, enter `=PY` and paste the following code for the **Market Share** data, then press **Ctrl+Enter**:

```python
# Generate dummy data for Pie Chart (Market Share)
import pandas as pd
import numpy as np
import random

product_lines = ['Product Line A', 'Product Line B', 'Product Line C', 'Product Line D', 'Product Line E']
# Simulate revenue contributing to total market
revenue_values = np.random.randint(50000, 200000, size=len(product_lines))
# Add one small category to represent 'Other' or smaller lines
revenue_values = np.append(revenue_values, 25000)
product_lines.append('Other')

data = {
    'ProductLine': product_lines,
    'Revenue': revenue_values
}

df_market_share = pd.DataFrame(data)

# Add a couple of missing values
missing_indices = random.sample(range(len(df_market_share)), 1)
df_market_share.loc[missing_indices, 'Revenue'] = np.nan


df_market_share # Output the DataFrame
```

Let's assume this data is placed in a range or Table named `MarketShareData`.

In a **separate, new** Excel cell, enter `=PY` and paste the following code for the **Sales by Category and Region** data, then press **Ctrl+Enter**:

```python
# Generate dummy data for Stacked Bar Chart (Sales by Category and Region)
import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()

regions = ['North', 'South', 'East', 'West']
categories = ['Electronics', 'Clothing', 'Home Goods', 'Groceries']
num_records_per_region = 100

data = []
for region in regions:
    for _ in range(num_records_per_region):
        category = random.choice(categories)
        sales_amount = random.uniform(50, 500)
        data.append([region, category, sales_amount])

df_sales_composition = pd.DataFrame(data, columns=['Region', 'ProductCategory', 'Sales'])

# Add some missing values
missing_indices_sales = random.sample(range(len(df_sales_composition)), int(len(df_sales_composition) * 0.03))
df_sales_composition.loc[missing_indices_sales, 'Sales'] = np.nan

missing_indices_cat = random.sample(range(len(df_sales_composition)), int(len(df_sales_composition) * 0.02))
df_sales_composition.loc[missing_indices_cat, 'ProductCategory'] = np.nan

missing_indices_region = random.sample(range(len(df_sales_composition)), int(len(df_sales_composition) * 0.02))
df_sales_composition.loc[missing_indices_region, 'Region'] = np.nan


df_sales_composition # Output the DataFrame
```

Let's assume this data is placed in a range or Table named `SalesCompositionData`.

**Step 2: Create and Style Composition Plots**

Now, we'll load these two dummy DataFrames from Excel, aggregate the data as needed, and create a pie chart and a stacked bar chart, applying the specified style guidelines.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"MarketShareData"` and `"SalesCompositionData"` with the actual names of the Excel ranges/Tables where your dummy data is. Press **Ctrl+Enter**.

```python
# Create and style composition plots (Pie Chart, Stacked Bar Chart)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DataFrames from Excel
# IMPORTANT: Replace the source names with your actual names
df_market_share = xl("MarketShareData[#All]", headers=True)
df_sales_composition = xl("SalesCompositionData[#All]", headers=True)

# Ensure data types are appropriate
df_market_share['Revenue'] = pd.to_numeric(df_market_share['Revenue'], errors='coerce')
df_market_share['ProductLine'] = df_market_share['ProductLine'].astype(str).replace('nan', np.nan)

df_sales_composition['Sales'] = pd.to_numeric(df_sales_composition['Sales'], errors='coerce')
df_sales_composition['Region'] = df_sales_composition['Region'].astype(str).replace('nan', np.nan)
df_sales_composition['ProductCategory'] = df_sales_composition['ProductCategory'].astype(str).replace('nan', np.nan)


# --- Apply Custom Style Guidelines ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24' # Axis color (less relevant for pie)
plt.rcParams['axes.linewidth'] = 1 # Axis line width (less relevant for pie)
plt.rcParams['axes.grid'] = False # Turn off default grid
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs


# Define color palette from guidelines
plot_colors = ['#ffe600', '#188ce5', '#2db757', '#ff6d00', '#750e5c', '#ff4136', '#27acaa', '#1a1a24'] # Yellow, Blue, Green, Orange, Purple, Salmon, Teal, Off-black


# --- 1. Pie Chart (Market Share by Product Line) ---
# Aggregate data: Sum Revenue by ProductLine, dropping NaNs for aggregation
market_share_agg = df_market_share.groupby('ProductLine')['Revenue'].sum().reset_index()
# Remove product lines with 0 or NaN revenue after sum (e.g., if all values were NaN)
market_share_agg = market_share_agg[market_share_agg['Revenue'].notna() & (market_share_agg['Revenue'] > 0)]

# Sort by Revenue descending to order slices
market_share_agg = market_share_agg.sort_values('Revenue', ascending=False)

# Handle potential missing ProductLine if any existed after aggregation
# If 'nan' is a product line after replace/astype, handle it
market_share_agg['ProductLine'] = market_share_agg['ProductLine'].replace({np.nan: 'Missing Category'})


fig1, ax1 = plt.subplots(figsize=(8, 8)) # Square figure for the pie chart

# Create pie chart
wedges, texts, autotexts = ax1.pie(market_share_agg['Revenue'],
                                   labels=market_share_agg['ProductLine'],
                                   autopct='%1.1f%%', # Format percentages with 1 decimal place
                                   startangle=90,    # Start the first slice at the top
                                   colors=plot_colors[:len(market_share_agg)]) # Use colors from palette

# Style autotexts (percentage labels)
for autotext in autotexts:
    autotext.set_color('#1a1a24') # Off-black color for percentage text
    autotext.set_fontsize(10)

# Style texts (label texts)
for text in texts:
    text.set_color('#1a1a24') # Off-black color for slice labels
    text.set_fontsize(10)

ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Market Share by Product Line', fontsize=14, color='#1a1a24')

# Remove default axes appearance (pie charts don't usually have standard axes)
ax1.set_frame_on(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines[['top', 'right', 'left', 'bottom']].set_visible(False)


plt.tight_layout()


# --- 2. Stacked Bar Chart (Sales by Category per Region) ---
# Aggregate data: Sum Sales by Region and ProductCategory
# Pivot table to get Region as index, ProductCategory as columns, Sales as values
sales_composition_pivot = pd.pivot_table(df_sales_composition,
                                        values='Sales',
                                        index='Region',
                                        columns='ProductCategory',
                                        aggfunc='sum',
                                        fill_value=0) # Fill missing combinations with 0 sales

# Handle potential missing Category/Region after pivot
# If 'nan' is a category after replace/astype, rename the column
sales_composition_pivot = sales_composition_pivot.rename(columns={np.nan: 'Missing Category'})
# If 'nan' is an index after pivot, rename the index
sales_composition_pivot = sales_composition_pivot.rename(index={np.nan: 'Missing Region'})

# Optional: Sort regions or columns if needed
# sales_composition_pivot = sales_composition_pivot.sort_index() # Sort regions alphabetically


fig2, ax2 = plt.subplots(figsize=(10, 7))

# Create stacked bar chart using the DataFrame's plot method
# Pass ax=ax2 to draw on the created axes
# Use colors from the palette, ensure enough colors for categories
sales_composition_pivot.plot(kind='bar', stacked=True, ax=ax2, color=plot_colors[:len(sales_composition_pivot.columns)], width=0.8)

ax2.set_title('Total Sales by Product Category per Region', fontsize=14, color='#1a1a24')
ax2.set_xlabel('Region', fontsize=12, color='#1a1a24')
ax2.set_ylabel('Total Sales', fontsize=12, color='#1a1a24')

# Rotate x-axis labels for readability if needed
plt.xticks(rotation=0, ha='center') # Keep horizontal for few regions

ax2.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside plot
sns.despine(ax=ax2, top=True, right=True)
ax2.grid(False)


plt.tight_layout()


# Output results
# Return a dictionary containing the plot figures
output = {
    'Market_Share_Pie_Chart': fig1,
    'Sales_Composition_Stacked_Bar_Chart': fig2,
    'Market_Share_Data_Aggregated': market_share_agg, # Return aggregated data for reference
    'Sales_Composition_Data_Aggregated': sales_composition_pivot # Return pivot data for reference
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy data using `xl()`. **Remember to replace the source names.**
*   We ensure numerical columns are numeric and categorical columns are string/object, handling potential `NaN` values appropriately.
*   **Style Guidelines:** Global style parameters are set using `rcParams` and `seaborn.set_theme()`. A color palette from the guidelines is defined.
*   **Pie Chart:**
    *   We group the `MarketShareData` by `ProductLine` and sum the `Revenue` to get the total contribution for each line. Rows with missing or zero revenue after summation are dropped.
    *   `matplotlib.pyplot.pie()` is used to create the pie chart.
        *   `labels` are the product line names.
        *   `autopct='%1.1f%%'` formats the percentage labels shown on each slice.
        *   `colors` are assigned from the defined palette.
    *   We manually style the percentage and label text color and font size to match the guidelines.
    *   `ax1.axis('equal')` makes the pie circular. Axis frame, ticks, and spines are removed as they are not standard for pie charts.
*   **Stacked Bar Chart:**
    *   We use `pd.pivot_table()` on the `SalesCompositionData` to aggregate `Sales` by `Region` (index) and `ProductCategory` (columns), summing the sales (`aggfunc='sum'`) and filling any combinations with no sales with 0 (`fill_value=0`).
    *   We handle potential missing category/region labels that might appear as index/column names after aggregation.
    *   The `pivot_table` DataFrame has a built-in `.plot()` method which is convenient for generating plots directly from the aggregated data. We use `kind='bar'` and `stacked=True`.
    *   Applied title, labels, used palette colors for the stacks, and placed the legend outside the plot area. Rotated x-ticks slightly for better readability if regions had longer names. Despined and turned off the grid.
*   We return a dictionary containing the two Matplotlib figure objects and the aggregated DataFrames used to create the plots for reference.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames ('Market_Share_Data_Aggregated', 'Sales_Composition_Data_Aggregated') to spill them into your sheet.
*   For each plot figure object ('Market_Share_Pie_Chart', 'Sales_Composition_Stacked_Bar_Chart'), select "Picture in Cell" > "Create Reference" to see the plots spilled into your worksheet.

These plots visually represent the composition of your data, showing how different parts contribute to a whole in simple and grouped scenarios.

Would you like to proceed to the next use case: "Visualization - 5. Geospatial Plots"?