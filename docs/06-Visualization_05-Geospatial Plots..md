**Visualization - 5. Geospatial Plots**

Geospatial plots display data on maps to show geographic distributions and patterns. While full geographic visualization typically requires specialized libraries like `geopandas`, `folium`, or `plotly.express`, this guide demonstrates how to create basic geographic visualizations using standard plotting libraries.

Based on [`piplist.txt`](./README.md) output, you should have `matplotlib` and `seaborn` to create scatter plots with latitude and longitude coordinates. While these plots won't include map backgrounds or geographical boundaries, they effectively show the relative geographic distribution of data points.

Note: For interactive maps, geographical boundaries, or map tiles, additional libraries would be required. This guide focuses on what's possible with the core visualization libraries.

I will generate dummy data with Latitude and Longitude and show you how to plot points on a standard scatter plot to visualize their relative locations.

**Step 1: Generate Sample Data with Geographic Coordinates**

We'll create a dummy dataset with customer locations, including Latitude and Longitude coordinates and a value (e.g., customer spending) associated with that location.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy data with Geographic Coordinates
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

num_locations = 300

# Simulate locations within a rough geographic area (e.g., North America bounds)
# Note: For a realistic plot, these coordinates would need to correspond to your actual region
min_lat, max_lat = 25, 50 # Example Latitude range
min_lon, max_lon = -125, -70 # Example Longitude range

data = {
    'LocationID': range(1, num_locations + 1),
    'City': [fake.city() for _ in range(num_locations)],
    'State': [fake.state_abbr() for _ in range(num_locations)],
    'Latitude': [random.uniform(min_lat, max_lat) for _ in range(num_locations)],
    'Longitude': [random.uniform(min_lon, max_lon) for _ in range(num_locations)],
    'CustomerSpending': [round(random.uniform(100, 5000), 2) for _ in range(num_locations)] # A value metric
}

df_geo_data = pd.DataFrame(data)

# Add some missing values
for col in ['Latitude', 'Longitude', 'CustomerSpending']:
    missing_indices = random.sample(range(num_locations), int(num_locations * random.uniform(0.02, 0.05))) # 2-5% missing
    df_geo_data.loc[missing_indices, col] = np.nan


df_geo_data # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_geo_data` with columns for `LocationID`, `City`, `State`, `Latitude`, `Longitude`, and `CustomerSpending`.
*   Latitude and Longitude are generated within a predefined range to simulate points in a geographic area.
*   `CustomerSpending` is included as a numerical value that could be represented by marker size or color on the plot.
*   Missing values are introduced.
*   The result, `df_geo_data`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `GeospatialData`.

**Step 2: Create and Style a Geographic Scatter Plot**

Now, we'll load this dummy data, drop rows with missing coordinates, and create a scatter plot using Latitude and Longitude, applying the specified style guidelines. We will use `CustomerSpending` to vary the size of the points.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"GeospatialData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Create and style a geographic scatter plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from Excel
# IMPORTANT: Replace "GeospatialData" with the actual name of your Excel range or Table
df = xl("GeospatialData[#All]", headers=True)

# Ensure numerical columns are numeric, coercing errors
numerical_cols = ['Latitude', 'Longitude', 'CustomerSpending']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing Latitude or Longitude for plotting
df_plot = df.dropna(subset=['Latitude', 'Longitude']).copy()

# Handle potential outliers in CustomerSpending before mapping to size
# Cap spending at a reasonable value for plotting to avoid huge markers
max_spending_cap = df_plot['CustomerSpending'].quantile(0.95) # Cap at 95th percentile
df_plot['Spending_For_Size'] = df_plot['CustomerSpending'].clip(upper=max_spending_cap)

# Scale the Spending_For_Size to map to marker size aesthetically
# Example: map [min, max] spending to [5, 200] marker size
min_spending = df_plot['Spending_For_Size'].min()
max_spending = df_plot['Spending_For_Size'].max()
min_marker_size, max_marker_size = 20, 300 # Define range for marker sizes

# Apply scaling formula: new_value = (old_value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
df_plot['Marker_Size'] = ((df_plot['Spending_For_Size'] - min_spending) / (max_spending - min_spending)) * (max_marker_size - min_marker_size) + min_marker_size
# Handle case where all spending values are identical (division by zero)
df_plot['Marker_Size'] = df_plot['Marker_Size'].fillna(min_marker_size)


# --- Apply Custom Style Guidelines ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False # Turn off default grid
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs


# --- Create Geographic Scatter Plot ---
fig1, ax1 = plt.subplots(figsize=(10, 8))

# Use seaborn scatterplot
# Use Latitude on Y axis, Longitude on X axis as is conventional for map coordinates
# Use size='Marker_Size' to make point size represent customer spending
# Use color from guidelines, e.g., Teal
sns.scatterplot(data=df_plot, x='Longitude', y='Latitude', size='Marker_Size', sizes=(min_marker_size, max_marker_size), legend=False, ax=ax1, color='#27acaa', alpha=0.7) # Teal, sizes maps data range to marker size range

ax1.set_title('Geographic Distribution of Customers (Colored points by location, Size by Spending)', fontsize=14, color='#1a1a24')
ax1.set_xlabel('Longitude', fontsize=12, color='#1a1a24')
ax1.set_ylabel('Latitude', fontsize=12, color='#1a1a24')

# Optional: Set aspect ratio to better represent geographic space (requires careful handling of limits)
# ax1.set_aspect('equal', adjustable='box') # Might distort if the lat/lon range is large and not projected

sns.despine(ax=ax1, top=True, right=True)
ax1.grid(False) # Explicitly turn off grid


plt.tight_layout()

# --- Explanation of Limitations ---
limitations_text = """
Limitations:
This plot shows relative geographic positions using Latitude and Longitude on a standard scatter plot.
Due to available libraries ('matplotlib', 'seaborn'), it does NOT:
- Show actual geographical boundaries (states, countries, etc.).
- Use a map image or interactive map tiles.
- Handle complex geographic projections accurately over large areas.

True geospatial plotting often requires libraries like 'geopandas', 'folium', or 'plotly.express[geo]', which are not in your current environment ('pip list' output).
"""
print(limitations_text) # Print this note to the console/error details

# Cannot return multi-line text easily as a DataFrame element.
# Can return a single line note DataFrame if needed, or rely on the print output.
# Let's add a note in the output dictionary as a single string in a DataFrame.
limitations_df = pd.DataFrame({'Note on Geospatial Plots': [limitations_text]})


# Output results
# Return a dictionary containing the plot figure and the limitations note
output = {
    'Geographic_Scatter_Plot': fig1,
    'Plotting_Limitations_Note': limitations_df, # Return limitations note as DataFrame
    'Data_Used_For_Plotting_Head': df_plot.head() # Show head of cleaned data used for plot
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy geospatial data. **Remember to replace `"GeospatialData"`**.
*   We ensure numerical columns are correctly typed, coercing errors to `NaN`.
*   We drop rows with missing `Latitude` or `Longitude` since these points cannot be placed on the plot.
*   We handle potential outliers in `CustomerSpending` by capping values at the 95th percentile before using them for point sizing. This prevents a few extreme values from making other points too small to see.
*   We scale the capped spending values to map them to a reasonable range of marker sizes using a linear scaling formula.
*   **Style Guidelines:** Global style parameters are set using `rcParams` and `seaborn.set_theme()`.
*   **Geographic Scatter Plot:**
    *   `seaborn.scatterplot()` is used. We map `Longitude` to the x-axis and `Latitude` to the y-axis, which is standard for geographic coordinates.
    *   `size='Marker_Size'` tells seaborn to vary the point size based on the `Marker_Size` column we created (derived from `CustomerSpending`). `sizes=(min, max)` specifies the minimum and maximum marker sizes to use.
    *   Applied teal color (`#27acaa`) with some transparency (`alpha`), title, labels, spines, and turned off the grid.
*   **Limitations Note:** The code includes a print statement outlining the limitations of this plot compared to true geospatial visualization tools due to the available libraries. This is important for setting user expectations. A DataFrame containing this note is also included in the output dictionary.
*   We return a dictionary containing the Matplotlib figure object, the DataFrame note, and the head of the data actually used for plotting.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames ('Plotting_Limitations_Note', 'Data_Used_For_Plotting_Head') to spill them into your sheet.
*   For the plot figure object ('Geographic_Scatter_Plot'), select "Picture in Cell" > "Create Reference" to see the plot spilled into your worksheet.

This scatter plot demonstrates how to visualize spatial data distributions and incorporate additional dimensions through marker properties like size. While basic, these plots can provide valuable geographic insights when working with coordinate-based data.

**Further Analysis:**

Here are some advanced geospatial visualization techniques you could explore with additional libraries:

1. **Interactive Maps:**
   - Implement interactive maps using Folium or Plotly
   - Add clickable markers with pop-up information
   - Create choropleth maps for region-based analysis

2. **Advanced Spatial Analysis:**
   - Perform spatial clustering analysis
   - Calculate and visualize geographic density
   - Create distance-based relationship maps

3. **Multi-layered Visualization:**
   - Combine multiple data layers on a single map
   - Add geographical boundaries and features
   - Implement heat maps for density visualization

4. **Time-based Geographic Analysis:**
   - Create animated geographic visualizations
   - Show movement patterns over time
   - Visualize temporal-spatial relationships

5. **Custom Base Maps:**
   - Add different map styles (satellite, terrain, etc.)
   - Implement custom tile layers
   - Create specialized map projections

The next topic in the series is [Reporting & Automation - Generating Summaries](./07-Reporting%20%26%20Automation_01-Generating%20Summaries.md), which covers techniques for generating summaries, reports, and automating Excel tasks with Python.