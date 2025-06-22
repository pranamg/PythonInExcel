# Leveraging Matplotlib in Python in Excel

Python in Excel automatically provides the Matplotlib library as `plt`, enabling you to craft a wide variety of static visualizations directly within Excel cells. By combining `xl()` data references with Matplotlib’s plotting functions, you can generate, customize, and embed charts without leaving the spreadsheet environment.

## 1. Getting Started

To use Matplotlib in Excel, simply begin a cell with the `=PY(` formula and write Python code using Matplotlib’s `pyplot` interface. For example:

```python
=PY(
import matplotlib.pyplot as plt
)
```

You only need to import once—subsequent Python cells can reference `plt` directly.

## 2. Referencing Excel Data

Matplotlib plots require numeric arrays or sequences. Use the `xl()` function to pull cell ranges or table columns into Python:

- Single range with headers: `xl("A1:B10", headers=True)`
- Table column: `xl("Table1[Sales]")`
- Entire table: `xl("Table1[#All]", headers=True)`

These return pandas DataFrames or Series suitable for plotting.

## 3. Core Plot Types

### 3.1 Scatter Plots

Visualize relationships between variables:

```python
=PY(
plt.scatter(
    xl("Table1[sepal_length]"),
    xl("Table1[sepal_width]")
)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Scatter Plot")
)
```

### 3.2 Line Charts

Ideal for time-series or ordered data:

```python
=PY(
dates = xl("A2:A50")
values = xl("B2:B50")
plt.plot(dates, values, marker="o", linestyle="-", color="purple")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Trend")
plt.grid(True)
)
```

### 3.3 Bar Charts

For categorical comparisons:

```python
=PY(
categories = xl("B2:B6")
values     = xl("C2:C6")
plt.bar(categories, values, color="skyblue")
plt.xlabel("Category")
plt.ylabel("Value")
plt.title("Sales by Category")
)
```

### 3.4 Histograms

To view data distribution:

```python
=PY(
data = xl("A1:A100")
plt.hist(data, bins=20, color="steelblue", alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Data Distribution")
)
```

## 4. Customizations and Styling

- **Colors and Themes**: Pass `color` or use Matplotlib colormaps (e.g., `cmap="coolwarm"`).
- **Legends**: Call `plt.legend()` after plotting multiple series.
- **Annotations**: Use `plt.text()` or `ax.bar_label()` to add data labels.
- **Figure Size**: Set with `plt.figure(figsize=(width, height))` if needed.

## 5. Displaying and Extracting Plots

After committing the `=PY()` formula, Excel returns an image object. To view and manipulate it:

1. Click the image icon in the cell for a preview.
2. Right-click and select **Display Plot over Cells** to extract and resize the chart within the grid.

## 6. Best Practices

- **Row-Major Order**: If splitting code across cells, ensure dependent cells follow the order of execution.
- **Performance**: For large datasets, filter or sample data before plotting to maintain responsiveness.
- **Reuse Imports**: Place import statements once on the first worksheet to avoid redundancy.
- **Axis Formatting**: For dates, use Matplotlib’s `dates` module if necessary.

---

Leveraging Matplotlib within Python in Excel empowers analysts to create publication-quality static visualizations without leaving the spreadsheet interface, combining Excel’s familiarity with Python’s rich plotting capabilities.
