# Leveraging Bokeh in Python in Excel

Bokeh is a powerful Python library for creating interactive, web-ready visualizations. Within Python in Excel, you can harness Bokeh to build dynamic charts directly in your spreadsheet cells. Below is a practical guide to setting up and using Bokeh in Python in Excel.

## 1. Availability and Setup

Before you begin, ensure your Excel environment supports Python formulas and that Bokeh is included in the Anaconda distribution used by Python in Excel. No additional installation is required—simply import Bokeh within a Python cell to get started.

1. In your Excel workbook, select a cell and enter `=PY(` to enable Python mode.
2. On the first worksheet, import Bokeh modules in a Python cell:

```python
=PY(
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
)
```

This makes Bokeh’s `figure`, `show`, and related functions available throughout your workbook.

## 2. Referencing Excel Data

Use the `xl()` function to pull cell ranges or table columns into Python. Bokeh works with pandas DataFrames and ColumnDataSource objects:

- Entire table with headers: `xl("Table1[#All]", headers=True)`
- Single column: `xl("Table1[Sales]")`
- Range with headers: `xl("A1:B100", headers=True)`

Convert a DataFrame into a Bokeh data source:

```python
=PY(
df = xl("SalesData[#All]", headers=True)
from bokeh.models import ColumnDataSource
source = ColumnDataSource(df)
)
```

This `source` can then be used in Bokeh glyph methods.

## 3. Creating Core Plot Types

### 3.1 Line Chart

Ideal for time-series data:

```python
=PY(
p = figure(x_axis_type="datetime", title="Sales Over Time")
p.line(x=xl("Dates[Date]"), y=xl("Values[Amount]"), line_width=2, color="navy")
show(p)
)
```

### 3.2 Scatter Plot

Visualize relationships between numeric variables:

```python
=PY(
p = figure(title="Height vs Weight", tools="pan,wheel_zoom")
p.circle(x=xl("Data[Height]"), y=xl("Data[Weight]"), size=8, color="green", alpha=0.6)
show(p)
)
```

### 3.3 Bar Chart

For categorical comparisons:

```python
=PY(
from bokeh.transform import factor_cmap
categories = xl("Categories[Name]")
values     = xl("Categories[Value]")
p = figure(x_range=categories, title="Category Values")
p.vbar(x=categories, top=values, width=0.8,
       color=factor_cmap('x', palette=["#718dbf","#e84d60"], factors=list(set(categories))))
show(p)
)
```

### 3.4 Histogram

To view data distribution:

```python
=PY(
hist, edges = np.histogram(xl("Data[Value]"), bins=20)
p = figure(title="Value Distribution")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="orange", line_color="white")
show(p)
)
```

## 4. Interactivity and Layouts

Bokeh supports interactive tools and layouts:

- **Tools**: Enable pan, zoom, hover, box select, and reset by specifying `tools="hover,box_zoom,reset"` in the `figure()` call.
- **Hover Tool**:

```python
=PY(
from bokeh.models import HoverTool
p = figure(tools="hover")
hover = p.select_one(HoverTool)
hover.tooltips = [("Category", "@x"), ("Value", "@top")]
p.vbar(x=categories, top=values)
show(p)
)
```

- **Layouts**: Combine multiple plots with row/column layouts:

```python
=PY(
from bokeh.layouts import row
row(p1, p2)
)
```

## 5. Best Practices

- **Import Once**: Place Bokeh imports on the first sheet to persist across cells.
- **Data Preparation**: Clean and aggregate data with pandas before plotting to improve performance.
- **Performance**: Limit glyph counts or sample data for large datasets to maintain responsiveness.
- **Display in Excel**: After entering a Python cell with a Bokeh chart, click the PY icon and choose **Display Plot over Cells** to resize and position the chart within the grid.

By following these steps, you can leverage Bokeh’s interactive visualization capabilities directly within Excel, blending Python’s powerful plotting API with Excel’s familiar interface.
