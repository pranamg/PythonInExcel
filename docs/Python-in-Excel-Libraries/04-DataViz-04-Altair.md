# Leveraging Altair in Python in Excel

Altair is a declarative visualization library for Python built on the Vega-Lite grammar. Within Python in Excel, Altair enables you to craft interactive, web-ready charts directly in your spreadsheet cells. Below is a guide to get you started.

## 1. Availability and Setup

To use Altair in Python in Excel, ensure your Excel environment supports Python formulas and that Altair is included in the Anaconda distribution powering Python in Excel. Then:

1. **Import Altair**
Reserve the first worksheet for import statements. In a Python cell, type:

```python
=PY(
import altair as alt
)
```

This makes `alt` available for all subsequent Python formulas in the workbook.
2. **Reference Excel Data**
Use the `xl()` function to load ranges or tables into a pandas DataFrame:
    - Entire table with headers: `xl("MyTable[#All]", headers=True)`
    - Range: `xl("A1:C100", headers=True)`
    - Single column: `xl("SalesData[Revenue]")`.

## 2. Basic Chart Construction

Altair charts follow a layered pipeline: define data, map aesthetics, and add mark types.

- **Scatter Plot**

```python
=PY(
data = xl("Table1[#All]", headers=True)
chart = alt.Chart(data).mark_circle(size=60).encode(
  x='VariableX:Q',
  y='VariableY:Q',
  color='Category:N'
)
chart
)
```

- **Bar Chart**

```python
=PY(
df = xl("Sales[#All]", headers=True)
alt.Chart(df).mark_bar().encode(
  x='Region:N',
  y='sum(Revenue):Q'
)
)
```

## 3. Adding Interactivity

Altair’s built-in interactivity transforms static charts into dashboards:

- **Interactive Selection**

```python
=PY(
data = xl("Data[#All]", headers=True)
selection = alt.selection_multi(fields=['Category'])
alt.Chart(data).mark_point().encode(
  x='X:Q', y='Y:Q',
  color=alt.condition(selection, 'Category:N', alt.value('lightgray'))
).add_selection(selection)
)
```

- **Tooltips**

```python
=PY(
alt.Chart(data).mark_circle().encode(
  x='X:Q', y='Y:Q',
  tooltip=['Category:N','Value:Q']
)
)
```

## 4. Layered and Faceted Views

- **Layering**
Combine multiple marks in one chart:

```python
=PY(
base = alt.Chart(data).encode(x='X:Q')
points = base.mark_point(color='steelblue').encode(y='Y1:Q')
line   = base.mark_line(color='darkorange').encode(y='Y2:Q')
alt.layer(points, line)
)
```

- **Faceting**
Create small multiples:

```python
=PY(
alt.Chart(data).mark_bar().encode(
  x='Category:N', y='Value:Q'
).facet(
  column='Group:N'
)
)
```

## 5. Styling and Themes

- **Themes**
Apply built-in themes to adjust overall aesthetics:

```python
=PY(
alt.themes.enable('dark')
)
```

- **Customizing Marks**
Pass style arguments to `mark_*()` methods:

```python
=PY(
alt.Chart(data).mark_circle(size=100, opacity=0.7, color='teal')
)
```

## 6. Best Practices

- **Import Once**: Place imports on the first worksheet to avoid redundancy.
- **Data Preparation**: Clean and aggregate data via pandas operations within Python cells before plotting.
- **Performance**: For large datasets, filter or sample in Python to keep charts responsive.
- **Execution Order**: Python in Excel follows row-major order—ensure prerequisite cells precede dependent charts.

By following these steps, you can harness Altair’s grammar-of-graphics approach to build expressive, interactive visualizations directly in Excel, combining Python’s declarative APIs with Excel’s accessibility and collaboration features.
