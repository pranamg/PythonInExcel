# Leveraging Seaborn in Python in Excel

Seaborn is included by default in Python in Excel via the Anaconda distribution, imported as `sns`. Its high-level interface simplifies creation of statistically informed, aesthetically pleasing visualizations directly within Excel cells. Below is an organized guide to using Seaborn for data visualization in Python in Excel.

## 1. Setup and Imports

1. Reserve the first worksheet in your workbook for imports and settings.
2. Enter import statements in a Python cell (`=PY(`…`)`) so that they load before any dependent formulas:

```python
=PY(
import seaborn as sns
)
```

Python in Excel persists imports across the workbook.

## 2. Referencing Excel Data

- Use `xl()` to pull data into pandas structures:
    - Entire table: `xl("Table1[#All]", headers=True)`
    - Single column: `xl("Table1[ColumnName]")`
    - Range with headers: `xl("A1:C100", headers=True)`

## 3. Axis-Level Plots

Axis-level functions create single, focused plots.

### 3.1 Scatter Plot

```python
=PY(
sns.scatterplot(
    x=xl("MyData[FeatureX]"),
    y=xl("MyData[FeatureY]"),
    hue=xl("MyData[Category]")
)
)
```

### 3.2 Bar Plot

```python
=PY(
sns.barplot(
    x=xl("Sales[Region]"),
    y=xl("Sales[Revenue]")
)
)
```

### 3.3 Histogram

```python
=PY(
sns.histplot(
    data=xl("Data[Value]"),
    bins=20,
    kde=True
)
)
```

## 4. Figure-Level Plots

### 4.1 Pair Plot

```python
=PY(
pairplot = sns.pairplot(
    xl("Table1[#All]", headers=True)
)
)
```

### 4.2 Cat Plot (Categorical Facets)

```python
=PY(
sns.catplot(
    data=xl("Survey[#All]", headers=True),
    x="Group", y="Score", col="Category",
    kind="box"
)
)
```

## 5. Styling and Themes

Adjust global aesthetics before plotting:

```python
=PY(
sns.set_theme(style="whitegrid", palette="pastel")
)
```

## 6. Customization

- **Titles & Labels**: Call Matplotlib methods after Seaborn call:

```python
=PY(
ax = sns.barplot(...)
ax.set_title("Revenue by Region")
ax.set_xlabel("Region")
)
```

- **Figure Size**: Use Matplotlib’s `plt.gcf().set_size_inches(width, height)` within the same cell.

## 7. Extracting and Displaying Plots

After committing a Python formula, Excel shows an image icon. To view or resize:
- **Display over cells**: Right-click the icon → **Display Plot over Cells**.
- **Embed in cell**: Select **Excel Value** in the Python output menu.

## 8. Best Practices

- **Row-Major Order**: Enter dependent formulas in subsequent rows.
- **Performance**: For large datasets, filter or sample in Python before plotting.
- **Reuse Imports**: Consolidate imports on the first sheet to avoid redundancy.

By following these steps, you can harness Seaborn’s powerful visualization capabilities directly within Excel, enabling sophisticated exploration and presentation of data without leaving the spreadsheet environment.
