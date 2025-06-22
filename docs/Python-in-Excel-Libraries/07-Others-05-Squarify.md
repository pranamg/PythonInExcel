# Leveraging Squarify in Python in Excel

Squarify is a Python library for creating treemap visualizations, which display hierarchical data as nested rectangles. With Python in Excel, you can use Squarify to generate treemaps for visualizing proportions and part-to-whole relationships directly within your spreadsheets.

## 1. Setup and Imports

To use Squarify, reserve the first worksheet for import statements:

```python
=PY(
import squarify
import matplotlib.pyplot as plt
)
```

This makes the Squarify and Matplotlib libraries available for all subsequent Python cells.

## 2. Creating a Treemap

- **Basic treemap:**

```python
=PY(
values = [500, 300, 200, 100]
labels = ["A", "B", "C", "D"]
colors = ["#ffe600", "#188ce5", "#2db757", "#ff4136"]
plt.figure(figsize=(6, 4))
squarify.plot(sizes=values, label=labels, color=colors, alpha=0.8)
plt.axis('off')
plt.show()
)
```

- **Treemap from Excel data:**

```python
=PY(
import pandas as pd
df = xl("SalesData[#All]", headers=True)
plt.figure(figsize=(8, 5))
squarify.plot(sizes=df['Sales'], label=df['Category'], alpha=0.7)
plt.axis('off')
plt.show()
)
```

## 3. Customization

- **Change color palette:**

```python
=PY(
colors = plt.cm.viridis([0.2, 0.4, 0.6, 0.8])
squarify.plot(sizes=values, color=colors)
plt.show()
)
```

## 4. Best Practices

- **Imports Once**: Place all import statements on the first worksheet.
- **Data Preparation**: Clean and structure data before plotting.
- **Output Management**: Use `plt.show()` to display the treemap in Excel.
- **Performance**: For large datasets, sample or aggregate data to maintain responsiveness.

By leveraging Squarify in Python in Excel, you can create clear, informative treemaps for visualizing hierarchical and proportional data within your spreadsheets.
