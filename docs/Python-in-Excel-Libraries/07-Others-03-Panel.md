# Leveraging Panel in Python in Excel

Panel is a high-level Python library for creating interactive dashboards and data apps. With Python in Excel, you can use Panel to build interactive widgets, visualizations, and layouts that respond to user input—all embedded directly in your spreadsheets.

## 1. Setup and Imports

To use Panel, reserve the first worksheet for import statements:

```python
=PY(
import panel as pn
pn.extension()
)
```

This loads Panel and enables its interactive features for all subsequent Python cells.

## 2. Creating Widgets

- **Slider widget:**

```python
=PY(
slider = pn.widgets.IntSlider(name='Value', start=0, end=10, value=5)
slider
)
```

- **Text input widget:**

```python
=PY(
text = pn.widgets.TextInput(name='Enter text')
text
)
```

## 3. Building Interactive Layouts

- **Combine widgets and plots:**

```python
=PY(
import matplotlib.pyplot as plt
import numpy as np
slider = pn.widgets.IntSlider(name='Multiplier', start=1, end=10, value=2)
@pn.depends(slider)
def plot(mult):
    x = np.linspace(0, 10, 100)
    y = mult * np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    return fig
panel = pn.Column(slider, plot)
panel
)
```

## 4. Displaying DataFrames and Tables

- **Show a DataFrame as a table:**

```python
=PY(
import pandas as pd
df = pd.DataFrame({'A': range(5), 'B': range(5, 10)})
pn.widgets.DataFrame(df)
)
```

## 5. Best Practices

- **Imports Once**: Place all import statements on the first worksheet.
- **Use `pn.extension()`**: Always call this once to enable Panel features.
- **Performance**: For complex dashboards, break layouts into multiple cells for clarity.
- **Output Management**: Return Panel objects or widgets for interactive display in Excel.

By leveraging Panel in Python in Excel, you can build interactive dashboards and data apps, making your spreadsheets more dynamic and user-friendly.
