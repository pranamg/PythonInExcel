# Leveraging NetworkX in Python in Excel

NetworkX is a powerful Python library for the creation, manipulation, and study of complex networks (graphs). With Python in Excel, you can use NetworkX to analyze relationships, model networks, and visualize graph structures directly within your spreadsheet environment.

## 1. Setup and Import

To use NetworkX in Python in Excel, import it on the first worksheet so it is available workbook-wide:

```python
=PY(
import networkx as nx
)
```

This import persists across all Python formulas in the workbook.

## 2. Creating Graphs from Excel Data

You can create graphs from edge lists or adjacency matrices loaded from Excel tables or ranges:

- **Edge List**: `xl("Edges[#All]", headers=True)`
- **Adjacency Matrix**: `xl("AdjMatrix[#All]", headers=True)`

Example (edge list):

```python
=PY(
edges = xl("Edges[#All]", headers=True)
G = nx.from_pandas_edgelist(edges, source='Source', target='Target')
G
)
```

## 3. Analyzing Graph Properties

NetworkX provides functions for common graph metrics:

```python
=PY(
nodes = G.number_of_nodes()
edges = G.number_of_edges()
degree_centrality = nx.degree_centrality(G)
clustering = nx.clustering(G)
{"Nodes": nodes, "Edges": edges, "DegreeCentrality": degree_centrality, "Clustering": clustering}
)
```

## 4. Visualizing Graphs

You can use Matplotlib to visualize graphs in Excel:

```python
=PY(
import matplotlib.pyplot as plt
nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray')
plt.title("Network Graph")
)
```

## 5. Best Practices

- Place all imports on the first worksheet.
- Use `xl()` to load edge lists or adjacency matrices.
- For large graphs, sample or filter data before visualization.
- Use Excel’s “Display Plot over Cells” to position network diagrams in the grid.

By integrating NetworkX with Python in Excel, you can perform advanced network analysis and visualization directly in your spreadsheets.

<div style="text-align: center">⁂</div>
