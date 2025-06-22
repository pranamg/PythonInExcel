The next topic in the data preparation series is **Data Cleaning & Preparation - 3. Data Integration**.

Data integration is a crucial process that combines data from multiple sources into a single, unified view. This section demonstrates how to use Python and pandas to merge DataFrames based on common columns or concatenate them into comprehensive datasets.

Based on [`piplist.txt`](./README.md) output, `pandas` is the primary library available and needed for these operations. `numpy` might be used in dummy data generation.

**Step 1: Generate Sample Data from Multiple Sources**

We'll create two dummy DataFrames representing different pieces of information that need to be combined, for example, customer demographics and their order history.

In a new Excel cell, enter `=PY` and paste the following code for the **Customers** data, then press **Ctrl+Enter**:

```python
# Generate dummy Customer Data
import pandas as pd
from faker import Faker
import random
import numpy as np # Needed for potential np.nan

fake = Faker()

num_customers = 150

customer_data = {
    'CustomerID': [f'C{1000 + i}' for i in range(num_customers)],
    'Name': [fake.name() for _ in range(num_customers)],
    'City': [fake.city() if random.random() > 0.05 else np.nan for _ in range(num_customers)], # Some missing cities
    'JoinDate': [fake.date_between(start_date='-5y', end_date='today') for _ in range(num_customers)]
}

df_customers = pd.DataFrame(customer_data)

# Add a few customers who might not have orders later (for demonstration of join types)
for i in range(5):
     new_cust_id = f'C{2000 + i}'
     new_cust_data = {'CustomerID': new_cust_id, 'Name': fake.name(), 'City': fake.city(), 'JoinDate': fake.date_between(start_date='-1y', end_date='today')}
     df_customers = pd.concat([df_customers, pd.DataFrame([new_cust_data])], ignore_index=True)


df_customers # Output the DataFrame
```

Let's assume this data is placed in a range or Table named `CustomerInfo`.

In a **separate, new** Excel cell, enter `=PY` and paste the following code for the **Orders** data, then press **Ctrl+Enter**:

```python
# Generate dummy Orders Data
import pandas as pd
from faker import Faker
import random
from datetime import date, timedelta
import numpy as np # Needed for potential np.nan

fake = Faker()

num_orders = 500 # Total orders
start_date = date(2023, 1, 1)
end_date = date(2024, 5, 31)

# Use Customer IDs generated previously, but not all of them will have orders
# Mix in some CustomerIDs that do NOT exist in df_customers (for demonstration of join types)
existing_customer_ids = [f'C{1000 + i}' for i in range(150)]
non_existing_customer_ids = [f'C{3000 + i}' for i in range(10)] # These won't match CustomerInfo
all_possible_ids = existing_customer_ids + non_existing_customer_ids

order_data = []
for i in range(num_orders):
    order_id = f'O{10000 + i}'
    customer_id = random.choice(all_possible_ids) # Choose from existing or non-existing
    order_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    amount = round(random.uniform(20, 1000), 2) if random.random() > 0.02 else np.nan # Some missing amounts

    order_data.append([order_id, customer_id, order_date, amount])

df_orders = pd.DataFrame(order_data, columns=['OrderID', 'CustomerID', 'OrderDate', 'TotalAmount'])

# Ensure OrderDate is datetime
df_orders['OrderDate'] = pd.to_datetime(df_orders['OrderDate'])


df_orders # Output the DataFrame
```

Let's assume this data is placed in a range or Table named `OrderHistory`.

**Step 2: Perform Data Integration (Merge/Join)**

Now, we'll load these two dummy DataFrames from Excel and merge them based on the common `CustomerID` column. We will demonstrate an `inner` merge (only keeping customers with orders) and a `left` merge (keeping all customers and adding their order info where available).

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"CustomerInfo"` and `"OrderHistory"` with the actual names of the Excel ranges/Tables where your dummy data is. Press **Ctrl+Enter**.

```python
# Perform data integration using merge/join
import pandas as pd
import numpy as np

# Load the two DataFrames from Excel
# IMPORTANT: Replace "CustomerInfo" and "OrderHistory" with your actual names
df_customers = xl("CustomerInfo[#All]", headers=True)
df_orders = xl("OrderHistory[#All]", headers=True)

# Ensure key column has consistent type (often read as object/string)
df_customers['CustomerID'] = df_customers['CustomerID'].astype(str)
df_orders['CustomerID'] = df_orders['CustomerID'].astype(str)

# Ensure Date columns are datetime if needed (though not used for merge key)
df_customers['JoinDate'] = pd.to_datetime(df_customers['JoinDate'], errors='coerce')
df_orders['OrderDate'] = pd.to_datetime(df_orders['OrderDate'], errors='coerce')
df_orders['TotalAmount'] = pd.to_numeric(df_orders['TotalAmount'], errors='coerce')


# --- Integration Method 1: Inner Merge ---
# Keeps only rows where the merge key (CustomerID) exists in *both* DataFrames.
# Result will have customers who have placed at least one order.
df_inner_merged = pd.merge(df_customers, df_orders, on='CustomerID', how='inner')


# --- Integration Method 2: Left Merge ---
# Keeps all rows from the 'left' DataFrame (df_customers) and adds matching rows from the 'right' (df_orders).
# If a customer from df_customers has no matching order in df_orders, the order columns will have NaN values.
df_left_merged = pd.merge(df_customers, df_orders, on='CustomerID', how='left')


# --- Optional: Concatenation Example ---
# If you had order data split into two tables, e.g., 'Orders_2023' and 'Orders_2024'
# Assuming they have the same column structure: OrderID, CustomerID, OrderDate, TotalAmount
# df_orders_2023 = xl("Orders_2023[#All]", headers=True)
# df_orders_2024 = xl("Orders_2024[#All]", headers=True)
# df_all_orders = pd.concat([df_orders_2023, df_orders_2024], ignore_index=True)
# This df_all_orders could then be merged with df_customers

# Since we only generated one orders table, we can't demonstrate concat with separate sources directly.
# But know that pd.concat stacks DataFrames vertically (rows) or horizontally (columns).
# If stacking rows, columns must align or you handle non-matching columns.


# Output results
# Return a dictionary containing both merged DataFrames and a note about concat
output = {
    'Inner Merged Data Head': df_inner_merged.head(),
    'Inner Merged Data Shape': pd.DataFrame({'Rows': [df_inner_merged.shape[0]], 'Columns': [df_inner_merged.shape[1]]}),
    'Left Merged Data Head': df_left_merged.head(),
    'Left Merged Data Shape': pd.DataFrame({'Rows': [df_left_merged.shape[0]], 'Columns': [df_left_merged.shape[1]]}),
    # Add a note about concat - can't return direct text, use a simple DataFrame
    'Note on Concatenation': pd.DataFrame({'Info': ["Use pd.concat to stack DataFrames with similar columns (e.g., historical data split by year)."]})
}

output # Output the dictionary
```

**Explanation:**

*   We load the `CustomerInfo` and `OrderHistory` DataFrames using `xl()`. **Remember to replace the source names.**
*   We ensure the `CustomerID` column in both DataFrames is treated as a string type, which is good practice for join keys to avoid unexpected behavior. We also clean up other column types.
*   `pd.merge()` is the core function used for joining DataFrames based on one or more common columns (`on='CustomerID'`).
*   `how='inner'` performs an inner join: it keeps only rows where the `CustomerID` is present in *both* `df_customers` and `df_orders`.
*   `how='left'` performs a left join: it keeps all rows from `df_customers` (the "left" DataFrame) and brings in matching rows from `df_orders`. If a `CustomerID` from `df_customers` doesn't appear in `df_orders`, the columns from `df_orders` will show `NaN` (Not a Number) for that row.
*   A note about `pd.concat()` is included to mention its use for stacking DataFrames that have a similar column structure (like combining yearly datasets).
*   We return a dictionary containing the heads and shapes of the resulting merged DataFrames, plus the note.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames within the output dictionary to spill them into your sheet. This will show the first few rows and the dimensions of the integrated data.

The resulting merged DataFrames (`df_inner_merged`, `df_left_merged`) combine the customer information with their order details, enabling advanced analytics such as calculating total customer spending and analyzing order frequency based on demographics. The next topic in the series is ["Data Cleaning & Preparation - 4. Data Reshaping"](./03-Data%20Cleaning%20&%20Preparation_04-Data%20Reshaping.md), which explores techniques for restructuring data layouts to support different analysis requirements.

**Further Analysis:**
* **Advanced Join Operations:** Implementing complex multi-key joins, fuzzy matching for imperfect keys, or rolling joins for time-series data
* **Data Reconciliation:** Building automated validation checks to ensure data consistency post-integration
* **Entity Resolution:** Using machine learning techniques to match records across datasets without exact keys
* **Schema Evolution:** Handling changes in data structure over time with dynamic schema mapping
* **Performance Optimization:** Implementing efficient chunking strategies for large dataset integration