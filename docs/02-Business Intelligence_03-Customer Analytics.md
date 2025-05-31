Okay, let's move on to **Business Intelligence - 3. Customer Analytics**.

This area is crucial for understanding your customer base, identifying your most valuable customers, and tailoring strategies. We'll cover RFM analysis, customer segmentation, and basic sentiment analysis.

Your `piplist.txt` confirms you have `pandas`, `numpy`, `scikit-learn` (for clustering), `nltk` (for text analysis), and `seaborn`/`matplotlib` (for visualization), which are all needed for these tasks.

**Step 1: Generate Sample Customer Transaction Data**

We'll create dummy data representing customer transactions over time, including a simple feedback field.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy customer transaction data
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta

fake = Faker()

num_customers = 300
num_transactions_total = 8000 # Total number of transactions
start_date = date(2022, 1, 1)
end_date = date(2024, 5, 15) # A specific 'analysis date' reference point
transaction_dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, periods=num_transactions_total)) # Evenly distributed dates

customer_ids = [f'Cust_{i}' for i in range(num_customers)]

data = []

# Generate transactions
for transaction_date in transaction_dates:
    customer_id = random.choice(customer_ids)
    amount = round(random.uniform(10, 500), 2) # Simulate varied transaction amounts

    # Simulate some simple feedback text
    feedback_choice = random.choices(['positive', 'negative', 'neutral', 'empty'], weights=[0.3, 0.15, 0.45, 0.1], k=1)[0]
    feedback = ""
    if feedback_choice == 'positive':
        feedback = random.choice(["Great product!", "Very happy with the service.", "Highly recommend.", "Fantastic experience.", "Will buy again."])
    elif feedback_choice == 'negative':
        feedback = random.choice(["Disappointed.", "Slow delivery.", "Issue with the item.", "Poor customer support.", "Not satisfied."])
    elif feedback_choice == 'neutral':
        feedback = random.choice(["Product is okay.", "Arrived on time.", "No issues.", "Met expectations."])

    data.append([customer_id, transaction_date, amount, feedback])

df_customers = pd.DataFrame(data, columns=['CustomerID', 'TransactionDate', 'Amount', 'Feedback'])

# Add a transaction ID for uniqueness
df_customers['TransactionID'] = df_customers.index + 1

# Randomize row order after generation
df_customers = df_customers.sample(frac=1, random_state=42).reset_index(drop=True)


df_customers # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_customers` with simulated transaction data for multiple customers.
*   It includes `CustomerID`, `TransactionDate`, `Amount`, and a simple `Feedback` text field.
*   `Faker` is used for generic data simulation setup.
*   It distributes transactions over a period, ensuring customers have varying numbers of transactions and different last purchase dates and total spending.
*   It generates simple positive, negative, or neutral feedback text.
*   The result, `df_customers`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `CustomerTransactions`.

**Step 2: Perform RFM Analysis, Segmentation, Sentiment Analysis, and Visualize**

Now, we'll calculate RFM metrics, segment customers based on these metrics, perform sentiment analysis on feedback, and visualize the results.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"CustomerTransactions"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**.

```python
# Perform RFM analysis, segmentation, sentiment analysis, and visualize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler # For scaling RFM values
from sklearn.cluster import KMeans # For K-Means clustering
import nltk
try: nltk.data.find('sentiment/vader_lexicon.zip'); print('VADER lexicon found.')
except: nltk.download('vader_lexicon'); print('VADER lexicon downloaded.')
from nltk.sentiment.vader import SentimentIntensityAnalyzer # For sentiment analysis

# Ensure NLTK data is available (VADER lexicon).
# In Python in Excel, this might be pre-downloaded. If not, this command might fail.
# You can try running this in a separate cell first if needed:
# =PY(import nltk; try: nltk.data.find('sentiment/vader_lexicon.zip'); print('VADER lexicon found.') except: nltk.download('vader_lexicon'); print('VADER lexicon downloaded.'))


# Load the transaction data from Excel
# IMPORTANT: Replace "CustomerTransactions" with the actual name of your Excel range or Table
df_customers = xl("CustomerTransactions[#All]", headers=True)

# Ensure 'TransactionDate' is a datetime column
df_customers['TransactionDate'] = pd.to_datetime(df_customers['TransactionDate'])
df_customers['CustomerID'] = df_customers['CustomerID'].astype(str)
df_customers['Amount'] = pd.to_numeric(df_customers['Amount'])


# --- RFM Analysis ---

# Define a snapshot date (e.g., the day after the last transaction date)
snapshot_date = df_customers['TransactionDate'].max() + timedelta(days=1)

# Calculate RFM metrics
rfm_df = df_customers.groupby('CustomerID').agg(
    Recency=('TransactionDate', lambda date: (snapshot_date - date.max()).days), # Days since last transaction
    Frequency=('TransactionID', 'count'), # Number of transactions
    Monetary=('Amount', 'sum') # Total spending
).reset_index()

# Ensure Monetary is not zero for clustering (add epsilon or handle separately if needed)
rfm_df['Monetary'] = rfm_df['Monetary'] + 1 # Simple add 1 to avoid log(0) if scaling later, or just handle 0s


# --- Customer Segmentation (using K-Means on RFM) ---

# Scale RFM values - necessary for K-Means as it's distance-based
# Consider log scaling for highly skewed data like Monetary/Frequency, then Standardize
rfm_scaled = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
# Log transform for skewed distributions (handle Recency differently as lower is better)
rfm_scaled['Frequency'] = np.log1p(rfm_scaled['Frequency']) # log(1+x)
rfm_scaled['Monetary'] = np.log1p(rfm_scaled['Monetary'])
# Recency is usually inverse, maybe log(max_recency - recency + 1) or just inverse scale
# Let's just standardize the raw RFM values for simplicity first, after adding 1 to Monetary/Frequency
# rfm_scaled = rfm_df[['Recency', 'Frequency', 'Monetary']].copy() # Use original if log scaling is too complex or data not skewed

scaler = StandardScaler()
rfm_normalized = scaler.fit_transform(rfm_scaled) # Scale the (potentially log-transformed) data

# Determine optimal number of clusters (Elbow method requires plotting, let's pick a number)
n_clusters = 4 # Example: 4 clusters (e.g., Champions, Loyal Customers, etc.)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init to suppress warning

rfm_df['Cluster'] = kmeans.fit_predict(rfm_normalized)

# Analyze cluster characteristics (mean RFM values per cluster)
cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
# Sort clusters based on average Monetary value to get a rough ordering (e.g., 0 being high value)
# cluster_summary = cluster_summary.sort_values('Monetary', ascending=False) # Optional sorting

# Map cluster IDs to potential names based on summary (manual inspection needed)
# Example mapping based on typical RFM patterns (High F/M, Low R = Champions)
# This mapping logic would ideally be based on analyzing the cluster_summary output
# Simple mapping based on sorting might assign 0 to the highest Monetary cluster
# Let's just keep them as numerical IDs for now, but print the summary
# rfm_df['Cluster_Name'] = rfm_df['Cluster'].map(...) # Requires cluster_summary analysis


# --- Sentiment Analysis on Feedback ---

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment_score(text):
    if pd.isna(text) or text.strip() == "":
        return None # Handle missing or empty feedback
    return analyzer.polarity_scores(text)['compound']

# Apply sentiment analysis
df_customers['Sentiment_Score'] = df_customers['Feedback'].apply(get_sentiment_score)

# Classify sentiment
def classify_sentiment(score):
    if score is None:
        return 'No Feedback'
    elif score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df_customers['Sentiment_Category'] = df_customers['Sentiment_Score'].apply(classify_sentiment)

# Summarize sentiment distribution
sentiment_counts = df_customers['Sentiment_Category'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment Category', 'Count']


# --- Visualization ---

# Apply custom style guidelines
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False # Turn off default grid
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs

# 1. RFM Distribution Histograms
fig1, axes = plt.subplots(1, 3, figsize=(15, 5)) # One row, three columns for R, F, M

sns.histplot(rfm_df['Recency'], bins=30, kde=True, ax=axes[0], color='#ff6d00') # Orange
axes[0].set_title('Recency Distribution', fontsize=14, color='#1a1a24')
axes[0].set_xlabel('Days Since Last Purchase', fontsize=12, color='#1a1a24')
axes[0].set_ylabel('Number of Customers', fontsize=12, color='#1a1a24')
sns.despine(ax=axes[0], top=True, right=True)
axes[0].grid(False)


sns.histplot(rfm_df['Frequency'], bins=30, kde=True, ax=axes[1], color='#188ce5') # Blue
axes[1].set_title('Frequency Distribution', fontsize=14, color='#1a1a24')
axes[1].set_xlabel('Number of Transactions', fontsize=12, color='#1a1a24')
axes[1].set_ylabel('', fontsize=12, color='#1a1a24') # Share Y-axis label with the first plot
sns.despine(ax=axes[1], top=True, right=True)
axes[1].grid(False)

sns.histplot(rfm_df['Monetary'], bins=30, kde=True, ax=axes[2], color='#2db757') # Green
axes[2].set_title('Monetary Distribution', fontsize=14, color='#1a1a24')
axes[2].set_xlabel('Total Spending', fontsize=12, color='#1a1a24')
axes[2].set_ylabel('', fontsize=12, color='#1a1a24') # Share Y-axis label
sns.despine(ax=axes[2], top=True, right=True)
axes[2].grid(False)

plt.tight_layout()


# 2. Customer Segmentation Scatter Plot (Frequency vs Monetary, colored by cluster)
fig2, ax2 = plt.subplots(figsize=(10, 6))
# Use original (unscaled) values for plotting, but color by cluster from scaled data
sns.scatterplot(x='Frequency', y='Monetary', hue='Cluster', data=rfm_df, palette='viridis', s=50, ax=ax2) # 'viridis' is a common colormap

ax2.set_title(f'Customer Segmentation by RFM ({n_clusters} Clusters)', fontsize=14, color='#1a1a24')
ax2.set_xlabel('Frequency (Number of Transactions)', fontsize=12, color='#1a1a24')
ax2.set_ylabel('Monetary (Total Spending)', fontsize=12, color='#1a1a24')
ax2.grid(False)
sns.despine(ax=ax2, top=True, right=True)

# Add cluster centers to the plot (requires inverse transform if scaled)
# kmeans.cluster_centers_ are in the scaled space. Need to inverse transform.
# cluster_centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)
# ax2.scatter(cluster_centers_original_scale[:, 1], cluster_centers_original_scale[:, 2], s=300, c='red', marker='X', label='Cluster Centers') # [:,1] for Frequency, [:,2] for Monetary


plt.tight_layout()


# 3. Sentiment Distribution Bar Chart
fig3, ax3 = plt.subplots(figsize=(8, 5))
# Order bars logically
sentiment_order = ['Positive', 'Neutral', 'Negative', 'No Feedback']
# Using a categorical color palette - let's map colors to sentiment
sentiment_colors = {'Positive': '#2db757', 'Neutral': '#188ce5', 'Negative': '#ff4136', 'No Feedback': '#750e5c'} # Green, Blue, Salmon, Purple

# Ensure all categories are in the DataFrame for plotting consistency
sentiment_counts = sentiment_counts.set_index('Sentiment Category').reindex(sentiment_order).fillna(0).reset_index()


sns.barplot(x='Count', y='Sentiment Category', hue = 'Sentiment Category', legend=False, data=sentiment_counts, ax=ax3, palette=[sentiment_colors[cat] for cat in sentiment_counts['Sentiment Category']])


ax3.set_title('Customer Feedback Sentiment Distribution', fontsize=14, color='#1a1a24')
ax3.set_xlabel('Number of Feedbacks', fontsize=12, color='#1a1a24')
ax3.set_ylabel('Sentiment', fontsize=12, color='#1a1a24')
sns.despine(ax=ax3, top=True, right=True)
ax3.grid(False)

# Add data labels (counts)
for index, row in sentiment_counts.iterrows():
     ax3.text(row['Count'], index, f' {int(row["Count"]):,}', color='#1a1a24', va='center')


plt.tight_layout()


# Output results
output = {
    'RFM Metrics Head': rfm_df.head(),
    'Customer Segmentation (with Clusters) Head': rfm_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster']].head(),
    'Cluster Summary (Mean RFM per Cluster)': cluster_summary,
    'Sentiment Analysis Results Head': df_customers[['TransactionID', 'CustomerID', 'Feedback', 'Sentiment_Score', 'Sentiment_Category']].head(),
    'Sentiment Distribution Counts': sentiment_counts,
    'RFM_Distribution_Histograms': fig1,
    'Customer_Segmentation_Scatter_Plot': fig2,
    'Sentiment_Distribution_Bar_Chart': fig3,
}

output # Output the dictionary
```

**Explanation:**

*   We load the dummy transaction data. **Remember to replace `"CustomerTransactions"`**.
*   We calculate Recency, Frequency, and Monetary (RFM) values for each customer using `groupby()` and aggregation. Recency is calculated relative to a defined `snapshot_date`.
*   We prepare the RFM data for clustering by scaling the values. Scaling is important for K-Means as it's based on distances. We use `StandardScaler`.
*   We apply K-Means clustering with a predefined number of clusters (`n_clusters = 4`) to segment customers based on their scaled RFM values.
*   We calculate the mean RFM values for each cluster to understand the characteristics of each segment.
*   We perform sentiment analysis on the 'Feedback' column using `nltk.sentiment.vader`. This assigns a compound score to each feedback string. We then classify the score into 'Positive', 'Negative', 'Neutral', or 'No Feedback'.
*   We count the occurrences of each sentiment category.
*   **Visualization:**
    *   `fig1`: Three histograms showing the distribution of Recency, Frequency, and Monetary values across all customers.
    *   `fig2`: A scatter plot showing customers based on their Frequency and Monetary values, with points colored according to their assigned cluster. This helps visualize the segmentation.
    *   `fig3`: A horizontal bar chart showing the count of feedbacks in each sentiment category. Data labels are included.
*   **Custom Style:** Applied the specified style guidelines (font, colors - using different colors for each RFM histogram and mapping colors to sentiment categories, axes, spines, grid, data labels, negative number format - although not applicable here).
*   We return a dictionary containing heads of calculated DataFrames/Series (RFM, Segmentation, Sentiment), the Cluster Summary, Sentiment Counts, and the three plot figures.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames/Series ('RFM Metrics Head', 'Customer Segmentation (with Clusters) Head', 'Cluster Summary (Mean RFM per Cluster)', 'Sentiment Analysis Results Head', 'Sentiment Distribution Counts') to spill them into your sheet.
*   For each plot figure object ('RFM_Distribution_Histograms', 'Customer_Segmentation_Scatter_Plot', 'Sentiment_Distribution_Bar_Chart'), select "Picture in Cell" > "Create Reference" to see the plots.

This provides a good starting point for understanding your customer base. Would you like to proceed to the next category: **Data Cleaning & Preparation**, or perhaps focus on a specific aspect of Customer Analytics like predicting churn?