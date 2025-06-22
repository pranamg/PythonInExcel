The next topic in the business intelligence series is **Business Intelligence - 2. Marketing Analytics**.

Marketing analytics is a data-driven approach to measuring and analyzing marketing campaign performance, understanding customer behavior influenced by marketing efforts, and optimizing strategies based on quantitative insights.

Based on [`piplist.txt`](./README.md) output, you should have `pandas` for data handling, `numpy` for calculations, `scipy.stats` which is great for statistical tests needed for A/B testing, `matplotlib` and `seaborn` for visualization, and `Faker` for dummy data generation. We are well-equipped for this.

**Step 1: Generate Sample Marketing Campaign and A/B Test Data**

We'll create dummy data representing marketing campaign events (impressions, clicks, conversions) and include a column for A/B testing variants.

In a new Excel cell, enter `=PY` and paste the following code, then press **Ctrl+Enter**:

```python
# Generate dummy marketing campaign and A/B test data
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta

fake = Faker()

num_users = 10000
num_campaigns = 5
start_date = date(2023, 10, 1)
end_date = date(2024, 3, 31)

# Generate dummy campaigns
campaigns = {f'Campaign_{i}': {'channel': random.choice(['Email', 'Social Media', 'Display', 'Search']),
                               'cost': random.randint(1000, 10000),
                               'conversion_value_avg': random.uniform(20, 100)} # Average value per conversion
             for i in range(num_campaigns)}
campaign_ids = list(campaigns.keys())

data = []

# Simulate user interactions
for user_id in range(num_users):
    user_fake_id = fake.uuid4() # Unique ID for simulation
    # Assign user to an A/B test variant (assuming one test is running across users)
    ab_variant = random.choice(['A', 'B']) # Variant A or B

    # Simulate interactions over several days for some users
    num_interaction_days = random.randint(1, 10) if random.random() < 0.3 else 1 # Some users interact on multiple days

    for _ in range(num_interaction_days):
        date_offset = random.randint(0, (end_date - start_date).days)
        event_date = start_date + timedelta(days=date_offset)

        # Simulate interactions across campaigns (user might see multiple)
        viewed_campaigns = random.sample(campaign_ids, random.randint(1, min(3, num_campaigns)))

        for campaign_id in viewed_campaigns:
            # Simulate Impression
            data.append([user_fake_id, campaign_id, campaigns[campaign_id]['channel'], event_date, 'Impression', 0, ab_variant])

            # Simulate Click (higher probability if impressed)
            if random.random() < 0.15: # 15% Click-through rate possibility
                data.append([user_fake_id, campaign_id, campaigns[campaign_id]['channel'], event_date, 'Click', 0, ab_variant])

                # Simulate Conversion (higher probability if clicked)
                # Slightly different conversion rate for Variant B for A/B test
                conversion_prob = 0.05 if ab_variant == 'A' else 0.06 # A/B test: B has slightly higher conversion rate
                if random.random() < conversion_prob:
                     conversion_value = round(np.random.normal(campaigns[campaign_id]['conversion_value_avg'], campaigns[campaign_id]['conversion_value_avg']*0.2), 2) # Simulate value with some variance
                     conversion_value = max(1.0, conversion_value) # Ensure positive value
                     data.append([user_fake_id, campaign_id, campaigns[campaign_id]['channel'], event_date, 'Conversion', conversion_value, ab_variant])


df_marketing = pd.DataFrame(data, columns=['UserID', 'CampaignID', 'Channel', 'Date', 'EventType', 'EventValue', 'AB_Variant'])

# Ensure Date column is datetime
df_marketing['Date'] = pd.to_datetime(df_marketing['Date'])

df_marketing # Output the DataFrame
```

**Explanation:**

*   This code generates a DataFrame `df_marketing` with simulated user interactions across different marketing campaigns.
*   Each row represents an event ('Impression', 'Click', or 'Conversion').
*   It includes a `AB_Variant` column to simulate data from an A/B test.
*   `EventValue` stores the conversion value if the event is 'Conversion'.
*   `Faker` is used to generate unique User IDs.
*   Campaign costs and average conversion values are stored in a dictionary `campaigns` for later ROI calculation.
*   The result, `df_marketing`, will be spilled into your Excel sheet. Let's assume this data is placed in a range or Table named `MarketingData`.

**Step 2: Analyze Campaigns, Perform A/B Test, Calculate ROI, and Visualize**

Now, we'll analyze overall campaign performance, conduct a simple A/B test comparison on conversion rates, calculate ROI, and visualize key metrics.

In a **new** Excel cell, enter `=PY` and paste the following code. Replace `"MarketingData"` with the actual name of the Excel range/Table where your dummy data is. Press **Ctrl+Enter**. Also, update the `campaign_costs` dictionary within the code block to match the costs used in the data generation step (or load them from Excel if they were added to the data).

```python
# Analyze campaigns, perform A/B test, calculate ROI, and visualize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency # For A/B test

# Load the marketing data from Excel
# IMPORTANT: Replace "MarketingData" with the actual name of your Excel range or Table
df_marketing = xl("MarketingData[#All]", headers=True)

# Ensure Date column is datetime
df_marketing['Date'] = pd.to_datetime(df_marketing['Date'])

# --- Campaign Performance Metrics ---

# Calculate Impressions, Clicks, Conversions, and Total Value per Campaign
campaign_summary = df_marketing.groupby('CampaignID').agg(
    Impressions=('EventType', lambda x: (x == 'Impression').sum()),
    Clicks=('EventType', lambda x: (x == 'Click').sum()),
    Conversions=('EventType', lambda x: (x == 'Conversion').sum()),
    Total_Conversion_Value=('EventValue', 'sum')
).reset_index()

# Calculate CTR and Conversion Rate
# Add epsilon to avoid division by zero
epsilon = 1e-9
campaign_summary['CTR'] = (campaign_summary['Clicks'] / (campaign_summary['Impressions'] + epsilon)) * 100
campaign_summary['Conversion_Rate'] = (campaign_summary['Conversions'] / (campaign_summary['Clicks'] + epsilon)) * 100 # Click-based Conversion Rate

# --- ROI Calculation ---

# Define campaign costs (MUST match the costs used in data generation Step 1 or load from Excel)
# Assuming costs were NOT loaded with the data, define them here based on Step 1's logic
# In a real scenario, load costs from a separate Excel range/table if stored there
campaign_costs = {
    'Campaign_0': 7411, # Example cost - UPDATE these values to match your data generation exactly!
    'Campaign_1': 9876,
    'Campaign_2': 2345,
    'Campaign_3': 5678,
    'Campaign_4': 8765
}

# Map costs to the summary table
campaign_summary['Cost'] = campaign_summary['CampaignID'].map(campaign_costs)

# Calculate ROI: (Total Conversion Value - Cost) / Cost * 100
campaign_summary['ROI'] = ((campaign_summary['Total_Conversion_Value'] - campaign_summary['Cost']) / (campaign_summary['Cost'] + epsilon)) * 100


# --- A/B Test Analysis (Comparing Conversion Rates of Variant A vs B) ---

# Filter data for Click and Conversion events, group by AB_Variant
ab_data = df_marketing[df_marketing['EventType'].isin(['Click', 'Conversion'])]

# Count Clicks and Conversions per variant
ab_summary = ab_data.groupby('AB_Variant').agg(
    Clicks=('EventType', lambda x: (x == 'Click').sum()),
    Conversions=('EventType', lambda x: (x == 'Conversion').sum())
).reset_index()

# Calculate Conversion Rates for A and B
ab_summary['Conversion_Rate'] = (ab_summary['Conversions'] / (ab_summary['Clicks'] + epsilon)) * 100

# Perform Chi-Squared Test for independence of Conversion outcome vs Variant
# Need a contingency table:
#                 Converted | Not Converted
# Variant A:      [A_Conversions] | [A_Clicks - A_Conversions]
# Variant B:      [B_Conversions] | [B_Clicks - B_Conversions]

if len(ab_summary) == 2 and 'A' in ab_summary['AB_Variant'].values and 'B' in ab_summary['AB_Variant'].values:
    a_conversions = ab_summary[ab_summary['AB_Variant'] == 'A']['Conversions'].iloc[0]
    a_clicks = ab_summary[ab_summary['AB_Variant'] == 'A']['Clicks'].iloc[0]
    b_conversions = ab_summary[ab_summary['AB_Variant'] == 'B']['Conversions'].iloc[0]
    b_clicks = ab_summary[ab_summary['AB_Variant'] == 'B']['Clicks'].iloc[0]

    contingency_table = np.array([
        [a_conversions, a_clicks - a_conversions],
        [b_conversions, b_clicks - b_conversions]
    ])

    # Perform chi-squared test
    # chi2, p, dof, expected = chi2_contingency(contingency_table) # expected is not needed for output
    chi2, p, dof, _ = chi2_contingency(contingency_table)

    ab_test_result = {
        'Variant Summary': ab_summary,
        'Chi-Squared Stat': chi2,
        'P-value': p,
        'Degrees of Freedom': dof,
        'Significance': 'Statistically Significant (p < 0.05)' if p < 0.05 else 'Not Statistically Significant (p >= 0.05)'
    }
else:
    ab_test_result = "A/B test data not in expected format (requires Variants 'A' and 'B')"


# --- Visualization ---

# Apply custom style guidelines
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#1a1a24'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.grid'] = False # Turn off default grid
sns.set_theme(style="whitegrid") # Use a seaborn theme base, then apply customs


# 1. Campaign Conversion Rate and ROI Bar Chart (on separate y-axes if needed, or separate charts)
# Let's do two separate bar charts for clarity
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6)) # Two subplots side by side

# Plot Conversion Rate
sns.barplot(x='CampaignID', y='Conversion_Rate', hue='CampaignID', legend=False, data=campaign_summary.sort_values('Conversion_Rate', ascending=False), ax=ax1a, palette=sns.color_palette("viridis", len(campaign_summary)))
ax1a.set_title('Campaign Conversion Rate (%)', fontsize=14, color='#1a1a24')
ax1a.set_xlabel('Campaign ID', fontsize=12, color='#1a1a24')
ax1a.set_ylabel('Conversion Rate (%)', fontsize=12, color='#1a1a24')
sns.despine(ax=ax1a, top=True, right=True)
ax1a.grid(False)
# Add data labels
for container in ax1a.containers:
    ax1a.bar_label(container, fmt='%.1f%%', fontsize=9, color='#1a1a24')


# Plot ROI
sns.barplot(x='CampaignID', y='ROI', hue='CampaignID', legend=False, data=campaign_summary.sort_values('ROI', ascending=False), ax=ax1b, palette=sns.color_palette("plasma", len(campaign_summary)))
ax1b.set_title('Campaign ROI (%)', fontsize=14, color='#1a1a24')
ax1b.set_xlabel('Campaign ID', fontsize=12, color='#1a1a24')
ax1b.set_ylabel('ROI (%)', fontsize=12, color='#1a1a24')
sns.despine(ax=ax1b, top=True, right=True)
ax1b.grid(False)
# Add data labels
for container in ax1b.containers:
    ax1b.bar_label(container, fmt='%.1f%%', fontsize=9, color='#1a1a24')


plt.tight_layout()


# 2. A/B Test Conversion Rate Comparison Bar Chart
fig2, ax2 = plt.subplots(figsize=(8, 5))

if isinstance(ab_test_result, dict) and 'Variant Summary' in ab_test_result:
    sns.barplot(x='AB_Variant', y='Conversion_Rate', hue='AB_Variant', legend=False, data=ab_test_result['Variant Summary'], ax=ax2, palette=['#188ce5', '#ff6d00']) # Blue, Orange

    ax2.set_title('A/B Test Conversion Rate Comparison', fontsize=14, color='#1a1a24')
    ax2.set_xlabel('Variant', fontsize=12, color='#1a1a24')
    ax2.set_ylabel('Conversion Rate (%)', fontsize=12, color='#1a1a24')
    sns.despine(ax=ax2, top=True, right=True)
    ax2.grid(False)
    ax2.set_ylim(0, ab_test_result['Variant Summary']['Conversion_Rate'].max() * 1.2) # Give some space above bars
    # Add data labels
    for container in ax2.containers:
         ax2.bar_label(container, fmt='%.1f%%', fontsize=10, color='#1a1a24')

    # Add significance note
    sig_note = ab_test_result.get('Significance', '')
    if sig_note:
        # Position text above the bars, adjust coordinates as needed
        max_y = ax2.get_ylim()[1]
        ax2.text(0.5, max_y * 0.95, f'Statistical Test Result: {sig_note}\n(p-value: {ab_test_result["P-value"]:.3f})',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='#1a1a24', transform=ax2.transAxes, wrap=True)

else:
    ax2.text(0.5, 0.5, "A/B Test Plot N/A\n(Data not in expected format)",
             horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray', transform=ax2.transAxes)
    ax2.set_title('A/B Test Conversion Rate Comparison', fontsize=14, color='#1a1a24')
    ax2.set_xlabel('Variant', fontsize=12, color='#1a1a24')
    ax2.set_ylabel('Conversion Rate (%)', fontsize=12, color='#1a1a24')
    sns.despine(ax=ax2, top=True, right=True)
    ax2.grid(False)


plt.tight_layout()


# Output results
output = {
    'Campaign Summary (Metrics & ROI)': campaign_summary,
    'AB Test Result': ab_test_result, # This includes the variant summary and test stats
    'Campaign_Performance_Plots': fig1,
    'AB_Test_Conversion_Rate_Plot': fig2,
}

output # Output the dictionary
```

**Important Note:** You need to make sure the `campaign_costs` dictionary inside the analysis code matches the costs assumed when generating the dummy data in Step 1, or modify the analysis code to load costs from an Excel range if you add them there.

**Explanation:**

*   We load the dummy marketing data. **Remember to replace `"MarketingData"`**.
*   We group the data by `CampaignID` to count impressions, clicks, and conversions, and sum the conversion values.
*   We calculate Click-Through Rate (CTR) and Conversion Rate (Conversions per Click).
*   We define (or would load) the cost for each campaign and calculate ROI.
*   For the A/B test, we filter for relevant events ('Click', 'Conversion') and group by `AB_Variant`.
*   We use `scipy.stats.chi2_contingency` to perform a chi-squared test to see if there is a statistically significant difference in conversion rates between Variant A and Variant B. The p-value from this test helps determine significance.
*   **Visualization:**
    *   `fig1`: Two bar charts side-by-side showing Campaign Conversion Rate and Campaign ROI.
    *   `fig2`: A bar chart comparing the Conversion Rates of the A/B test variants. It also adds a note about the statistical significance result based on the chi-squared test p-value.
*   **Custom Style:** Applied the specified style guidelines (font, colors, axes, spines, grid, data labels). Specific colors from the palette are used.
*   We return a dictionary containing the `campaign_summary` DataFrame, the `ab_test_result` dictionary (which includes the variant summary and stats), and the two plot figures.

**Viewing the Output:**

*   Click the Python cell, then click the Python icon/button next to the formula bar.
*   Select "Excel Value" (**Ctrl+Shift+Alt+M**) for the DataFrames/dictionaries ('Campaign Summary (Metrics & ROI)', 'AB Test Result') to spill them into your sheet. Note that the AB Test Result is a dictionary containing a DataFrame and scalar values; you might need to access elements of this dictionary in separate cells if you want specific values like the p-value spilled directly.
*   For each plot figure object ('Campaign_Performance_Plots', 'AB_Test_Conversion_Rate_Plot'), select "Picture in Cell" > "Create Reference" to see the plots.

**Further Analysis:**

Here are some advanced marketing analytics techniques you could explore:

1. **Advanced A/B Testing:**
   - Implement multivariate testing (A/B/n)
   - Add Bayesian A/B testing methods
   - Create sequential testing frameworks

2. **Attribution Modeling:**
   - Implement multi-touch attribution
   - Add time-decay models
   - Create custom attribution rules

3. **Campaign Optimization:**
   - Add predictive campaign performance
   - Implement budget optimization
   - Create automated bidding strategies

4. **Customer Journey Analysis:**
   - Create funnel visualization tools
   - Implement path analysis
   - Add touchpoint optimization

5. **Marketing Mix Modeling:**
   - Implement media mix optimization
   - Add cross-channel attribution
   - Create budget allocation tools

This section provides a comprehensive overview of common Marketing Analytics tasks. The next topic in the series is [Business Intelligence - Customer Analytics](./02-Business%20Intelligence_03-Customer%20Analytics.md), which explores techniques for analyzing customer behavior and segmentation.