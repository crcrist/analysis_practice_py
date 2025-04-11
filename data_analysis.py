# Store Metrics Outlier Detection and Analysis

## 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('seaborn-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

## 2. Load the Data
# In a real scenario, you would load your data from a file or database
# For demonstration, let's create sample data that matches your description

def create_sample_data(n_rows=1000):
    """Create sample data for demonstration purposes."""
    stores = [1, 2, 3, 4]
    year_weeks = [202401, 202402, 202403, 202404, 202405]
    metrics = ['sales', 'visits', 'units_sold', 'items_per_person']
    
    data = []
    for store in stores:
        for yw in year_weeks:
            # Add regular metrics (sales, visits, units_sold)
            for metric in metrics[:3]:
                base_value = np.random.randint(5000, 40000)
                # Add some store-specific variation to create patterns
                if store == 1:
                    base_value *= 1.2  # Store 1 has higher values
                elif store == 4:
                    base_value *= 0.7  # Store 4 has lower values
                
                data.append({
                    'store': store,
                    'year_week': yw,
                    'metric': metric,
                    'numerator': base_value,
                    'denominator': None,
                    'isratio': False
                })
            
            # Add ratio metric (items_per_person)
            items = np.random.randint(20000, 50000)
            people = np.random.randint(5000, 15000)
            data.append({
                'store': store,
                'year_week': yw,
                'metric': metrics[3],  # items_per_person
                'numerator': items,
                'denominator': people,
                'isratio': True
            })
    
    # Add some outliers for demonstration
    # Make store 3 have an unusually high sales value in one week
    outlier_idx = next(i for i, d in enumerate(data) 
                      if d['store'] == 3 and d['year_week'] == 202403 and d['metric'] == 'sales')
    data[outlier_idx]['numerator'] = 90000
    
    # Make store 2 have an unusually low visits value in one week
    outlier_idx = next(i for i, d in enumerate(data) 
                      if d['store'] == 2 and d['year_week'] == 202402 and d['metric'] == 'visits')
    data[outlier_idx]['numerator'] = 2000
    
    return pd.DataFrame(data)

# Create and display the sample data
df = create_sample_data()
print(f"Sample data shape: {df.shape}")
df.head()

## 3. Exploratory Data Analysis

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Get distinct metrics
distinct_metrics = df['metric'].unique()
print(f"\nDistinct metrics in the dataset: {distinct_metrics}")

# Basic statistics for the dataframe
print("\nBasic statistics:")
print(df.describe())

## 4. Aggregate Metrics by Store

def aggregate_metrics(df):
    """
    Aggregate metrics by store, handling both regular and ratio metrics differently.
    
    For regular metrics (isratio=False), we'll simply calculate the sum and average.
    For ratio metrics (isratio=True), we'll sum numerator and denominator first, then calculate the ratio.
    """
    # Initialize empty dataframe to store results
    agg_results = pd.DataFrame()
    
    # Get distinct metrics
    metrics = df['metric'].unique()
    
    for metric in metrics:
        # Filter data for this metric
        metric_data = df[df['metric'] == metric]
        
        # Check if this is a ratio metric
        is_ratio = metric_data['isratio'].iloc[0]
        
        if is_ratio:
            # For ratio metrics, sum numerator and denominator first, then calculate ratio
            agg = metric_data.groupby('store').agg({
                'numerator': 'sum',
                'denominator': 'sum'
            }).reset_index()
            
            # Calculate the ratio
            agg[metric] = agg['numerator'] / agg['denominator']
            agg = agg[['store', metric]]
        else:
            # For regular metrics, take the sum
            agg = metric_data.groupby('store').agg({
                'numerator': 'sum'
            }).reset_index()
            
            # Rename column to the metric name
            agg.columns = ['store', metric]
        
        # Merge with results if not empty
        if agg_results.empty:
            agg_results = agg
        else:
            agg_results = agg_results.merge(agg, on='store', how='outer')
    
    return agg_results

# Aggregate metrics by store
aggregated_df = aggregate_metrics(df)
print("\nAggregated metrics by store:")
aggregated_df

## 5. Identify Outliers

def identify_outliers(df, metric, threshold=1.5):
    """
    Identify outliers for a given metric using IQR method.
    Returns a dataframe with outliers marked.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the data
    metric : str
        Name of the metric column to analyze
    threshold : float, default=1.5
        Threshold multiplier for IQR to identify outliers
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with outliers marked
    """
    # Calculate Q1, Q3, and IQR
    Q1 = df[metric].quantile(0.25)
    Q3 = df[metric].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Create a copy of the dataframe with outlier flag
    result_df = df.copy()
    result_df[f'{metric}_outlier'] = 'normal'
    
    # Mark outliers
    result_df.loc[result_df[metric] < lower_bound, f'{metric}_outlier'] = 'low outlier'
    result_df.loc[result_df[metric] > upper_bound, f'{metric}_outlier'] = 'high outlier'
    
    return result_df

# Identify outliers for each metric
outlier_results = aggregated_df.copy()
for metric in distinct_metrics:
    outlier_results = identify_outliers(outlier_results, metric)

print("\nOutlier detection results:")
outlier_results

## 6. Visualize Outliers

def plot_metric_by_store(df, metric):
    """Plot a metric across stores, highlighting outliers."""
    plt.figure(figsize=(10, 6))
    
    # Create a bar plot
    ax = sns.barplot(x='store', y=metric, data=df)
    
    # Highlight outliers with different colors
    for i, store in enumerate(df['store']):
        bar = ax.patches[i]
        outlier_status = df.loc[df['store'] == store, f'{metric}_outlier'].iloc[0]
        
        if outlier_status == 'high outlier':
            bar.set_facecolor('red')
        elif outlier_status == 'low outlier':
            bar.set_facecolor('blue')
    
    # Add labels and title
    plt.xlabel('Store')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} by Store (Outliers Highlighted)')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4c72b0', label='Normal'),
        Patch(facecolor='red', label='High Outlier'),
        Patch(facecolor='blue', label='Low Outlier')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()

# Plot each metric by store
for metric in distinct_metrics:
    plot_metric_by_store(outlier_results, metric)

## 7. Correlation Analysis

# Create a correlation matrix for all metrics
correlation_matrix = aggregated_df.drop('store', axis=1).corr()

print("\nCorrelation Matrix:")
correlation_matrix

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Store Metrics')
plt.tight_layout()
plt.show()

## 8. Summary of Results

# Summarize top and bottom performing stores for each metric
summary = pd.DataFrame(columns=['Metric', 'Top Performer (Store)', 'Bottom Performer (Store)'])

for metric in distinct_metrics:
    metric_data = aggregated_df[['store', metric]].sort_values(by=metric, ascending=False)
    top_store = metric_data['store'].iloc[0]
    bottom_store = metric_data['store'].iloc[-1]
    
    summary = pd.concat([summary, pd.DataFrame({
        'Metric': [metric],
        'Top Performer (Store)': [top_store],
        'Bottom Performer (Store)': [bottom_store]
    })], ignore_index=True)

print("\nPerformance Summary:")
summary

## 9. Additional Analysis: Week-over-Week Trends

# This section shows how you could extend the analysis to look at week-over-week trends

def analyze_weekly_trends(df, metric_to_analyze):
    """Analyze week-over-week trends for a specific metric."""
    # Filter data for the specific metric
    metric_data = df[df['metric'] == metric_to_analyze]
    
    # If it's a ratio metric, calculate the ratio
    if metric_data['isratio'].iloc[0]:
        # Group by store and year_week, then calculate ratio
        weekly_data = metric_data.groupby(['store', 'year_week']).agg({
            'numerator': 'sum',
            'denominator': 'sum'
        }).reset_index()
        
        weekly_data[metric_to_analyze] = weekly_data['numerator'] / weekly_data['denominator']
        weekly_data = weekly_data[['store', 'year_week', metric_to_analyze]]
    else:
        # For regular metrics, just sum the numerator
        weekly_data = metric_data.groupby(['store', 'year_week']).agg({
            'numerator': 'sum'
        }).reset_index()
        
        weekly_data.columns = ['store', 'year_week', metric_to_analyze]
    
    # Pivot the data for plotting
    pivot_data = weekly_data.pivot(index='year_week', columns='store', values=metric_to_analyze)
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    pivot_data.plot(marker='o')
    plt.title(f'Weekly {metric_to_analyze.capitalize()} by Store')
    plt.xlabel('Year-Week')
    plt.ylabel(metric_to_analyze.capitalize())
    plt.grid(True)
    plt.legend(title='Store')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return pivot_data

# Example: Analyze weekly trends for sales
sales_weekly = analyze_weekly_trends(df, 'sales')
print("\nWeekly sales data:")
sales_weekly

## 10. Save Results to CSV (commented out, uncomment to use)

# Save the aggregated metrics
# aggregated_df.to_csv('aggregated_metrics.csv', index=False)

# Save the outlier results
# outlier_results.to_csv('outlier_results.csv', index=False)

# Save the summary
# summary.to_csv('performance_summary.csv', index=False)

print("\nAnalysis complete!")
