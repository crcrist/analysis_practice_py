# Simplified Store Metrics Analysis for Large Scale Data
# Optimized for 5000+ stores and 50+ metrics

## 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Set basic figure size
plt.rcParams['figure.figsize'] = (12, 8)

## 2. Data Loading and Optimization

def optimize_dataframe(df):
    """Optimize dataframe memory usage by setting appropriate dtypes."""
    # Convert numeric columns to smaller types
    df['store'] = df['store'].astype('int32')
    df['year_week'] = df['year_week'].astype('int32')
    # Convert string columns to categories
    df['metric'] = df['metric'].astype('category')
    # Convert boolean columns
    df['isratio'] = df['isratio'].astype('bool')
    # For numeric columns that can be integers
    if 'numerator' in df.columns:
        df['numerator'] = pd.to_numeric(df['numerator'], downcast='integer')
    if 'denominator' in df.columns:
        # Keep NaN values but convert valid values to integers
        df['denominator'] = pd.to_numeric(df['denominator'], downcast='integer')
    
    # Print memory savings
    print(f"Optimized dataframe memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    return df

# Load your data from CSV file or database
# Example:
# df = pd.read_csv('your_metrics_data.csv')
df = pd.read_csv('your_metrics_data.csv')  # Replace with your actual data source

# Optimize the dataframe to reduce memory usage
df = optimize_dataframe(df)

print(f"Data shape: {df.shape}")
print(f"Number of stores: {df['store'].nunique()}")
print(f"Number of metrics: {df['metric'].nunique()}")
print(f"Number of time periods: {df['year_week'].nunique()}")
df.head()

## 3. Exploratory Data Analysis

def perform_eda(df):
    """Perform exploratory data analysis on large scale data."""
    start_time = time.time()
    
    # Get distinct metrics
    distinct_metrics = df['metric'].unique()
    print(f"Found {len(distinct_metrics)} distinct metrics in the dataset")
    
    # Check for missing values in important columns
    print("\nMissing values in key columns:")
    print(df[['store', 'year_week', 'metric', 'numerator']].isnull().sum())
    
    # For ratio metrics, check denominator nulls
    ratio_metrics = df[df['isratio'] == True]['metric'].unique()
    if len(ratio_metrics) > 0:
        print(f"\nRatio metrics in dataset: {ratio_metrics}")
        denominator_nulls = df[df['isratio'] == True]['denominator'].isnull().sum()
        print(f"Missing denominators in ratio metrics: {denominator_nulls}")
    
    # Basic statistics on data volume
    store_count = df['store'].nunique()
    week_count = df['year_week'].nunique()
    print(f"\nDataset contains {store_count} stores across {week_count} weeks")
    
    # Sample distribution of one numeric metric
    sample_metric = distinct_metrics[0]
    sample_data = df[df['metric'] == sample_metric]
    
    print(f"\nSample distribution statistics for '{sample_metric}':")
    
    if sample_data['isratio'].iloc[0]:
        # Calculate the ratio for ratio metrics
        sample_data['value'] = sample_data['numerator'] / sample_data['denominator']
    else:
        sample_data['value'] = sample_data['numerator']
    
    print(sample_data['value'].describe())
    
    end_time = time.time()
    print(f"EDA completed in {end_time - start_time:.2f} seconds")

# Perform EDA
perform_eda(df)

## 4. Efficient Metric Aggregation

def aggregate_metrics_by_store(df, process_in_chunks=True, chunk_size=1000):
    """
    Aggregate metrics by store efficiently, handling large datasets.
    Process in chunks to avoid memory issues.
    """
    start_time = time.time()
    print("Aggregating metrics by store...")
    
    # Get distinct metrics
    metrics = df['metric'].unique()
    
    # Initialize result dataframe with store as index
    all_stores = df['store'].unique()
    results = pd.DataFrame({'store': all_stores})
    results.set_index('store', inplace=True)
    
    # Process metrics
    for metric in metrics:
        print(f"Processing metric: {metric}")
        
        # Filter data for this metric
        metric_data = df[df['metric'] == metric].copy()
        
        # Check if this is a ratio metric
        is_ratio = metric_data['isratio'].iloc[0]
        
        if process_in_chunks and len(metric_data) > chunk_size:
            # Process in chunks for large datasets
            if is_ratio:
                # Pre-aggregate numerator and denominator by store
                num_denom = metric_data.groupby('store').agg({
                    'numerator': 'sum',
                    'denominator': 'sum'
                })
                # Calculate the ratio
                results[metric] = num_denom['numerator'] / num_denom['denominator']
            else:
                # For regular metrics, just sum the numerator
                results[metric] = metric_data.groupby('store')['numerator'].sum()
        else:
            # Small enough to process at once
            if is_ratio:
                # Pre-aggregate numerator and denominator by store
                num_denom = metric_data.groupby('store').agg({
                    'numerator': 'sum',
                    'denominator': 'sum'
                })
                # Calculate the ratio
                results[metric] = num_denom['numerator'] / num_denom['denominator']
            else:
                # For regular metrics, just sum the numerator
                results[metric] = metric_data.groupby('store')['numerator'].sum()
    
    # Reset index to make store a column
    results = results.reset_index()
    
    end_time = time.time()
    print(f"Aggregation completed in {end_time - start_time:.2f} seconds")
    
    return results

# Aggregate metrics by store
aggregated_df = aggregate_metrics_by_store(df)
print(f"Aggregated dataframe shape: {aggregated_df.shape}")
aggregated_df.head()

## 5. Store Performance Analysis

def identify_store_performance(df, metrics, percentile_threshold=10):
    """
    Identify consistently high and low performing stores across multiple metrics.
    
    Parameters:
    -----------
    df : DataFrame with stores as rows and metrics as columns
    metrics : List of metric column names to analyze
    percentile_threshold : Threshold for considering a store as top/bottom performer
    
    Returns:
    --------
    DataFrame with store performance summary
    """
    print(f"Analyzing store performance using {percentile_threshold}% percentile threshold...")
    
    # Calculate percentile ranks for each metric
    percentile_df = df.copy()
    for metric in metrics:
        percentile_df[f'{metric}_percentile'] = df[metric].rank(pct=True) * 100
    
    # Count how many metrics each store is in the top or bottom percentile
    store_performance = pd.DataFrame({'store': df['store']})
    
    # Count top performances
    top_count = 0
    for metric in metrics:
        top_mask = percentile_df[f'{metric}_percentile'] >= (100 - percentile_threshold)
        store_performance[f'top_{metric}'] = top_mask
        top_count += top_mask
    
    store_performance['top_metric_count'] = top_count
    
    # Count bottom performances
    bottom_count = 0
    for metric in metrics:
        bottom_mask = percentile_df[f'{metric}_percentile'] <= percentile_threshold
        store_performance[f'bottom_{metric}'] = bottom_mask
        bottom_count += bottom_mask
    
    store_performance['bottom_metric_count'] = bottom_count
    
    # Calculate a simple performance score (-1 for each bottom, +1 for each top)
    store_performance['performance_score'] = store_performance['top_metric_count'] - store_performance['bottom_metric_count']
    
    return store_performance

# Get list of metrics (exclude store column)
metrics_list = [col for col in aggregated_df.columns if col != 'store']

# Analyze store performance
performance_df = identify_store_performance(aggregated_df, metrics_list)

# Find consistently high performers
high_performers = performance_df.sort_values('top_metric_count', ascending=False).head(20)
print("\nTop 20 consistently high-performing stores:")
print(high_performers[['store', 'top_metric_count', 'bottom_metric_count', 'performance_score']])

# Find consistently low performers
low_performers = performance_df.sort_values('bottom_metric_count', ascending=False).head(20)
print("\nTop 20 consistently low-performing stores:")
print(low_performers[['store', 'bottom_metric_count', 'top_metric_count', 'performance_score']])

## 6. Enhanced Outlier Detection

def identify_outliers_efficiently(df, metrics, threshold=1.5):
    """
    Identify outliers for all metrics efficiently.
    Returns a dataframe with outlier flags added.
    """
    start_time = time.time()
    print("Identifying outliers...")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Find outliers based on IQR method
    outlier_counts = {'high': 0, 'low': 0}
    
    for metric in metrics:
        # Calculate IQR
        Q1 = result_df[metric].quantile(0.25)
        Q3 = result_df[metric].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Mark outliers
        result_df[f'{metric}_outlier'] = 'normal'
        
        # Low outliers
        low_mask = result_df[metric] < lower_bound
        result_df.loc[low_mask, f'{metric}_outlier'] = 'low outlier'
        outlier_counts['low'] += low_mask.sum()
        
        # High outliers
        high_mask = result_df[metric] > upper_bound
        result_df.loc[high_mask, f'{metric}_outlier'] = 'high outlier'
        outlier_counts['high'] += high_mask.sum()
    
    # Summary of outliers
    total_cells = len(result_df) * len(metrics)
    total_outliers = outlier_counts['high'] + outlier_counts['low']
    
    print(f"Found {outlier_counts['high']} high outliers and {outlier_counts['low']} low outliers")
    print(f"Total outliers: {total_outliers} ({total_outliers/total_cells:.2%} of all values)")
    
    end_time = time.time()
    print(f"Outlier detection completed in {end_time - start_time:.2f} seconds")
    
    return result_df

# Identify outliers
outlier_results = identify_outliers_efficiently(aggregated_df, metrics_list)

## 7. Visualizations for Store Performance

def plot_top_bottom_stores(df, metric, n=10):
    """Plot top and bottom n performing stores for a given metric."""
    # Sort and get top/bottom n stores
    sorted_df = df.sort_values(by=metric, ascending=False)
    top_n = sorted_df.head(n)
    bottom_n = sorted_df.tail(n)
    
    # Combine and plot
    plot_df = pd.concat([top_n, bottom_n])
    plot_df = plot_df.sort_values(by=metric, ascending=True)  # Sort for better display
    
    plt.figure(figsize=(14, 8))
    colors = ['red' if x == 'low outlier' else 'green' if x == 'high outlier' else 'blue' 
              for x in plot_df[f'{metric}_outlier']]
    
    plt.barh(plot_df['store'].astype(str), plot_df[metric], color=colors)
    plt.title(f'Top and Bottom {n} stores for {metric}')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Normal'),
        Patch(facecolor='green', label='High Outlier'),
        Patch(facecolor='red', label='Low Outlier')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()

def plot_performance_distribution(df):
    """Plot distribution of the performance score"""
    plt.figure(figsize=(12, 6))
    
    # Create histogram with KDE
    sns.histplot(df['performance_score'], kde=True)
    
    # Add vertical line for mean and median
    plt.axvline(df['performance_score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["performance_score"].mean():.2f}')
    plt.axvline(df['performance_score'].median(), color='green', linestyle='--',
                label=f'Median: {df["performance_score"].median():.2f}')
    
    plt.title('Distribution of Store Performance Scores')
    plt.xlabel('Performance Score (higher is better)')
    plt.ylabel('Number of Stores')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualize top/bottom stores for a sample metric
sample_metric = metrics_list[0]  # First metric
plot_top_bottom_stores(outlier_results, sample_metric, n=10)

# Visualize performance score distribution
plot_performance_distribution(performance_df)

## 8. Correlation Analysis

def analyze_correlations(df, metrics):
    """Analyze correlations between metrics."""
    print("Analyzing correlations between metrics...")
    
    # Calculate correlation matrix
    corr = df[metrics].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of Store Metrics')
    plt.tight_layout()
    plt.show()
    
    # Find the most correlated pairs
    corr_pairs = []
    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            corr_value = corr.iloc[i, j]
            if abs(corr_value) > 0.5:  # Only show strong correlations
                corr_pairs.append((metrics[i], metrics[j], corr_value))
    
    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Show top correlations
    print("\nStrongest metric correlations:")
    for metric1, metric2, corr_value in corr_pairs[:10]:  # Show top 10
        direction = "positive" if corr_value > 0 else "negative"
        print(f"{metric1} and {metric2}: {corr_value:.3f} ({direction})")
    
    return corr

# Analyze correlations
correlation_matrix = analyze_correlations(aggregated_df, metrics_list)

## 9. Generate Summary Report

def generate_summary_report(df, metrics, outlier_df, performance_df, n_top_bottom=5):
    """Generate a compact summary report with insights."""
    print("Generating summary report...")
    
    # Initialize summary dataframe
    summary = pd.DataFrame(columns=['Metric', 'Mean', 'Median', 'Std Dev', 
                                   'Top 5 Stores', 'Bottom 5 Stores',
                                   'Outlier Count'])
    
    # Calculate correlations once
    corr = df[metrics].corr()
    
    for metric in metrics:
        # Calculate statistics
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        std_val = df[metric].std()
        
        # Get top and bottom stores
        sorted_df = df[['store', metric]].sort_values(by=metric, ascending=False)
        top_stores = sorted_df.head(n_top_bottom)['store'].tolist()
        bottom_stores = sorted_df.tail(n_top_bottom)['store'].tolist()
        
        # Count outliers
        outlier_count = outlier_df[outlier_df[f'{metric}_outlier'] != 'normal'].shape[0]
        
        # Add to summary
        summary = pd.concat([summary, pd.DataFrame({
            'Metric': [metric],
            'Mean': [mean_val],
            'Median': [median_val],
            'Std Dev': [std_val],
            'Top 5 Stores': [top_stores],
            'Bottom 5 Stores': [bottom_stores],
            'Outlier Count': [outlier_count]
        })], ignore_index=True)
    
    # Add overall top/bottom performers from performance analysis
    print("\nOverall Top Performing Stores:")
    top_overall = performance_df.sort_values('performance_score', ascending=False).head(10)
    print(top_overall[['store', 'top_metric_count', 'bottom_metric_count', 'performance_score']])
    
    print("\nOverall Bottom Performing Stores:")
    bottom_overall = performance_df.sort_values('performance_score', ascending=True).head(10)
    print(bottom_overall[['store', 'top_metric_count', 'bottom_metric_count', 'performance_score']])
    
    return summary

# Generate summary report
summary_report = generate_summary_report(aggregated_df, metrics_list, outlier_results, performance_df)
print("\nMetric Performance Summary:")
summary_report

## 10. Save Results (Optional)

# Uncomment these lines to save results to CSV files
# print("Saving results to CSV files...")
# aggregated_df.to_csv('aggregated_metrics.csv', index=False)
# outlier_results.to_csv('outlier_results.csv', index=False)
# performance_df.to_csv('store_performance.csv', index=False)
# summary_report.to_csv('performance_summary.csv', index=False)

print("\nAnalysis complete!")
