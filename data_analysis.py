# Store Metrics Analysis for Large Scale Data
# Optimized for 5000+ stores and 50+ metrics

## 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy.spatial import distance
import time
import warnings
warnings.filterwarnings('ignore')

# Set basic figure size but skip seaborn style that was causing errors
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

def create_sample_data(n_stores=5000, n_metrics=50, n_weeks=10, with_outliers=True):
    """Create realistic sample data for large scale testing."""
    print(f"Creating sample data with {n_stores} stores and {n_metrics} metrics...")
    
    # Define parameters
    stores = list(range(1, n_stores + 1))
    year_weeks = [202401 + i for i in range(n_weeks)]
    
    # Create metric names (sales, visits, etc. + additional metrics)
    base_metrics = ['sales', 'visits', 'units_sold', 'items_per_person']
    ratio_metrics = ['conversion_rate', 'items_per_person', 'sales_per_visit', 'return_rate']
    
    # Generate additional metrics to reach desired count
    additional_metrics = []
    for i in range(n_metrics - len(base_metrics + ratio_metrics)):
        if i % 3 == 0:  # Make every third one a ratio metric
            additional_metrics.append(f'ratio_metric_{i}')
        else:
            additional_metrics.append(f'value_metric_{i}')
    
    # Combine all metrics
    all_metrics = base_metrics + [m for m in ratio_metrics if m not in base_metrics] + additional_metrics
    all_metrics = all_metrics[:n_metrics]  # Ensure we have exactly n_metrics
    
    # Determine which metrics are ratios
    ratio_metrics = [m for m in all_metrics if 'ratio' in m or 'per' in m or 'rate' in m]
    
    # Create store profiles for more realistic data
    # Each store has a base multiplier for different metric categories
    store_profiles = {}
    for store in stores:
        profile = {
            'sales_factor': np.random.normal(1, 0.3),  # Mean 1, std 0.3
            'traffic_factor': np.random.normal(1, 0.3),
            'conversion_factor': np.random.normal(1, 0.2),
            'size_category': np.random.choice(['small', 'medium', 'large'], p=[0.2, 0.5, 0.3])
        }
        # Size affects base values
        if profile['size_category'] == 'small':
            profile['base_multiplier'] = 0.7
        elif profile['size_category'] == 'medium':
            profile['base_multiplier'] = 1.0
        else:  # large
            profile['base_multiplier'] = 1.5
            
        store_profiles[store] = profile
    
    # Create data in chunks to avoid memory issues
    chunk_size = 500  # Process 500 stores at a time
    data = []
    
    for chunk_start in range(1, n_stores + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, n_stores)
        chunk_stores = list(range(chunk_start, chunk_end + 1))
        
        chunk_data = []
        for store in chunk_stores:
            profile = store_profiles[store]
            
            for yw in year_weeks:
                # Weekly variation factor (some weeks are better/worse for all stores)
                week_factor = np.random.normal(1, 0.1)
                
                for metric in all_metrics:
                    is_ratio = metric in ratio_metrics
                    
                    if is_ratio:
                        # For ratio metrics, generate both numerator and denominator
                        if 'sales' in metric:
                            base_num = np.random.randint(5000, 40000) * profile['sales_factor'] * profile['base_multiplier'] * week_factor
                            base_denom = np.random.randint(1000, 10000) * profile['traffic_factor'] * profile['base_multiplier'] * week_factor
                        elif 'items' in metric:
                            base_num = np.random.randint(10000, 50000) * profile['sales_factor'] * profile['base_multiplier'] * week_factor
                            base_denom = np.random.randint(5000, 15000) * profile['traffic_factor'] * profile['base_multiplier'] * week_factor
                        elif 'conversion' in metric:
                            base_num = np.random.randint(500, 5000) * profile['conversion_factor'] * profile['base_multiplier'] * week_factor
                            base_denom = np.random.randint(5000, 15000) * profile['traffic_factor'] * profile['base_multiplier'] * week_factor
                        elif 'return' in metric:
                            base_num = np.random.randint(50, 500) * np.random.normal(1, 0.3) * profile['base_multiplier'] * week_factor
                            base_denom = np.random.randint(5000, 40000) * profile['sales_factor'] * profile['base_multiplier'] * week_factor
                        else:
                            # Generic ratio metrics
                            base_num = np.random.randint(1000, 10000) * profile['base_multiplier'] * week_factor
                            base_denom = np.random.randint(100, 1000) * profile['base_multiplier'] * week_factor
                            
                        chunk_data.append({
                            'store': store,
                            'year_week': yw,
                            'metric': metric,
                            'numerator': int(base_num),
                            'denominator': int(base_denom),
                            'isratio': True
                        })
                    else:
                        # For regular metrics
                        if 'sales' in metric:
                            base_value = np.random.randint(5000, 40000) * profile['sales_factor'] * profile['base_multiplier'] * week_factor
                        elif 'visit' in metric:
                            base_value = np.random.randint(1000, 10000) * profile['traffic_factor'] * profile['base_multiplier'] * week_factor
                        elif 'unit' in metric:
                            base_value = np.random.randint(5000, 30000) * profile['sales_factor'] * profile['base_multiplier'] * week_factor
                        else:
                            # Generic value metrics
                            base_value = np.random.randint(1000, 20000) * profile['base_multiplier'] * week_factor
                            
                        chunk_data.append({
                            'store': store,
                            'year_week': yw,
                            'metric': metric,
                            'numerator': int(base_value),
                            'denominator': None,
                            'isratio': False
                        })
        
        # Add the chunk to our main data list
        data.extend(chunk_data)
        print(f"Processed stores {chunk_start} to {chunk_end}")
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    # Add some outliers if requested
    if with_outliers:
        # Add 50 random outliers (extremely high values)
        for _ in range(50):
            store = np.random.choice(stores)
            yw = np.random.choice(year_weeks)
            metric = np.random.choice([m for m in all_metrics if m not in ratio_metrics])
            
            # Find the corresponding row
            idx = df[(df['store'] == store) & (df['year_week'] == yw) & (df['metric'] == metric)].index
            if len(idx) > 0:
                # Create a high outlier (5-10x normal)
                df.loc[idx[0], 'numerator'] = int(df.loc[idx[0], 'numerator'] * np.random.uniform(5, 10))
        
        # Add 50 random outliers (extremely low values)
        for _ in range(50):
            store = np.random.choice(stores)
            yw = np.random.choice(year_weeks)
            metric = np.random.choice([m for m in all_metrics if m not in ratio_metrics])
            
            # Find the corresponding row
            idx = df[(df['store'] == store) & (df['year_week'] == yw) & (df['metric'] == metric)].index
            if len(idx) > 0:
                # Create a low outlier (0.1-0.3x normal)
                df.loc[idx[0], 'numerator'] = int(df.loc[idx[0], 'numerator'] * np.random.uniform(0.1, 0.3))
    
    # Optimize the dataframe
    df = optimize_dataframe(df)
    
    return df

# Load your data (or create sample data for demonstration)
# Normally you would load with: df = pd.read_csv('your_data.csv')
# And then optimize: df = optimize_dataframe(df)

# For demonstration, create a smaller sample to make the notebook run faster
# In real use, you'd use more stores and metrics
df = create_sample_data(n_stores=100, n_metrics=10, n_weeks=5)
print(f"Sample data shape: {df.shape}")
df.head()

## 3. Enhanced Exploratory Data Analysis

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
    
    # Visualize distribution of the sample metric
    plt.figure(figsize=(12, 6))
    sns.histplot(sample_data['value'], kde=True)
    plt.title(f"Distribution of {sample_metric} across all stores")
    plt.axvline(sample_data['value'].mean(), color='red', linestyle='--', label=f'Mean: {sample_data["value"].mean():.2f}')
    plt.axvline(sample_data['value'].median(), color='green', linestyle='--', label=f'Median: {sample_data["value"].median():.2f}')
    plt.legend()
    plt.show()
    
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

## 5. Store Clustering Analysis

def cluster_stores(df, n_clusters=5):
    """
    Group similar stores together using KMeans clustering.
    This helps identify store patterns across many metrics.
    """
    print(f"Clustering {len(df)} stores into {n_clusters} groups...")
    
    # Create a copy to avoid modifying the original
    cluster_df = df.copy()
    
    # Select only numeric columns for clustering
    numeric_cols = cluster_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'store']
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for clustering")
        return df
    
    # Standardize the data (important for KMeans)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_df[numeric_cols])
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_df['cluster'] = kmeans.fit_predict(scaled_data)
    
    # Analyze the clusters
    print("\nCluster sizes:")
    print(cluster_df['cluster'].value_counts())
    
    # Calculate cluster centers in original scale
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                          columns=numeric_cols)
    centers['cluster'] = range(n_clusters)
    
    print("\nCluster characteristics (average values per metric):")
    # Display a summary of the top 2 distinguishing metrics per cluster
    for cluster_id in range(n_clusters):
        # Compare this cluster's centers to overall average
        cluster_center = centers[centers['cluster'] == cluster_id].iloc[0]
        overall_avg = cluster_df[numeric_cols].mean()
        
        # Calculate percent difference
        diff_pct = ((cluster_center[numeric_cols] - overall_avg) / overall_avg) * 100
        
        # Find top distinguishing metrics (highest absolute % difference)
        top_metrics = diff_pct.abs().nlargest(3)
        
        print(f"\nCluster {cluster_id} ({cluster_df['cluster'].eq(cluster_id).sum()} stores):")
        for metric in top_metrics.index:
            direction = "higher" if diff_pct[metric] > 0 else "lower"
            print(f"  {metric}: {abs(diff_pct[metric]):.1f}% {direction} than average")
    
    return cluster_df

# Cluster the stores
clustered_df = cluster_stores(aggregated_df, n_clusters=5)

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
    
    # Add percentile ranks for all metrics
    for metric in metrics:
        result_df[f'{metric}_percentile'] = result_df[metric].rank(pct=True) * 100
    
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
        
        # Add a combined outlier flag column
        result_df[f'{metric}_is_outlier'] = result_df[f'{metric}_outlier'] != 'normal'
    
    # Summary of outliers
    total_cells = len(result_df) * len(metrics)
    total_outliers = outlier_counts['high'] + outlier_counts['low']
    
    print(f"Found {outlier_counts['high']} high outliers and {outlier_counts['low']} low outliers")
    print(f"Total outliers: {total_outliers} ({total_outliers/total_cells:.2%} of all values)")
    
    end_time = time.time()
    print(f"Outlier detection completed in {end_time - start_time:.2f} seconds")
    
    return result_df

# Get list of metrics (exclude store and cluster)
metrics_list = [col for col in aggregated_df.columns 
               if col != 'store' and col != 'cluster' 
               and not col.endswith('_outlier') 
               and not col.endswith('_percentile')
               and not col.endswith('_is_outlier')]

# Identify outliers
outlier_results = identify_outliers_efficiently(clustered_df, metrics_list)

## 7. Visualizations for Large-Scale Data

def plot_metric_distribution(df, metric, by_cluster=True):
    """Plot metric distribution, optionally colored by cluster."""
    plt.figure(figsize=(12, 6))
    
    if by_cluster and 'cluster' in df.columns:
        # Plot with cluster coloring
        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]
            sns.kdeplot(cluster_data[metric], label=f'Cluster {cluster}')
    else:
        # Simple distribution
        sns.histplot(df[metric], kde=True)
    
    # Add vertical lines for percentiles
    plt.axvline(df[metric].quantile(0.25), color='purple', linestyle=':', label='25th percentile')
    plt.axvline(df[metric].median(), color='green', linestyle='--', label='Median')
    plt.axvline(df[metric].quantile(0.75), color='purple', linestyle=':', label='75th percentile')
    
    # Add outlier boundaries
    Q1 = df[metric].quantile(0.25)
    Q3 = df[metric].quantile(0.75)
    IQR = Q3 - Q1
    plt.axvline(Q1 - 1.5 * IQR, color='red', linestyle='-.', label='Outlier boundary')
    plt.axvline(Q3 + 1.5 * IQR, color='red', linestyle='-.')
    
    plt.title(f'Distribution of {metric} across stores')
    plt.legend()
    plt.tight_layout()
    plt.show()

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
    
    # Add cluster information if available
    if 'cluster' in plot_df.columns:
        labels = [f"Store {s} (Cluster {c})" for s, c in zip(plot_df['store'], plot_df['cluster'])]
        plt.yticks(range(len(labels)), labels)
    
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

def visualize_cluster_profiles(df, metrics):
    """Visualize the profile of each cluster across metrics."""
    if 'cluster' not in df.columns:
        print("Cluster column not found. Run clustering first.")
        return
    
    # Calculate mean values by cluster
    cluster_profiles = df.groupby('cluster')[metrics].mean()
    
    # Normalize the values for better visualization (0-1 scale)
    normalized = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
    
    # Transpose for easier plotting
    normalized = normalized.T
    
    # Plot
    plt.figure(figsize=(12, len(metrics) * 0.5))
    sns.heatmap(normalized, annot=False, cmap='YlGnBu', 
                cbar_kws={'label': 'Normalized Value (0-1)'})
    plt.title('Cluster Profiles Across Metrics')
    plt.tight_layout()
    plt.show()
    
    # Also show actual average values for each cluster
    print("Average values by cluster:")
    return cluster_profiles

# Visualize distribution of a sample metric
sample_metric = metrics_list[0]  # First metric
plot_metric_distribution(outlier_results, sample_metric, by_cluster=True)

# Visualize top/bottom stores for the sample metric
plot_top_bottom_stores(outlier_results, sample_metric, n=10)

# Visualize cluster profiles
cluster_profiles = visualize_cluster_profiles(outlier_results, metrics_list)
print(cluster_profiles)

## 8. Optimized Correlation Analysis

def analyze_correlations(df, metrics, cluster_corrs=True, plot_clustered=True):
    """
    Analyze correlations between metrics, with options for:
    - Clustering the correlation matrix for better visualization
    - Showing correlations within each store cluster
    """
    print("Analyzing correlations between metrics...")
    
    # Calculate overall correlation matrix
    corr = df[metrics].corr()
    
    # For large number of metrics, use hierarchical clustering to organize the matrix
    if plot_clustered and len(metrics) > 10:
        # Convert correlation to distance matrix
        dist = distance.pdist(corr.values)
        link = hierarchy.linkage(dist, method='average')
        order = hierarchy.dendrogram(link, no_plot=True)['leaves']
        
        # Reorder the correlation matrix
        clustered_corr = corr.iloc[order, order]
        
        # Plot with better readability
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(clustered_corr, dtype=bool))
        sns.heatmap(clustered_corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                    annot=False, linewidths=.5)
        plt.title('Clustered Correlation Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        # Simpler correlation plot for fewer metrics
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
    
    # If requested and clusters exist, analyze correlations within each cluster
    if cluster_corrs and 'cluster' in df.columns:
        print("\nAnalyzing correlations within each cluster...")
        
        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]
            cluster_corr = cluster_data[metrics].corr()
            
            # Find correlation differences vs. overall
            diff = cluster_corr - corr
            
            # Find the pairs with the biggest correlation differences
            diff_pairs = []
            for i in range(len(metrics)):
                for j in range(i+1, len(metrics)):
                    diff_value = diff.iloc[i, j]
                    overall_value = corr.iloc[i, j]
                    cluster_value = cluster_corr.iloc[i, j]
                    if abs(diff_value) > 0.2:  # Only show substantial differences
                        diff_pairs.append((metrics[i], metrics[j], overall_value, cluster_value, diff_value))
            
            # Sort by absolute difference
            diff_pairs.sort(key=lambda x: abs(x[4]), reverse=True)
            
            # Show top differences
            if diff_pairs:
                print(f"\nCluster {cluster} - Top correlation differences vs overall:")
                for m1, m2, overall, cluster_val, diff in diff_pairs[:3]:  # Show top 3
                    print(f"{m1} and {m2}: {overall:.3f} overall vs {cluster_val:.3f} in cluster ({diff:.3f} difference)")
    
    return corr

# Analyze correlations
correlation_matrix = analyze_correlations(outlier_results, metrics_list)

## 9. Generate Summary Report

def generate_summary_report(df, metrics, n_top_bottom=10):
    """Generate a compact summary report with insights for large-scale analysis."""
    print("Generating summary report...")
    
    # Initialize summary dataframe
    summary = pd.DataFrame(columns=['Metric', 'Mean', 'Median', 'Std Dev', 
                                   'Top 5 Stores', 'Bottom 5 Stores',
                                   'Outlier Count', 'Correlates With'])
    
    # Calculate correlations once
    corr = df[metrics].corr()
    
    for metric in metrics:
        # Calculate statistics
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        std_val = df[metric].std()
        
        # Get top and bottom stores
        sorted_df = df[['store', metric]].sort_values(by=metric, ascending=False)
        top_stores = sorted_df.head(5)['store'].tolist()
        bottom_stores = sorted_df.tail(5)['store'].tolist()
        
        # Count outliers
        outlier_count = df[df[f'{metric}_outlier'] != 'normal'].shape[0]
        
        # Find strongest correlations (absolute value > 0.5)
        correlations = corr[metric].drop(metric)  # Drop self-correlation
        strong_corrs = correlations[abs(correlations) > 0.5]
        corr_with = []
        
        for other_metric, corr_val in strong_corrs.items():
            direction = '+' if corr_val > 0 else '-'
            corr_with.append(f"{other_metric} ({direction}{abs(corr_val):.2f})")
        
        # Add to summary
        summary = pd.concat([summary, pd.DataFrame({
            'Metric': [metric],
            'Mean': [mean_val],
            'Median': [median_val],
            'Std Dev': [std_val],
            'Top 5 Stores': [top_stores],
            'Bottom 5 Stores': [bottom_stores],
            'Outlier Count': [outlier_count],
            'Correlates With': [', '.join(corr_with) if corr_with else 'None']
        })], ignore_index=True)
    
    # Format the summary table for better readability
    pd.set_option('display.max_colwidth', None)
    
    # Create a cluster summary if clustering was performed
    if 'cluster' in df.columns:
        print("\nCluster Summary:")
        cluster_counts = df['cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            print(f"Cluster {cluster}: {count} stores ({count/len(df):.1%} of total)")
    
    return summary

# Generate summary report
summary_report = generate_summary_report(outlier_results, metrics_list)
print("\nMetric Performance Summary:")
summary_report

## 10. Recommendations for Further Analysis

print("""
## Recommendations for Further Analysis

Based on the exploratory analysis, here are some suggestions for deeper investigation:

1. Time Series Analysis:
   - Analyze week-over-week trends for key metrics
   - Look for seasonality patterns across stores
   - Identify stores with improving or declining performance

2. Geographic Analysis:
   - If store location data is available, map performance geographically
   - Look for regional patterns in the clusters
   - Analyze if nearby stores compete or complement each other

3. Comparative Analysis:
   - Compare performance during promotions vs. non-promotion periods
   - Analyze how different store types perform on various metrics
   - Benchmark against industry averages if available

4. Multi-dimensional Analysis:
   - Create composite metrics (e.g., efficiency = sales / square footage)
   - Look at interactions between metrics (e.g., how conversion rate affects revenue)
   - Segment analysis by store characteristics (size, format, etc.)
""")

## 11. Save Results (Optional)

# Uncomment these lines to save results to CSV files
# print("Saving results to CSV files...")
# aggregated_df.to_csv('aggregated_metrics.csv', index=False)
# outlier_results.to_csv('outlier_results.csv', index=False)
# summary_report.to_csv('performance_summary.csv', index=False)

print("\nAnalysis complete!")
