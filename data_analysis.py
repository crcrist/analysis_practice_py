
# Store Metrics Analysis for Large Scale Data
# Enhanced with Advanced Outlier Visualizations
# Optimized for 5000+ stores and 50+ metrics

## 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from scipy.spatial import distance
import time
import warnings
warnings.filterwarnings('ignore')

# For interactive input
import sys

# Set basic figure size and improved style for better visualizations
plt.rcParams['figure.figsize'] = (14, 10)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['savefig.facecolor'] = '#f9f9f9'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

## 2. Data Loading and Optimization (Same as original)

def optimize_dataframe(df):
    """Optimize dataframe memory usage by setting appropriate dtypes."""
    # Convert numeric columns to smaller types
    if 'store' in df.columns:
        df['store'] = df['store'].astype('int32')
    if 'store_number' in df.columns:
        df['store_number'] = df['store_number'].astype('int32')
    if 'year_week' in df.columns:
        df['year_week'] = df['year_week'].astype('int32')
    
    # Convert string columns to categories
    if 'metric' in df.columns:
        df['metric'] = df['metric'].astype('category')
    
    # Convert boolean columns
    if 'isratio' in df.columns:
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

## 3. Efficient Metric Aggregation (Same as original)

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
    all_stores = df['store'].unique() if 'store' in df.columns else df['store_number'].unique()
    store_col = 'store' if 'store' in df.columns else 'store_number'
    
    results = pd.DataFrame({store_col: all_stores})
    results.set_index(store_col, inplace=True)
    
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
                num_denom = metric_data.groupby(store_col).agg({
                    'numerator': 'sum',
                    'denominator': 'sum'
                })
                # Calculate the ratio
                results[metric] = num_denom['numerator'] / num_denom['denominator']
            else:
                # For regular metrics, just sum the numerator
                results[metric] = metric_data.groupby(store_col)['numerator'].sum()
        else:
            # Small enough to process at once
            if is_ratio:
                # Pre-aggregate numerator and denominator by store
                num_denom = metric_data.groupby(store_col).agg({
                    'numerator': 'sum',
                    'denominator': 'sum'
                })
                # Calculate the ratio
                results[metric] = num_denom['numerator'] / num_denom['denominator']
            else:
                # For regular metrics, just sum the numerator
                results[metric] = metric_data.groupby(store_col)['numerator'].sum()
    
    # Reset index to make store a column
    results = results.reset_index()
    
    end_time = time.time()
    print(f"Aggregation completed in {end_time - start_time:.2f} seconds")
    
    return results

## 4. Store Clustering Analysis (Same as original)

def cluster_stores(df, n_clusters=5):
    """
    Group similar stores together using KMeans clustering.
    This helps identify store patterns across many metrics.
    """
    print(f"Clustering {len(df)} stores into {n_clusters} groups...")
    
    # Create a copy to avoid modifying the original
    cluster_df = df.copy()
    
    # Get the store column name
    store_col = 'store' if 'store' in df.columns else 'store_number'
    
    # Select only numeric columns for clustering
    numeric_cols = cluster_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != store_col]
    
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

## 5. Enhanced Outlier Detection

def calculate_outlier_scores(df, metrics, methods=['zscore', 'iqr', 'isolation_forest']):
    """
    Calculate outlier scores using multiple methods for more robust detection.
    
    Parameters:
    - df: Dataframe with store metrics
    - metrics: List of metrics to analyze
    - methods: List of outlier detection methods to use
    
    Returns:
    - Dataframe with outlier scores added
    """
    start_time = time.time()
    print("Calculating outlier scores using multiple methods...")
    
    # Get the store column name
    store_col = 'store' if 'store' in df.columns else 'store_number'
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Add normalized values for better comparison across metrics
    for metric in metrics:
        # Min-max scale to 0-1 range
        scaler = MinMaxScaler()
        result_df[f'{metric}_norm'] = scaler.fit_transform(result_df[[metric]])
    
    # Calculate Z-score based outliers
    if 'zscore' in methods:
        for metric in metrics:
            # Calculate z-scores
            mean = result_df[metric].mean()
            std = result_df[metric].std()
            result_df[f'{metric}_zscore'] = (result_df[metric] - mean) / std
    
    # Calculate IQR based outliers
    if 'iqr' in methods:
        for metric in metrics:
            # Calculate IQR
            Q1 = result_df[metric].quantile(0.25)
            Q3 = result_df[metric].quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate IQR-based outlier score 
            # (how many IQRs away from the median)
            median = result_df[metric].median()
            result_df[f'{metric}_iqr_score'] = abs(result_df[metric] - median) / IQR
            
            # Add direction (positive/negative)
            result_df[f'{metric}_iqr_direction'] = np.where(
                result_df[metric] > median, 'high', 'low')
    
    # Calculate Isolation Forest based outliers
    if 'isolation_forest' in methods:
        # Combine all normalized metrics
        norm_cols = [f'{metric}_norm' for metric in metrics]
        # For large datasets, use a sample to train the model
        sample_size = min(1000, len(result_df))
        sample_idx = np.random.choice(result_df.index, sample_size, replace=False)
        
        # Train isolation forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_forest.fit(result_df.loc[sample_idx, norm_cols])
        
        # Predict outlier scores (-1 for outliers, 1 for inliers, convert to 0-1 scale)
        # Lower score = more likely to be an outlier
        scores = iso_forest.decision_function(result_df[norm_cols])
        result_df['isolation_forest_score'] = (scores + 1) / 2  # Convert to 0-1 scale
    
    # Calculate composite outlier score
    result_df['composite_outlier_score'] = 0
    
    # Z-score contribution
    if 'zscore' in methods:
        zscore_cols = [f'{metric}_zscore' for metric in metrics]
        result_df['max_abs_zscore'] = result_df[zscore_cols].abs().max(axis=1)
        result_df['composite_outlier_score'] += (1 - 1/(1 + result_df['max_abs_zscore']))
    
    # IQR contribution
    if 'iqr' in methods:
        iqr_cols = [f'{metric}_iqr_score' for metric in metrics]
        result_df['max_iqr_score'] = result_df[iqr_cols].max(axis=1)
        result_df['composite_outlier_score'] += (1 - 1/(1 + result_df['max_iqr_score']))
    
    # Isolation Forest contribution
    if 'isolation_forest' in methods:
        result_df['composite_outlier_score'] += (1 - result_df['isolation_forest_score'])
    
    # Normalize the composite score to 0-1 range
    result_df['composite_outlier_score'] /= len(methods)
    
    # Add outlier flags based on composite score
    result_df['is_outlier'] = result_df['composite_outlier_score'] > 0.8
    
    # For each metric, add a specific outlier flag
    for metric in metrics:
        # Calculate a per-metric outlier score
        if 'zscore' in methods:
            zscore_contrib = abs(result_df[f'{metric}_zscore']) / 3  # Scale to approx. 0-1
            per_metric_score = zscore_contrib
        else:
            per_metric_score = 0
            
        if 'iqr' in methods:
            iqr_contrib = result_df[f'{metric}_iqr_score'] / 3  # Scale to approx. 0-1
            per_metric_score += iqr_contrib
            
        per_metric_score /= sum([m in methods for m in ['zscore', 'iqr']])
        
        # Flag as outlier if score > 0.8 (you can adjust this threshold)
        result_df[f'{metric}_is_outlier'] = per_metric_score > 0.8
        
        # Add direction (high/low)
        if 'iqr' in methods:
            result_df[f'{metric}_outlier_direction'] = result_df[f'{metric}_iqr_direction']
        else:
            mean = result_df[metric].mean()
            result_df[f'{metric}_outlier_direction'] = np.where(
                result_df[metric] > mean, 'high', 'low')
    
    # Count outliers
    num_outliers = result_df['is_outlier'].sum()
    print(f"Identified {num_outliers} stores ({num_outliers/len(result_df):.1%}) as outliers")
    
    end_time = time.time()
    print(f"Outlier detection completed in {end_time - start_time:.2f} seconds")
    
    return result_df

## 6. Advanced Outlier Visualizations

def visualize_multi_dimensional_outliers(df, x_metric, y_metric, color_metric=None, title=None):
    """
    Create a scatter plot showing stores in two dimensions with outliers highlighted.
    Optionally color by a third metric.
    
    Parameters:
    - df: Dataframe with store metrics and outlier scores
    - x_metric: Metric to use for x-axis
    - y_metric: Metric to use for y-axis  
    - color_metric: Optional metric to use for coloring
    - title: Optional title override
    """
    # Get the store column name
    store_col = 'store' if 'store' in df.columns else 'store_number'
    
    plt.figure(figsize=(14, 10))
    
    # Determine if a store is an outlier in either metric
    x_outliers = df[f'{x_metric}_is_outlier'] if f'{x_metric}_is_outlier' in df.columns else pd.Series(False, index=df.index)
    y_outliers = df[f'{y_metric}_is_outlier'] if f'{y_metric}_is_outlier' in df.columns else pd.Series(False, index=df.index)
    
    # Create a combined outlier flag
    df['point_is_outlier'] = x_outliers | y_outliers
    
    # Prepare color data
    if color_metric:
        # Use the provided color metric
        color_data = df[color_metric]
        cmap = 'viridis'
        vmin, vmax = color_data.min(), color_data.max()
    else:
        # Use composite outlier score for coloring
        color_data = df['composite_outlier_score'] if 'composite_outlier_score' in df.columns else pd.Series(0, index=df.index)
        cmap = 'YlOrRd'
        vmin, vmax = 0, 1
    
    # Plot normal points first
    normal_points = ~df['point_is_outlier']
    scatter_normal = plt.scatter(
        df.loc[normal_points, x_metric], 
        df.loc[normal_points, y_metric],
        c=color_data.loc[normal_points],
        cmap=cmap,
        alpha=0.6,
        s=50,
        edgecolor='none',
        vmin=vmin, vmax=vmax
    )
    
    # Plot outlier points with larger markers and different style
    outlier_points = df['point_is_outlier']
    if outlier_points.any():
        scatter_outliers = plt.scatter(
            df.loc[outlier_points, x_metric], 
            df.loc[outlier_points, y_metric],
            c=color_data.loc[outlier_points],
            cmap=cmap,
            alpha=1.0,
            s=120,
            edgecolor='black',
            linewidth=1.5,
            marker='o',
            vmin=vmin, vmax=vmax
        )
    
    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label(color_metric if color_metric else 'Outlier Score')
    
    # Add labels for the most extreme outliers
    if outlier_points.any():
        # Sort by composite outlier score to find the most extreme points
        top_outliers = df.loc[outlier_points].sort_values(
            by='composite_outlier_score' if 'composite_outlier_score' in df.columns else x_metric,
            ascending=False
        ).head(10)
        
        for idx, row in top_outliers.iterrows():
            plt.annotate(
                f"{row[store_col]}",
                (row[x_metric], row[y_metric]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=11,
                fontweight='bold',
                backgroundcolor='white',
                alpha=0.8
            )
    
    # Add title and labels
    plt.title(title if title else f'Store Performance: {x_metric} vs {y_metric}')
    plt.xlabel(x_metric.replace('_', ' ').title())
    plt.ylabel(y_metric.replace('_', ' ').title())
    
    # Add a grid to improve readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add reference lines for means or medians
    plt.axvline(df[x_metric].median(), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(df[y_metric].median(), color='gray', linestyle='--', alpha=0.5)
    
    # Make axes start at zero if appropriate
    if df[x_metric].min() >= 0:
        plt.xlim(left=0)
    if df[y_metric].min() >= 0:
        plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Show stats in a textbox
    outlier_pct = outlier_points.mean() * 100
    x_outlier_pct = x_outliers.mean() * 100
    y_outlier_pct = y_outliers.mean() * 100
    
    stats_text = (
        f"Total stores: {len(df)}\n"
        f"Outliers: {outlier_points.sum()} ({outlier_pct:.1f}%)\n"
        f"{x_metric} outliers: {x_outliers.sum()} ({x_outlier_pct:.1f}%)\n"
        f"{y_metric} outliers: {y_outliers.sum()} ({y_outlier_pct:.1f}%)"
    )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.show()

def visualize_deviation_from_expected(df, metric, by_cluster=True, n_stores=50):
    """
    Create a deviation-from-expected chart showing how stores perform 
    relative to what would be predicted based on their characteristics.
    
    Parameters:
    - df: Dataframe with store metrics and outlier scores
    - metric: Metric to analyze
    - by_cluster: Whether to calculate expected values by cluster
    - n_stores: Number of stores to show (most deviating)
    """
    # Get the store column name
    store_col = 'store' if 'store' in df.columns else 'store_number'
    
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    
    # Calculate expected values
    if by_cluster and 'cluster' in plot_df.columns:
        # Calculate expected value based on cluster average
        cluster_means = plot_df.groupby('cluster')[metric].mean()
        plot_df['expected_value'] = plot_df['cluster'].map(cluster_means)
    else:
        # Use overall mean as expected value
        plot_df['expected_value'] = plot_df[metric].mean()
    
    # Calculate deviation
    plot_df['deviation'] = plot_df[metric] - plot_df['expected_value']
    
    # Calculate percent deviation for easier interpretation
    plot_df['percent_deviation'] = (plot_df['deviation'] / plot_df['expected_value']) * 100
    
    # Sort by absolute percent deviation
    plot_df = plot_df.sort_values(by='percent_deviation', key=abs, ascending=False)
    
    # Take the top n stores
    plot_df = plot_df.head(n_stores)
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Set bar colors based on deviation direction
    colors = ['#d73027' if x < 0 else '#4575b4' for x in plot_df['percent_deviation']]
    
    # Create horizontal bar chart
    bars = plt.barh(
        y=plot_df[store_col].astype(str),
        width=plot_df['percent_deviation'],
        color=colors,
        height=0.7
    )
    
    # Add cluster information if available
    if by_cluster and 'cluster' in plot_df.columns:
        # Create labels with cluster information
        labels = [f"Store {s} (Cluster {c})" for s, c in zip(plot_df[store_col], plot_df['cluster'])]
        plt.yticks(range(len(labels)), labels)
    
    # Add value labels to the bars
    for bar, value, store_id in zip(bars, plot_df['percent_deviation'], plot_df[store_col]):
        label_color = 'black'
        xval = value
        if abs(value) < 5:  # Small values get special handling
            if value < 0:
                xval = -1  # Place just to the right of the zero line
            else:
                xval = 1  # Place just to the right of the zero line
                
        plt.text(
            xval, 
            bar.get_y() + bar.get_height()/2,
            f"{value:+.1f}% (Store {store_id})",
            va='center',
            ha='left' if value >= 0 else 'right',
            color=label_color,
            fontweight='bold',
            fontsize=9
        )
    
    # Add a vertical line at 0
    plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
    
    # Add reference lines
    plt.axvline(-25, color='#d73027', linestyle='--', alpha=0.3)
    plt.axvline(25, color='#4575b4', linestyle='--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel(f'Percent Deviation from Expected {metric}')
    plt.title(f'Top {n_stores} Stores by Deviation from Expected {metric}')
    
    # Add a grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Improve layout
    plt.tight_layout()
    
    # Add explanatory textbox
    if by_cluster:
        explanation = (
            "This chart shows how stores deviate from their expected performance\n"
            "based on their cluster average. Stores that deviate significantly\n"
            "may be outliers or might have special characteristics."
        )
    else:
        explanation = (
            "This chart shows how stores deviate from the overall average.\n"
            "Stores that deviate significantly may be outliers or have\n"
            "unique characteristics worth investigating."
        )
    
    plt.figtext(0.02, 0.02, explanation, fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.show()

def visualize_anomaly_heatmap(df, metrics, n_stores=50, title=None):
    """
    Create a heatmap showing outlier metrics across multiple stores.
    
    Parameters:
    - df: Dataframe with store metrics and outlier scores
    - metrics: List of metrics to include
    - n_stores: Number of most anomalous stores to show
    - title: Optional custom title
    """
    # Get the store column name
    store_col = 'store' if 'store' in df.columns else 'store_number'
    
    # Sort stores by composite outlier score
    if 'composite_outlier_score' in df.columns:
        # Use the composite score if available
        sorted_stores = df.sort_values('composite_outlier_score', ascending=False)
    else:
        # Create a simple score by counting outlier flags
        outlier_cols = [col for col in df.columns if col.endswith('_is_outlier')]
        if outlier_cols:
            df['temp_outlier_count'] = df[outlier_cols].sum(axis=1)
            sorted_stores = df.sort_values('temp_outlier_count', ascending=False)
        else:
            # Fallback to simple standard deviation from mean
            df['temp_std_dev'] = 0
            for metric in metrics:
                mean = df[metric].mean()
                std = df[metric].std()
                df['temp_std_dev'] += abs((df[metric] - mean) / std)
            df['temp_std_dev'] /= len(metrics)
            sorted_stores = df.sort_values('temp_std_dev', ascending=False)
    
    # Select top n_stores
    plot_stores = sorted_stores.head(n_stores)[store_col].values
    
    # Prepare data for heatmap
    heatmap_data = []
    
    for store_id in plot_stores:
        store_row = df[df[store_col] == store_id].iloc[0]
        
        metric_values = {}
        for metric in metrics:
            # Get the normalized value
            if f'{metric}_norm' in store_row:
                norm_value = store_row[f'{metric}_norm']
            else:
                # Min-max scaling on the fly if norm not pre-calculated
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    norm_value = (store_row[metric] - min_val) / (max_val - min_val)
                else:
                    norm_value = 0.5
            
            # Get the zscore if available
            if f'{metric}_zscore' in store_row:
                z_score = store_row[f'{metric}_zscore']
            else:
                mean = df[metric].mean()
                std = df[metric].std()
                z_score = (store_row[metric] - mean) / std if std > 0 else 0
            
            # Get outlier flag if available
            is_outlier = store_row.get(f'{metric}_is_outlier', False)
            
            # Get outlier direction if available
            direction = store_row.get(f'{metric}_outlier_direction', 
                                      'high' if z_score > 0 else 'low')
            
            # Store the data
            metric_values[metric] = {
                'norm_value': norm_value,
                'z_score': z_score,
                'is_outlier': is_outlier,
                'direction': direction
            }
        
        heatmap_data.append({
            'store_id': store_id,
            'metrics': metric_values,
            'cluster': store_row.get('cluster', None),
            'composite_score': store_row.get('composite_outlier_score', 0)
        })
    
    # Prepare data for the heatmap
    heatmap_array = np.zeros((len(plot_stores), len(metrics)))
    for i, store_data in enumerate(heatmap_data):
        for j, metric in enumerate(metrics):
            # Use z-score for the heatmap color (capped to prevent extreme values)
            z_score = store_data['metrics'][metric]['z_score']
            capped_z = max(min(z_score, 5), -5)  # Cap at +/- 5 stdev
            heatmap_array[i, j] = capped_z
    
    # Create store labels with cluster info if available
    if 'cluster' in df.columns:
        store_labels = [f"Store {data['store_id']} (C{data['cluster']})" 
                        for data in heatmap_data]
    else:
        store_labels = [f"Store {data['store_id']}" for data in heatmap_data]
    
    # Calculate a compact metric name for better display
    metric_labels = []
    for metric in metrics:
        # Shorten label if needed
        if len(metric) > 15:
            parts = metric.split('_')
            if len(parts) > 1:
                # Keep first letter of each part
                short_name = ''.join([p[0].upper() for p in parts])
            else:
                # Just truncate
                short_name = metric[:10] + '...'
            metric_labels.append(short_name)
        else:
            metric_labels.append(metric)
    
    # Create a helper function to add marker for extreme outliers
    def format_value(val):
        return f"{val:.1f}"
    
    # Create the heatmap
    plt.figure(figsize=(max(12, len(metrics) * 0.8), max(10, len(plot_stores) * 0.4)))
    
    # Use a diverging colormap for positive/negative z-scores
    cmap = plt.cm.coolwarm
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_array,
        cmap=cmap,
        center=0,
        vmin=-3, vmax=3,  # +/- 3 std dev range
        cbar_kws={'label': 'Z-Score'},
        linewidths=0.5,
        linecolor='lightgray',
        annot=False
    )
    
    # Add markers for outliers
    for i in range(len(plot_stores)):
        for j in range(len(metrics)):
            metric = metrics[j]
            is_outlier = heatmap_data[i]['metrics'][metric]['is_outlier']
            direction = heatmap_data[i]['metrics'][metric]['direction']
            
            if is_outlier:
                marker = 'o' if direction == 'high' else 'v'
                plt.scatter(
                    j + 0.5, i + 0.5,
                    marker=marker,
                    c='white' if abs(heatmap_array[i, j]) > 1.5 else 'black',
                    s=80,
                    alpha=0.8,
                    edgecolor='black'
                )
    
    # Add heatmap styling
    plt.yticks(np.arange(len(store_labels)) + 0.5, store_labels, rotation=0)
    plt.xticks(np.arange(len(metric_labels)) + 0.5, metric_labels, rotation=45, ha='right')
    
    plt.title(title if title else 'Multi-Metric Outlier Analysis Heatmap')
    
    # Add a legend for the markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=8, label='High Outlier'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='black', 
               markersize=8, label='Low Outlier')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Add explanation textbox
    explanation = (
        "This heatmap shows z-scores across multiple metrics for the most anomalous stores.\n"
        "Red cells indicate values above average, blue cells below average.\n"
        "The more intense the color, the more standard deviations from the mean.\n"
        "Markers indicate outliers: ○ for high outliers, ▽ for low outliers."
    )
    
    plt.figtext(0.02, 0.02, explanation, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.show()

def visualize_pca_with_outliers(df, metrics, n_components=2, title=None):
    """
    Create a PCA plot showing stores in a reduced dimension space with outliers highlighted.
    
    Parameters:
    - df: Dataframe with store metrics and outlier scores  
    - metrics: List of metrics to include in the PCA
    - n_components: Number of PCA components to use (2 or 3)
    - title: Optional custom title
    """
    # Get the store column name
    store_col = 'store' if 'store' in df.columns else 'store_number'
    
    # Select required metrics
    X = df[metrics].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a dataframe with PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df[store_col] = df[store_col].values
    
    # Add cluster information if available
    if 'cluster' in df.columns:
        pca_df['cluster'] = df['cluster'].values
    
    # Add outlier information
    if 'is_outlier' in df.columns:
        pca_df['is_outlier'] = df['is_outlier'].values
    elif 'composite_outlier_score' in df.columns:
        # Use composite score to determine outliers
        pca_df['is_outlier'] = df['composite_outlier_score'] > 0.8
    else:
        # Create a simple outlier flag based on PCA distance from center
        distances = np.sum(X_pca**2, axis=1)
        threshold = np.percentile(distances, 95)  # Top 5% as outliers
        pca_df['is_outlier'] = distances > threshold
    
    # Add outlier score if available
    if 'composite_outlier_score' in df.columns:
        pca_df['outlier_score'] = df['composite_outlier_score'].values
    
    # Plot
    if n_components == 2:
        plt.figure(figsize=(14, 10))
        
        # Plot normal points
        normal_mask = ~pca_df['is_outlier']
        
        if 'cluster' in pca_df.columns:
            # Color by cluster for normal points
            cluster_ids = sorted(pca_df['cluster'].unique())
            
            for cluster in cluster_ids:
                cluster_mask = (pca_df['cluster'] == cluster) & normal_mask
                plt.scatter(
                    pca_df.loc[cluster_mask, 'PC1'],
                    pca_df.loc[cluster_mask, 'PC2'],
                    alpha=0.7,
                    s=70,
                    label=f'Cluster {cluster}'
                )
        else:
            # No cluster info, use a single color
            plt.scatter(
                pca_df.loc[normal_mask, 'PC1'],
                pca_df.loc[normal_mask, 'PC2'],
                alpha=0.7,
                s=70,
                color='#3498db',
                label='Normal'
            )
        
        # Plot outliers with a distinct style
        outlier_mask = pca_df['is_outlier']
        
        if outlier_mask.any():
            if 'outlier_score' in pca_df.columns:
                # Use outlier score for color intensity
                scatter = plt.scatter(
                    pca_df.loc[outlier_mask, 'PC1'],
                    pca_df.loc[outlier_mask, 'PC2'],
                    c=pca_df.loc[outlier_mask, 'outlier_score'],
                    cmap='YlOrRd',
                    alpha=1.0,
                    s=120,
                    marker='*',
                    edgecolor='black',
                    linewidth=1,
                    label='Outlier'
                )
                plt.colorbar(scatter, label='Outlier Score')
            else:
                # Simple outlier highlighting
                plt.scatter(
                    pca_df.loc[outlier_mask, 'PC1'],
                    pca_df.loc[outlier_mask, 'PC2'],
                    alpha=1.0,
                    s=120,
                    color='#e74c3c',
                    marker='*',
                    edgecolor='black',
                    linewidth=1,
                    label='Outlier'
                )
            
            # Label the top outliers
            top_outliers = pca_df.loc[outlier_mask].sort_values(
                by='outlier_score' if 'outlier_score' in pca_df.columns else 'PC1'
            ).tail(10)
            
            for _, row in top_outliers.iterrows():
                plt.annotate(
                    f"{int(row[store_col])}",
                    (row['PC1'], row['PC2']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=11,
                    fontweight='bold',
                    backgroundcolor='white',
                    alpha=0.8
                )
        
        # Add principal component information
        pc1_var = pca.explained_variance_ratio_[0] * 100
        pc2_var = pca.explained_variance_ratio_[1] * 100
        
        plt.xlabel(f'Principal Component 1 ({pc1_var:.1f}% Variance)')
        plt.ylabel(f'Principal Component 2 ({pc2_var:.1f}% Variance)')
        
        # Add title
        plt.title(title if title else 'PCA Projection with Outliers Highlighted')
        
        # Add legend
        plt.legend(loc='best')
        
        # Add a grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add axes lines
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Show feature contribution arrows
        for i, feature in enumerate(metrics):
            # Get the feature loading for PC1 and PC2
            loading = pca.components_[:2, i]
            plt.arrow(0, 0, loading[0]*5, loading[1]*5, 
                      head_width=0.3, head_length=0.3, fc='gray', ec='gray', alpha=0.6)
            
            plt.text(loading[0]*5.2, loading[1]*5.2, feature, 
                     color='gray', ha='center', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Add explanation
        explanation = (
            "This PCA plot reduces the dimensionality of the data to 2 components.\n"
            f"PC1 and PC2 together explain {pc1_var + pc2_var:.1f}% of the variance.\n"
            "Outliers are marked with stars. The arrows show feature contributions.\n"
            f"Total stores: {len(pca_df)}, Outliers: {outlier_mask.sum()} ({outlier_mask.mean()*100:.1f}%)"
        )
        
        plt.figtext(0.02, 0.02, explanation, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
    else:  # 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot normal points
        normal_mask = ~pca_df['is_outlier']
        
        if 'cluster' in pca_df.columns:
            # Color by cluster for normal points
            cluster_ids = sorted(pca_df['cluster'].unique())
            
            for cluster in cluster_ids:
                cluster_mask = (pca_df['cluster'] == cluster) & normal_mask
                ax.scatter(
                    pca_df.loc[cluster_mask, 'PC1'],
                    pca_df.loc[cluster_mask, 'PC2'],
                    pca_df.loc[cluster_mask, 'PC3'],
                    alpha=0.7,
                    s=70,
                    label=f'Cluster {cluster}'
                )
        else:
            # No cluster info, use a single color
            ax.scatter(
                pca_df.loc[normal_mask, 'PC1'],
                pca_df.loc[normal_mask, 'PC2'],
                pca_df.loc[normal_mask, 'PC3'],
                alpha=0.7,
                s=70,
                color='#3498db',
                label='Normal'
            )
        
        # Plot outliers with a distinct style
        outlier_mask = pca_df['is_outlier']
        
        if outlier_mask.any():
            ax.scatter(
                pca_df.loc[outlier_mask, 'PC1'],
                pca_df.loc[outlier_mask, 'PC2'],
                pca_df.loc[outlier_mask, 'PC3'],
                alpha=1.0,
                s=120,
                color='#e74c3c',
                marker='*',
                edgecolor='black',
                linewidth=1,
                label='Outlier'
            )
        
        # Add principal component information
        pc1_var = pca.explained_variance_ratio_[0] * 100
        pc2_var = pca.explained_variance_ratio_[1] * 100
        pc3_var = pca.explained_variance_ratio_[2] * 100
        
        ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)')
        ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)')
        ax.set_zlabel(f'PC3 ({pc3_var:.1f}%)')
        
        # Add title
        plt.title(title if title else 'PCA Projection with Outliers Highlighted')
        
        # Add legend
        plt.legend(loc='best')
    
    plt.show()

def visualize_outlier_metrics_distribution(df, metric, title=None):
    """
    Create a distribution plot showing the distribution of a metric, with outliers highlighted.
    
    Parameters:
    - df: Dataframe with store metrics and outlier scores
    - metric: Metric to visualize
    - title: Optional custom title
    """
    # Check if outlier flag exists for this metric
    has_outlier_flag = f'{metric}_is_outlier' in df.columns
    
    plt.figure(figsize=(14, 10))
    
    # Plot the overall distribution with KDE
    sns.histplot(df[metric], kde=True, color='#3498db', alpha=0.6, bins=30)
    
    # Highlight outliers if available
    if has_outlier_flag:
        outlier_mask = df[f'{metric}_is_outlier']
        if outlier_mask.any():
            # Get high and low outliers if direction is available
            if f'{metric}_outlier_direction' in df.columns:
                high_mask = outlier_mask & (df[f'{metric}_outlier_direction'] == 'high')
                low_mask = outlier_mask & (df[f'{metric}_outlier_direction'] == 'low')
                
                # Plot high outliers
                if high_mask.any():
                    sns.histplot(df.loc[high_mask, metric], 
                                kde=False, color='#e74c3c', alpha=0.8, bins=30)
                
                # Plot low outliers
                if low_mask.any():
                    sns.histplot(df.loc[low_mask, metric], 
                                kde=False, color='#9b59b6', alpha=0.8, bins=30)
            else:
                # Just highlight all outliers
                sns.histplot(df.loc[outlier_mask, metric], 
                            kde=False, color='#e74c3c', alpha=0.8, bins=30)
    
    # Add statistical reference lines
    median = df[metric].median()
    mean = df[metric].mean()
    
    plt.axvline(mean, color='black', linestyle='-', linewidth=1.5, 
                label=f'Mean: {mean:.1f}')
    plt.axvline(median, color='green', linestyle='--', linewidth=1.5, 
                label=f'Median: {median:.1f}')
    
    # Add IQR and outlier boundaries
    Q1 = df[metric].quantile(0.25)
    Q3 = df[metric].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    plt.axvline(lower_bound, color='red', linestyle='-.', linewidth=1.2, 
                label=f'Lower Outlier Bound: {lower_bound:.1f}')
    plt.axvline(upper_bound, color='red', linestyle='-.', linewidth=1.2, 
                label=f'Upper Outlier Bound: {upper_bound:.1f}')
    
    # Add percentile lines
    for p in [10, 90]:
        p_val = df[metric].quantile(p/100)
        plt.axvline(p_val, color='gray', linestyle=':', linewidth=1, 
                    label=f'{p}th Percentile: {p_val:.1f}')
    
    # Add title and labels
    plt.title(title if title else f'Distribution of {metric} with Outliers Highlighted')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Count')
    
    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend
    plt.legend(loc='best')
    
    # Add stats in a textbox
    if has_outlier_flag:
        outlier_count = df[f'{metric}_is_outlier'].sum()
        outlier_pct = (outlier_count / len(df)) * 100
        
        # Get high/low counts if available
        if f'{metric}_outlier_direction' in df.columns:
            high_count = sum((df[f'{metric}_is_outlier']) & 
                            (df[f'{metric}_outlier_direction'] == 'high'))
            low_count = sum((df[f'{metric}_is_outlier']) & 
                            (df[f'{metric}_outlier_direction'] == 'low'))
            
            stats_text = (
                f"Total stores: {len(df)}\n"
                f"Outliers: {outlier_count} ({outlier_pct:.1f}%)\n"
                f"High outliers: {high_count} ({high_count/len(df)*100:.1f}%)\n"
                f"Low outliers: {low_count} ({low_count/len(df)*100:.1f}%)\n\n"
                f"Mean: {mean:.1f}, Median: {median:.1f}\n"
                f"Std Dev: {df[metric].std():.1f}\n"
                f"IQR: {IQR:.1f} (Q1: {Q1:.1f}, Q3: {Q3:.1f})"
            )
        else:
            stats_text = (
                f"Total stores: {len(df)}\n"
                f"Outliers: {outlier_count} ({outlier_pct:.1f}%)\n\n"
                f"Mean: {mean:.1f}, Median: {median:.1f}\n"
                f"Std Dev: {df[metric].std():.1f}\n"
                f"IQR: {IQR:.1f} (Q1: {Q1:.1f}, Q3: {Q3:.1f})"
            )
    else:
        stats_text = (
            f"Total stores: {len(df)}\n"
            f"Mean: {mean:.1f}, Median: {median:.1f}\n"
            f"Std Dev: {df[metric].std():.1f}\n"
            f"IQR: {IQR:.1f} (Q1: {Q1:.1f}, Q3: {Q3:.1f})"
        )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.show()

## 7. Main Function for Coordinating Analysis

def run_advanced_outlier_analysis(data_file=None, n_stores=100, n_metrics=10, n_weeks=5, selected_metrics=None):
    """
    Main function to coordinate the analysis with enhanced visualizations.
    
    Parameters:
    - data_file: Path to data file. If None, sample data is created.
    - n_stores: Number of stores for sample data
    - n_metrics: Number of metrics for sample data
    - n_weeks: Number of weeks for sample data
    - selected_metrics: List of specific metrics to analyze. If None, will ask interactively.
    """
    # Load data or create sample
    if data_file:
        # Load from file
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        df = optimize_dataframe(df)
    else:
        # Create sample data
        df = create_sample_data(n_stores=n_stores, n_metrics=n_metrics, n_weeks=n_weeks)
    
    # Determine store column name
    store_col = 'store' if 'store' in df.columns else 'store_number'
    
    print(f"Working with {df[store_col].nunique()} stores and {df['metric'].nunique()} metrics")
    
    # Aggregate metrics by store
    aggregated_df = aggregate_metrics_by_store(df)
    
    # Get all available metrics from the aggregated data
    all_available_metrics = [col for col in aggregated_df.columns 
                           if col != 'store' and col != 'store_number' and col != 'cluster' 
                           and not col.endswith('_outlier') 
                           and not col.endswith('_percentile')
                           and not col.endswith('_is_outlier')]
    
    print(f"\nFound {len(all_available_metrics)} available metrics in the dataset:")
    for i, metric in enumerate(all_available_metrics):
        print(f"{i+1}. {metric}")
    
    # If no metrics are provided, ask user which ones to analyze
    if selected_metrics is None:
        print("\nWhich metrics would you like to analyze?")
        print("Enter the numbers (e.g., '1,3,5') or names (e.g., 'sales,visits') of the metrics, comma-separated:")
        user_input = input("> ")
        
        # Parse user input
        if ',' in user_input:
            parts = [p.strip() for p in user_input.split(',')]
            
            # Check if they entered numbers or names
            if all(p.isdigit() for p in parts):
                # User entered numbers
                indices = [int(p) - 1 for p in parts]  # Convert to 0-based index
                metrics_list = [all_available_metrics[i] for i in indices if 0 <= i < len(all_available_metrics)]
            else:
                # User entered names
                metrics_list = [m for m in parts if m in all_available_metrics]
        elif user_input.strip().isdigit():
            # Single number entered
            index = int(user_input.strip()) - 1
            if 0 <= index < len(all_available_metrics):
                metrics_list = [all_available_metrics[index]]
            else:
                print(f"Invalid metric number. Using the first metric: {all_available_metrics[0]}")
                metrics_list = [all_available_metrics[0]]
        else:
            # Single name entered
            if user_input.strip() in all_available_metrics:
                metrics_list = [user_input.strip()]
            else:
                print(f"Metric '{user_input.strip()}' not found. Using the first metric: {all_available_metrics[0]}")
                metrics_list = [all_available_metrics[0]]
    else:
        # Use the provided metrics if they exist in the dataset
        metrics_list = [m for m in selected_metrics if m in all_available_metrics]
        if not metrics_list:
            print(f"None of the provided metrics were found. Using the first available metric: {all_available_metrics[0]}")
            metrics_list = [all_available_metrics[0]]
    
    print(f"\nAnalyzing {len(metrics_list)} metrics: {metrics_list}")
    
    # Apply clustering
    clustered_df = cluster_stores(aggregated_df, n_clusters=5)
    
    # Calculate outlier scores using multiple methods
    outlier_results = calculate_outlier_scores(clustered_df, metrics_list)
    
    # Select a sample metric for demonstration
    sample_metric = metrics_list[0]
    
    print("\n1. MULTI-DIMENSIONAL SCATTER PLOT WITH OUTLIERS")
    if len(metrics_list) >= 2:
        # Select two metrics for visualization
        metric_x = metrics_list[0]
        metric_y = metrics_list[1]
        visualize_multi_dimensional_outliers(
            outlier_results, metric_x, metric_y, 
            title=f'Store Performance: {metric_x} vs {metric_y} with Outliers Highlighted'
        )
    else:
        print("Need at least 2 metrics for scatter plot. Skipping this visualization.")
    
    print("\n2. DEVIATION FROM EXPECTED CHART")
    visualize_deviation_from_expected(
        outlier_results, sample_metric, by_cluster=True, n_stores=30
    )
    
    print("\n3. ANOMALY HEATMAP ACROSS MULTIPLE METRICS")
    # Use all selected metrics for the heatmap
    visualize_anomaly_heatmap(
        outlier_results, metrics_list, n_stores=25
    )
    
    print("\n4. PCA PROJECTION WITH OUTLIERS")
    if len(metrics_list) >= 2:
        visualize_pca_with_outliers(
            outlier_results, metrics_list
        )
    else:
        print("Need at least 2 metrics for PCA. Skipping this visualization.")
    
    print("\n5. METRIC DISTRIBUTION WITH OUTLIERS HIGHLIGHTED")
    visualize_outlier_metrics_distribution(
        outlier_results, sample_metric
    )
    
    return outlier_results

# Run the analysis with default settings
# For real use, specify a data file or increase the sample size
if __name__ == "__main__":
    # Check if specific metrics are provided as command line arguments
    if len(sys.argv) > 1:
        selected_metrics = sys.argv[1:]
        results = run_advanced_outlier_analysis(selected_metrics=selected_metrics)
    else:
        results = run_advanced_outlier_analysis()

