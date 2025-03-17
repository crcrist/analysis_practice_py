import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

def load_and_preprocess_data(file_path="store_metrics.csv"):
    """
    Load and preprocess the store metrics data
    
    This function:
    1. Loads data from the specified CSV file
    2. Calculates metric values based on whether a metric is a ratio or not
    3. Handles potential division by zero or null denominators
    4. Converts year_week to datetime for better plotting
    
    Parameters:
    ----------
    file_path : str
        Path to the CSV file with store metrics
        
    Returns:
    -------
    pandas.DataFrame
        Processed DataFrame with calculated metric values
    """
    # Load data from CSV
    df = pd.read_csv(file_path)
    
    # Calculate metric values based on isratio flag
    # If isratio is True: value = numerator / denominator
    # If isratio is False: value = numerator
    df['value'] = np.where(
        df['isratio'], 
        df['numerator'] / df['denominator'].replace({0: np.nan}), 
        df['numerator']
    )
    
    # Handle potential division by zero or null denominators
    # Replace infinite values with NaN
    df['value'] = df['value'].replace([np.inf, -np.inf], np.nan)
    
    # Convert year_week to datetime for better plotting
    # Assuming year_week is in YYYYWW format where WW is ISO week number
    try:
        df['date'] = df['year_week'].astype(str).apply(
            lambda x: datetime.strptime(x + '1', '%Y%W%w')  # Sunday of that week
        )
    except:
        print("Warning: Could not convert year_week to date. Format may be incorrect.")
    
    return df

def detect_outliers(df, method='zscore', threshold=3.0):
    """
    Detect outliers in the data using statistical methods
    
    This function identifies outliers in metric values using one or both of:
    1. Z-score method: Values more than 'threshold' standard deviations from the mean
    2. IQR method: Values outside Q1-threshold*IQR or Q3+threshold*IQR
    
    The detection is performed separately for each metric to account for different
    scales and distributions across metrics.
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with store metrics
    method : str
        Outlier detection method: 'zscore', 'iqr', or 'both'
    threshold : float
        Z-score threshold or IQR multiplier (higher = more conservative detection)
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with added outlier flag columns
    """
    # Work with a copy to avoid modifying the original
    result = df.copy()
    
    # Initialize outlier flag columns
    if method in ['zscore', 'both']:
        result['is_zscore_outlier'] = False
    if method in ['iqr', 'both']:
        result['is_iqr_outlier'] = False
    result['is_outlier'] = False
    
    # Group by metric to calculate outliers within each metric type
    # This is important because different metrics have different scales and distributions
    for metric_name in df['metric'].unique():
        metric_data = df[df['metric'] == metric_name]
        
        # Skip if not enough data for reliable outlier detection
        if len(metric_data) < 4:
            print(f"Warning: Not enough data for outlier detection in {metric_name}")
            continue
            
        # Get the values for this metric
        values = metric_data['value'].dropna()
        
        # Z-score method (standard deviations from mean)
        if method in ['zscore', 'both']:
            # Calculate absolute z-scores (how many SDs from mean)
            z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
            is_zscore_outlier = np.abs(z_scores) > threshold
            
            # Map back to the original indices
            z_outlier_indices = values.index[is_zscore_outlier]
            result.loc[z_outlier_indices, 'is_zscore_outlier'] = True
        
        # IQR method (interquartile range)
        if method in ['iqr', 'both']:
            # Calculate quartiles and IQR
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            # Identify values outside the "whiskers"
            is_iqr_outlier = (values < (Q1 - threshold * IQR)) | (values > (Q3 + threshold * IQR))
            
            # Map back to the original indices
            iqr_outlier_indices = values.index[is_iqr_outlier]
            result.loc[iqr_outlier_indices, 'is_iqr_outlier'] = True
    
    # Combine outlier detection methods if both were used
    if method == 'both':
        result['is_outlier'] = result['is_zscore_outlier'] | result['is_iqr_outlier']
    else:
        result['is_outlier'] = result[f'is_{method}_outlier']
    
    return result

def generate_insights(df):
    """
    Generate key insights from the store metrics data
    
    This function performs comprehensive analysis to extract actionable insights:
    1. Outlier analysis - Which metrics have the most outliers and which stores
    2. Performance analysis - Top and bottom performing stores by metric
    3. Trend analysis - Whether metrics are trending up or down over time
    4. Consistency analysis - Which stores have the most/least consistent performance
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with store metrics and outlier flags
        
    Returns:
    -------
    dict
        Dictionary containing various insights, including:
        - outlier_counts: Count of outliers by metric
        - stores_with_most_outliers: Stores with the most outliers
        - metrics_summary: Statistical summary by store and metric
        - top_performers: Top performing stores by metric
        - bottom_performers: Bottom performing stores by metric
        - trend_analysis: Trend direction and significance by metric
        - most_consistent_stores: Most consistent stores by metric
        - least_consistent_stores: Least consistent stores by metric
    """
    insights = {}
    
    # SECTION 1: OUTLIER ANALYSIS
    # ===========================
    
    # Count outliers per metric
    outlier_counts = df[df['is_outlier'] == True].groupby('metric').size()
    insights['outlier_counts'] = outlier_counts.to_dict()
    
    # Identify stores with most outliers
    store_outliers = df[df['is_outlier'] == True].groupby('store_number').size().sort_values(ascending=False)
    insights['stores_with_most_outliers'] = store_outliers.head(5).to_dict()
    
    # SECTION 2: PERFORMANCE ANALYSIS
    # ==============================
    
    # Calculate comprehensive metrics summary by store
    metrics_by_store = []
    for metric_name in df['metric'].unique():
        # Filter data for this metric
        metric_df = df[df['metric'] == metric_name]
        
        # Group by store and calculate key statistics
        store_summary = metric_df.groupby('store_number')['value'].agg(['mean', 'std', 'min', 'max'])
        store_summary['metric'] = metric_name
        
        # Add coefficient of variation (relative standard deviation)
        # This shows consistency relative to the mean (lower is more consistent)
        store_summary['cv'] = store_summary['std'] / store_summary['mean']
        
        # Calculate rank (1 is best) based on mean value
        store_summary['rank'] = store_summary['mean'].rank(ascending=False)
        
        metrics_by_store.append(store_summary.reset_index())
    
    # Combine all metrics summaries
    metrics_summary = pd.concat(metrics_by_store)
    insights['metrics_summary'] = metrics_summary
    
    # Identify top and bottom performers by metric
    top_performers = {}
    bottom_performers = {}
    
    for metric_name in df['metric'].unique():
        # Filter summary for this metric
        metric_summary = metrics_summary[metrics_summary['metric'] == metric_name]
        
        # Top 3 performers (lowest rank number = highest performance)
        top_3 = metric_summary.nsmallest(3, 'rank')[['store_number', 'mean', 'rank']]
        top_performers[metric_name] = top_3.set_index('store_number').to_dict('index')
        
        # Bottom 3 performers (highest rank number = lowest performance)
        bottom_3 = metric_summary.nlargest(3, 'rank')[['store_number', 'mean', 'rank']]
        bottom_performers[metric_name] = bottom_3.set_index('store_number').to_dict('index')
    
    insights['top_performers'] = top_performers
    insights['bottom_performers'] = bottom_performers
    
    # SECTION 3: TREND ANALYSIS
    # =========================
    
    # Analyze whether metrics are trending up or down over time
    trend_analysis = {}
    
    for metric_name in df['metric'].unique():
        metric_df = df[df['metric'] == metric_name]
        
        # Get weekly averages across all stores
        weekly_avg = metric_df.groupby('year_week')['value'].mean().reset_index()
        
        # Check if there's enough data for trend analysis
        if len(weekly_avg) >= 3:
            # Simple linear regression to determine trend
            x = np.arange(len(weekly_avg))
            y = weekly_avg['value'].values
            
            # Calculate trend line parameters and statistics
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trend_analysis[metric_name] = {
                'slope': slope,
                'direction': 'up' if slope > 0 else 'down',
                'strength': abs(r_value),  # Correlation coefficient (0-1)
                'p_value': p_value,
                'significant': p_value < 0.05  # Standard statistical significance
            }
    
    insights['trend_analysis'] = trend_analysis
    
    # SECTION 4: CONSISTENCY ANALYSIS
    # ===============================
    
    # Identify most and least consistent stores 
    # Consistency is measured by coefficient of variation (CV)
    # First flatten the metrics_summary to avoid MultiIndex issues
    consistency = metrics_summary[['store_number', 'metric', 'cv']].copy()
    
    # Find most consistent stores (lowest CV) for each metric
    most_consistent = consistency.sort_values('cv').groupby('metric').first().reset_index()
    
    # Find least consistent stores (highest CV) for each metric
    least_consistent = consistency.sort_values('cv', ascending=False).groupby('metric').first().reset_index()
    
    insights['most_consistent_stores'] = dict(zip(most_consistent['metric'], 
                                                   zip(most_consistent['store_number'], most_consistent['cv'])))
    insights['least_consistent_stores'] = dict(zip(least_consistent['metric'], 
                                                    zip(least_consistent['store_number'], least_consistent['cv'])))
    
    return insights

def rank_metrics_by_significance(df, top_n=10, min_periods=3):
    """
    Rank metrics by their statistical significance of change over time
    
    This function:
    1. For each metric, calculates the magnitude of change over time
    2. Determines statistical significance of trends
    3. Ranks metrics by a composite score based on:
       - Absolute percentage change (start to end)
       - Statistical significance (p-value)
       - Consistency of the trend (R-squared value)
       - Presence of outliers
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with store metrics that includes outlier flags
    top_n : int
        Number of most significant metrics to return
    min_periods : int
        Minimum number of time periods required for trend analysis
        
    Returns:
    -------
    pandas.DataFrame
        Top N metrics ranked by significance, with significance metrics
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    metric_significance = []
    
    # Group by metric to analyze each one separately
    for metric_name in df['metric'].unique():
        metric_df = df[df['metric'] == metric_name]
        
        # Calculate weekly average across all stores
        weekly_avg = metric_df.groupby('year_week')['value'].mean().reset_index()
        weekly_avg = weekly_avg.sort_values('year_week')
        
        # Skip metrics with too few data points
        if len(weekly_avg) < min_periods:
            continue
            
        # Calculate percentage change from first to last period
        first_value = weekly_avg['value'].iloc[0]
        last_value = weekly_avg['value'].iloc[-1]
        
        # Avoid division by zero
        if first_value == 0:
            first_value = 0.0001  # Small non-zero value
            
        pct_change = ((last_value - first_value) / abs(first_value)) * 100
        abs_pct_change = abs(pct_change)
        
        # Perform linear regression to assess trend
        x = np.arange(len(weekly_avg))
        y = weekly_avg['value'].values
        
        # Calculate trend statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate median absolute deviation as a measure of volatility
        median_value = np.median(weekly_avg['value'])
        mad = np.median(np.abs(weekly_avg['value'] - median_value))
        
        # Calculate percentage of outliers for this metric
        outlier_pct = (metric_df['is_outlier'].sum() / len(metric_df)) * 100
        
        # Composite significance score:
        # Higher for:
        # - Large percentage changes
        # - Statistically significant trends (low p-value)
        # - Consistent trends (high R-squared)
        # - High volatility (measured by MAD relative to median)
        # - Presence of outliers
        
        # Convert p-value to significance factor (higher = more significant)
        # p-value of 0.05 or less is considered statistically significant
        significance_factor = 1 - min(p_value, 0.99)  # Invert p-value, cap at 0.99
        
        # R-squared measures how well the trend line fits the data
        r_squared = r_value ** 2
        
        # Calculate composite score - you can adjust weights as needed
        composite_score = (
            (0.4 * abs_pct_change) +  # 40% weight on absolute percentage change
            (0.3 * significance_factor * 100) +  # 30% weight on statistical significance
            (0.2 * r_squared * 100) +  # 20% weight on trend consistency
            (0.1 * outlier_pct)  # 10% weight on presence of outliers
        )
        
        # Collect all metrics for this metric
        metric_significance.append({
            'metric': metric_name,
            'abs_pct_change': abs_pct_change,
            'pct_change': pct_change,
            'direction': 'up' if pct_change > 0 else 'down',
            'p_value': p_value,
            'significant': p_value < 0.05,
            'r_squared': r_squared,
            'volatility': mad / (median_value if median_value != 0 else 0.0001),
            'outlier_pct': outlier_pct,
            'composite_score': composite_score,
            'slope': slope,
            'num_periods': len(weekly_avg)
        })
    
    # Convert to DataFrame and sort by composite score (descending)
    significance_df = pd.DataFrame(metric_significance)
    
    if len(significance_df) == 0:
        return pd.DataFrame(columns=['metric', 'composite_score', 'abs_pct_change', 'direction', 
                                    'p_value', 'significant', 'r_squared', 'outlier_pct'])
    
    # Sort by composite score (descending)
    significance_df = significance_df.sort_values('composite_score', ascending=False)
    
    # Return the top N most significant metrics
    return significance_df.head(top_n)

def plot_metric_summary(df, metric_name, output_dir=None):
    """
    Create summary visualizations for a metric that work well with large store counts
    
    Parameters:
    - df: DataFrame with store metrics and outlier flags
    - metric_name: The specific metric to plot
    - output_dir: Directory to save the plot, if None, just show it
    
    Returns:
    - Path to saved plot if output_dir is provided
    """
    # Filter data for the specified metric
    metric_df = df[df['metric'] == metric_name].copy()
    
    if len(metric_df) == 0:
        print(f"No data for metric: {metric_name}")
        return None
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 1. Overall Distribution Histogram (top left)
    ax1 = axs[0, 0]
    sns.histplot(metric_df['value'], kde=True, ax=ax1, color='skyblue')
    ax1.set_title(f'Distribution of {metric_name}', fontsize=12)
    ax1.set_xlabel(f'{metric_name} value')
    ax1.set_ylabel('Frequency')
    
    # Mark outliers on histogram with red lines
    outliers = metric_df[metric_df['is_outlier'] == True]['value']
    for outlier in outliers:
        ax1.axvline(x=outlier, color='red', alpha=0.3, linestyle='--')
    
    # 2. Top and Bottom Performers (top right)
    ax2 = axs[0, 1]
    
    # Get average by store
    store_avg = metric_df.groupby('store_number')['value'].mean().reset_index()
    
    # Get top 5 and bottom 5 stores
    top_5 = store_avg.nlargest(5, 'value')
    bottom_5 = store_avg.nsmallest(5, 'value')
    
    # Sort each group by value (descending for top, ascending for bottom)
    top_5 = top_5.sort_values('value', ascending=False)
    bottom_5 = bottom_5.sort_values('value', ascending=True)  # Sort ascending for visual clarity
    
    # Combine into one DataFrame for plotting
    top_bottom = pd.concat([top_5, bottom_5])
    top_bottom['performance'] = ['Top 5'] * 5 + ['Bottom 5'] * 5
    
    # Convert store_number to string for better x-axis labels
    top_bottom['store_number'] = top_bottom['store_number'].astype(str)
    
    # Create grouped bar chart
    sns.barplot(x='store_number', y='value', hue='performance', 
                data=top_bottom, ax=ax2, palette={'Top 5': 'green', 'Bottom 5': 'tomato'})
    
    ax2.set_title(f'Top 5 and Bottom 5 Stores - {metric_name}', fontsize=12)
    ax2.set_xlabel('Store Number')
    ax2.set_ylabel(f'{metric_name} (Avg)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Time Trend (bottom left)
    ax3 = axs[1, 0]
    
    # Calculate weekly average across all stores
    if 'year_week' in metric_df.columns:
        weekly_avg = metric_df.groupby('year_week')['value'].agg(['mean', 'std']).reset_index()
        weekly_avg = weekly_avg.sort_values('year_week')
        
        # Plot the trend line with confidence interval
        ax3.plot(weekly_avg.index, weekly_avg['mean'], marker='o', color='blue')
        ax3.fill_between(
            weekly_avg.index, 
            weekly_avg['mean'] - weekly_avg['std'],
            weekly_avg['mean'] + weekly_avg['std'],
            alpha=0.2,
            color='blue'
        )
        
        ax3.set_title(f'Weekly Trend - {metric_name}', fontsize=12)
        ax3.set_xlabel('Week Index')
        ax3.set_ylabel(f'{metric_name} (Avg)')
        
        # Set x-ticks to show week numbers
        if len(weekly_avg) <= 20:  # Only show all weeks if there aren't too many
            ax3.set_xticks(weekly_avg.index)
            ax3.set_xticklabels(weekly_avg['year_week'], rotation=45)
        else:
            # Show a subset of weeks
            step = max(1, len(weekly_avg) // 10)
            ticks = weekly_avg.index[::step]
            ax3.set_xticks(ticks)
            ax3.set_xticklabels(weekly_avg['year_week'].iloc[::step], rotation=45)
    else:
        ax3.text(0.5, 0.5, 'No time data available', ha='center', va='center')
    
    # 4. Test vs Control comparison (bottom right) - if applicable
    ax4 = axs[1, 1]
    
    if 'test_flag' in metric_df.columns:
        # Create box plot for test vs control
        sns.boxplot(x='test_flag', y='value', data=metric_df, ax=ax4, palette='Set3')
        
        # Add swarm plot with outliers highlighted
        non_outliers = metric_df[metric_df['is_outlier'] == False]
        outliers = metric_df[metric_df['is_outlier'] == True]
        
        sns.swarmplot(x='test_flag', y='value', data=non_outliers, ax=ax4, 
                      color='black', alpha=0.5, size=3)
        sns.swarmplot(x='test_flag', y='value', data=outliers, ax=ax4, 
                      color='red', alpha=0.7, size=5)
        
        # Calculate and show averages
        group_means = metric_df.groupby('test_flag')['value'].mean()
        for i, mean_val in enumerate(group_means):
            ax4.text(i, mean_val, f'Avg: {mean_val:.2f}', 
                     ha='center', va='bottom', color='darkblue', fontweight='bold')
        
        ax4.set_title(f'Test vs Control - {metric_name}', fontsize=12)
        ax4.set_xlabel('Group (0=Control, 1=Test)')
        ax4.set_ylabel(f'{metric_name}')
    else:
        # Create a box plot showing overall distribution
        sns.boxplot(y='value', data=metric_df, ax=ax4, color='lightblue')
        sns.swarmplot(y='value', data=metric_df[metric_df['is_outlier']], ax=ax4, color='red', size=5)
        
        ax4.set_title(f'Overall Distribution with Outliers - {metric_name}', fontsize=12)
        ax4.set_ylabel(f'{metric_name}')
        ax4.set_xlabel('')
    
    # Overall title
    plt.suptitle(f'Summary Analysis: {metric_name}', fontsize=16)
    plt.tight_layout()
    
    # Save or show the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{metric_name}_summary.png')
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close()
        return file_path
    else:
        plt.show()
        return None

def plot_outliers(df, metric_name, output_dir=None):
    """
    Create summary visualizations for outlier analysis
    
    Parameters:
    - df: DataFrame with store metrics and outlier flags
    - metric_name: The specific metric to plot
    - output_dir: Directory to save the plot, if None, just show it
    
    Returns:
    - Path to saved plot if output_dir is provided
    """
    return plot_metric_summary(df, metric_name, output_dir)

# Add a visualization specifically for the most significant metrics
def plot_significant_metrics(significant_metrics, df, output_dir=None):
    """
    Create a summary visualization of the most significant metrics
    
    Parameters:
    - significant_metrics: DataFrame with metrics ranked by significance
    - df: Original DataFrame with all metrics data
    - output_dir: Directory to save the plot, if None, just show it
    
    Returns:
    - Path to saved plot if output_dir is provided
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    from matplotlib.gridspec import GridSpec
    
    # Get the top 10 metrics or less if fewer are available
    top_metrics = significant_metrics.head(min(10, len(significant_metrics)))
    
    if len(top_metrics) == 0:
        print("No significant metrics to plot.")
        return None
        
    # Create a larger figure to show all the metrics
    plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2])
    
    # 1. Bar chart of composite scores (top left)
    ax1 = plt.subplot(gs[0, 0])
    
    # Plot the metrics by composite score
    score_chart = sns.barplot(
        x='metric', 
        y='composite_score',
        data=top_metrics.sort_values('composite_score', ascending=False),
        ax=ax1,
        palette='viridis'
    )
    
    ax1.set_title('Metrics Ranked by Significance Score', fontsize=14)
    ax1.set_ylabel('Composite Score')
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    
    # 2. Percentage change visualization (top right)
    ax2 = plt.subplot(gs[0, 1])
    
    # Create color mapping based on direction
    colors = ['green' if x > 0 else 'red' for x in top_metrics['pct_change']]
    
    # Plot horizontal bars for percentage change
    bars = ax2.barh(
        top_metrics['metric'], 
        top_metrics['pct_change'],
        color=colors
    )
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 1 if width > 0 else width - 5
        ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center', fontsize=9)
    
    ax2.set_title('Percentage Change', fontsize=14)
    ax2.set_xlabel('% Change')
    ax2.set_ylabel('')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # 3. Trend visualization (bottom)
    ax3 = plt.subplot(gs[1, :])
    
    # Plot trend lines for all top metrics
    for i, (_, row) in enumerate(top_metrics.iterrows()):
        metric_name = row['metric']
        metric_df = df[df['metric'] == metric_name].copy()
        
        # Calculate weekly average across all stores
        weekly_avg = metric_df.groupby('year_week')['value'].mean().reset_index()
        weekly_avg = weekly_avg.sort_values('year_week')
        
        # Normalize values to start at 100 for better comparison
        first_val = weekly_avg['value'].iloc[0]
        weekly_avg['normalized'] = (weekly_avg['value'] / first_val) * 100
        
        # Plot the trend line
        ax3.plot(
            weekly_avg.index, 
            weekly_avg['normalized'], 
            marker='o', 
            markersize=4,
            linewidth=2,
            alpha=0.8,
            label=f"{metric_name} ({row['pct_change']:.1f}%)"
        )
    
    ax3.set_title('Normalized Trends of Top Metrics (Starting at 100)', fontsize=14)
    ax3.set_xlabel('Time Period')
    ax3.set_ylabel('Normalized Value (Starting at 100)')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle('Summary of Most Significant Metrics', fontsize=16)
    plt.tight_layout()
    
    # Save or show the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, 'significant_metrics_summary.png')
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close()
        return file_path
    else:
        plt.show()
        return None

def format_insights_email(insights):
    """
    Format insights into an email-ready format
    
    Parameters:
    - insights: Dictionary of insights from generate_insights function
    
    Returns:
    - String containing formatted email
    """
    email = "STORE METRICS ANALYSIS SUMMARY\n"
    email += "=" * 50 + "\n\n"
    
    # Overall stats
    email += "OUTLIER DETECTION\n"
    email += "-" * 30 + "\n"
    
    for metric, count in insights['outlier_counts'].items():
        email += f"- {metric}: {count} outliers detected\n"
    
    email += "\nStores with most outliers:\n"
    for store, count in insights['stores_with_most_outliers'].items():
        email += f"- Store #{store}: {count} outliers\n"
    
    # Top and bottom performers
    email += "\n\nPERFORMANCE SUMMARY\n"
    email += "-" * 30 + "\n"
    
    for metric in insights['top_performers'].keys():
        email += f"\n{metric.upper()}\n"
        
        # Top performers
        email += "Top performers:\n"
        for store, data in insights['top_performers'][metric].items():
            email += f"- Store #{store}: {data['mean']:.2f} (Rank: {data['rank']:.0f})\n"
        
        # Bottom performers
        email += "\nBottom performers:\n"
        for store, data in insights['bottom_performers'][metric].items():
            email += f"- Store #{store}: {data['mean']:.2f} (Rank: {data['rank']:.0f})\n"
    
    # Trend analysis
    email += "\n\nTREND ANALYSIS\n"
    email += "-" * 30 + "\n"
    
    for metric, trend in insights['trend_analysis'].items():
        direction = "↑" if trend['direction'] == 'up' else "↓"
        significance = "significant" if trend['significant'] else "not significant"
        email += f"- {metric}: Trending {direction} ({significance}, strength: {trend['strength']:.2f})\n"
    
    # Consistency analysis
    email += "\n\nCONSISTENCY ANALYSIS\n"
    email += "-" * 30 + "\n"
    
    email += "Most consistent stores by metric:\n"
    for metric, (store, cv) in insights['most_consistent_stores'].items():
        email += f"- {metric}: Store #{store} (CV: {cv:.2f})\n"
    
    email += "\nLeast consistent stores by metric:\n"
    for metric, (store, cv) in insights['least_consistent_stores'].items():
        email += f"- {metric}: Store #{store} (CV: {cv:.2f})\n"
    
    email += "\n\nSee attached visualizations for more detailed analysis. Each metric has a summary chart showing:\n"
    email += "- Overall distribution with outliers highlighted\n"
    email += "- Top 5 and bottom 5 performing stores\n"
    email += "- Weekly trend analysis\n"
    if any('test_flag' in df.columns for df in [insight for insight in insights.values() if isinstance(insight, pd.DataFrame)]):
        email += "- Test vs. control comparison\n"
    
    return email


def analyze_store_metrics(file_path="store_metrics.csv", output_dir="output", 
                      test_flag_column=None, outlier_threshold=2.5, 
                      top_metrics=10, analyze_all=False):
    """
    Main function to analyze store metrics data
    
    Parameters:
    - file_path: Path to the CSV file with store metrics
    - output_dir: Directory to save plots and output files
    - test_flag_column: Column name that indicates test/control groups (if applicable)
    - outlier_threshold: Threshold for outlier detection (Z-score or IQR multiplier)
    - top_metrics: Number of most significant metrics to focus on
    - analyze_all: If True, analyze all metrics; if False, focus on top significant metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data (with chunking for large files)
    print(f"Loading data from {file_path}...")
    try:
        # Try normal loading first (faster for smaller files)
        df = load_and_preprocess_data(file_path)
        print(f"Loaded {len(df)} records for {df['store_number'].nunique()} stores " +
              f"over {df['year_week'].nunique()} weeks.")
    except MemoryError:
        print("File too large for direct loading. Processing in chunks...")
        # For very large datasets, implement chunking here
        # This would require modifying the load_and_preprocess_data function
        # to support chunked processing
        raise NotImplementedError("Chunked processing not implemented yet.")
    
    # Check data size and complexity
    num_stores = df['store_number'].nunique()
    num_weeks = df['year_week'].nunique()
    num_metrics = df['metric'].nunique()
    total_records = len(df)
    
    print(f"Data summary:")
    print(f"- Stores: {num_stores}")
    print(f"- Weeks: {num_weeks}")
    print(f"- Metrics: {num_metrics}")
    print(f"- Total records: {total_records}")
    
    # Detect outliers with progress tracking for large datasets
    print("Detecting outliers...")
    metrics = df['metric'].unique()
    
    # For very large datasets, process each metric separately to save memory
    if total_records > 100000:  # Arbitrary threshold, adjust as needed
        print("Large dataset detected. Processing metrics separately...")
        df_with_outliers_list = []
        
        for i, metric_name in enumerate(metrics, 1):
            print(f"Processing metric {i}/{len(metrics)}: {metric_name}")
            metric_df = df[df['metric'] == metric_name].copy()
            metric_df_with_outliers = detect_outliers(metric_df, method='both', threshold=outlier_threshold)
            df_with_outliers_list.append(metric_df_with_outliers)
            
        # Combine results
        df_with_outliers = pd.concat(df_with_outliers_list)
    else:
        # For smaller datasets, process all at once
        df_with_outliers = detect_outliers(df, method='both', threshold=outlier_threshold)
    
    # Save processed data with outlier flags
    processed_file = os.path.join(output_dir, "processed_data.csv")
    df_with_outliers.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")
    
    # Rank metrics by significance and identify the most important ones
    print(f"Ranking metrics by significance...")
    significant_metrics = rank_metrics_by_significance(df_with_outliers, top_n=top_metrics)
    
    # Save the list of significant metrics
    sig_metrics_file = os.path.join(output_dir, "significant_metrics.csv")
    significant_metrics.to_csv(sig_metrics_file, index=False)
    print(f"Significant metrics saved to {sig_metrics_file}")
    
    # Print the list of significant metrics
    print("\nMost significant metrics:")
    for i, (_, row) in enumerate(significant_metrics.iterrows(), 1):
        print(f"{i}. {row['metric']} - Score: {row['composite_score']:.2f}, " +
             f"Change: {row['pct_change']:.2f}% ({row['direction']}), " +
             f"p-value: {row['p_value']:.4f}")
    
    # Determine which metrics to analyze further
    if analyze_all:
        metrics_to_analyze = metrics
        print("\nAnalyzing all metrics...")
    else:
        metrics_to_analyze = significant_metrics['metric'].tolist()
        print(f"\nFocusing analysis on the top {len(metrics_to_analyze)} most significant metrics...")
    
    # Filter the data to include only the metrics we're analyzing
    focused_df = df_with_outliers[df_with_outliers['metric'].isin(metrics_to_analyze)]
    
    # Generate insights on the focused metrics
    print("Generating insights...")
    insights = generate_insights(focused_df)
    
    # Format insights for email
    email_text = format_insights_email(insights)
    
    # Add the significance ranking to the email
    email_text += "\n\nMOST SIGNIFICANT METRICS\n"
    email_text += "-" * 30 + "\n"
    
    for i, (_, row) in enumerate(significant_metrics.iterrows(), 1):
        sig_status = "statistically significant" if row['significant'] else "not statistically significant"
        email_text += f"{i}. {row['metric']}:\n"
        email_text += f"   - Change: {row['pct_change']:.2f}% ({row['direction']})\n"
        email_text += f"   - Trend: {sig_status} (p-value: {row['p_value']:.4f})\n"
        email_text += f"   - Consistency (R²): {row['r_squared']:.2f}\n"
        email_text += f"   - Outliers: {row['outlier_pct']:.1f}% of data points\n"
        email_text += f"   - Overall Significance Score: {row['composite_score']:.2f}\n\n"
    
    # Save the updated email
    email_file = os.path.join(output_dir, "store_metrics_insights.txt")
    with open(email_file, 'w') as f:
        f.write(email_text)
    print(f"Insights email saved to {email_file}")
    
    # Generate summary plots for the focused metrics
    print("Generating plots...")
    for i, metric_name in enumerate(metrics_to_analyze, 1):
        print(f"Creating summary visualization for metric {i}/{len(metrics_to_analyze)}: {metric_name}")
        plot_outliers(focused_df, metric_name, output_dir)
    
    print(f"Analysis complete! All output saved to {output_dir}")
    
    return df_with_outliers, insights, email_text, significant_metrics

if __name__ == "__main__":
    # Updated main execution block
    df, insights, email, significant_metrics = analyze_store_metrics(
        "store_metrics.csv", 
        "analysis_output",
        top_metrics=10,
        analyze_all=False
    )
    
    # Create a summary visualization of the significant metrics
    plot_significant_metrics(significant_metrics, df, "analysis_output")
    
    # Print the email to console
    print("\nSample Email Content:")
    print("=" * 50)
    print(email)
