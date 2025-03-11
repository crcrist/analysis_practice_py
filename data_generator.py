import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import csv

def generate_fake_store_data(
    num_stores=10,
    start_week="202401",
    num_weeks=12,
    metrics=["sales", "visits", "items_per_visit", "conversion_rate", "avg_ticket"],
    output_file="store_metrics.csv"
):
    """
    Generate fake store metrics data for testing.
    
    Parameters:
    - num_stores: Number of stores to generate data for
    - start_week: Starting week in YYYYWW format
    - num_weeks: Number of weeks to generate data for
    - metrics: List of metrics to generate
    - output_file: Output CSV file name
    """
    # Convert start_week to epoch_week (assuming epoch starts at 2000-01-01)
    start_date = datetime.strptime(start_week + '1', '%Y%W%w')  # Sunday of that week
    epoch_start = datetime(2000, 1, 1)
    start_epoch_week = int((start_date - epoch_start).days / 7) + 1
    
    # Define metric properties
    metric_properties = {
        "sales": {"is_ratio": False, "base_value": 10000, "variation": 3000},
        "visits": {"is_ratio": False, "base_value": 500, "variation": 150},
        "items_per_visit": {"is_ratio": True, "base_value": 2.5, "variation": 0.8, "denominator_base": 500},
        "conversion_rate": {"is_ratio": True, "base_value": 0.3, "variation": 0.1, "denominator_base": 500},
        "avg_ticket": {"is_ratio": True, "base_value": 20, "variation": 5, "denominator_base": 500},
    }
    
    # Create store personalities (some stores consistently perform better/worse)
    store_personalities = {i: random.uniform(0.7, 1.3) for i in range(1, num_stores + 1)}
    
    # Create weekly seasonality factors
    weekly_factors = [random.uniform(0.9, 1.1) for _ in range(52)]
    
    # Generate records
    records = []
    
    for store_number in range(1, num_stores + 1):
        # Add some store personality
        personality = store_personalities[store_number]
        
        for week_offset in range(num_weeks):
            # Calculate current week
            current_date = start_date + timedelta(weeks=week_offset)
            year_week = current_date.strftime('%Y%W')
            epoch_week = start_epoch_week + week_offset
            
            # Weekly factor for seasonality
            week_of_year = current_date.isocalendar()[1]
            weekly_factor = weekly_factors[week_of_year - 1]
            
            for metric in metrics:
                if metric not in metric_properties:
                    continue
                    
                props = metric_properties[metric]
                is_ratio = props["is_ratio"]
                
                # Calculate base value with store personality and weekly seasonality
                base_value = props["base_value"] * personality * weekly_factor
                
                # Add some random variation
                variation = random.uniform(-props["variation"], props["variation"])
                value = max(0, base_value + variation)
                
                if is_ratio:
                    # For ratios, we need to generate numerator and denominator
                    denominator = max(1, int(props["denominator_base"] * personality * weekly_factor * random.uniform(0.8, 1.2)))
                    numerator = value * denominator
                else:
                    # For non-ratios, numerator is the value and denominator is null
                    numerator = value
                    denominator = None
                
                # Sometimes inject outliers (about 2% of data points)
                if random.random() < 0.02:
                    if random.choice([True, False]):  # High outlier
                        numerator *= random.uniform(1.5, 3.0)
                    else:  # Low outlier
                        numerator *= random.uniform(0.1, 0.5)
                
                # Round appropriately
                if isinstance(numerator, float):
                    if metric == "sales":
                        numerator = round(numerator, 2)
                    else:
                        numerator = round(numerator, 2 if is_ratio else 0)
                
                # Create record
                record = {
                    "store_number": store_number,
                    "year_week": year_week,
                    "epoch_week": epoch_week,
                    "metric": metric,
                    "numerator": numerator,
                    "denominator": denominator,
                    "isratio": is_ratio
                }
                records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {len(records)} records for {num_stores} stores over {num_weeks} weeks.")
    print(f"Data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Generate fake data
    generate_fake_store_data(
        num_stores=15,
        start_week="202401",
        num_weeks=12,
        output_file="store_metrics.csv"
    )
