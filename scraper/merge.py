import os
import json
import pandas as pd
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.json', help='Path to config file')
args = parser.parse_args()

config = json.load(open(args.config))
sources = config["sources"]
intervals = config["intervals"]

def merge_files(export_dir, type, symbol, start_date, end_date):
    print(f"\nMerging {type} files...")
    
    # Get extracted directory
    source_dir = os.path.join(export_dir, "extracted")
    
    # Create output directory
    output_dir = os.path.join(export_dir, "merged")
    os.makedirs(output_dir, exist_ok=True)

    # Get column headers from config schema
    headers = [field["name"] for field in config["klines-schema"]]

    # Group files by interval
    interval_files = {}
    for file in sorted(os.listdir(source_dir)):
        if file.endswith('.csv'):
            # Extract interval from filename (e.g. BTCUSDT-1d-2024-01.csv)
            interval = file.split('-')[1]
            if interval not in interval_files:
                interval_files[interval] = []
            interval_files[interval].append(file)

    # Process each interval separately
    for interval, files in interval_files.items():
        all_data = []
        for file in files:
            file_path = os.path.join(source_dir, file)
            try:
                df = pd.read_csv(file_path, header=None, names=headers)
                all_data.append(df)
                print(f"Processed {file}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            
            # Create interval-specific directory
            interval_dir = os.path.join(output_dir, interval)
            os.makedirs(interval_dir, exist_ok=True)
            
            # Save merged file for this interval
            output_file = os.path.join(interval_dir, f"{symbol}-{interval}-{start_date}-to-{end_date}.csv")
            merged_df.to_csv(output_file, index=False)
            print(f"Saved merged file to {output_file}")
        else:
            print(f"No data found for interval {interval}")

for src in sources:
    print(f"\nProcessing {src['base_url']}")
    merge_files(src["export_dir"], src["type"], src["symbol"], config["start_date"], config["end_date"])