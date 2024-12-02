import os
import json
import pandas as pd

# Directory to save downloaded files
download_dir = "binance_data"
output_dir = "binance_data_merged"

# Load schemas
agg_trades_schema = json.load(open("binance-agg-trades-schema.json"))
trades_schema = json.load(open("binance-trades-schema.json")) 
klines_schema = json.load(open("binance-klines-schema.json"))

sources = json.load(open("sources.json"))
intervals = json.load(open("binance-intervals.json"))

# Create the directories if they don't exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def merge_files(download_dir, source_name):
    print(f"\nMerging {source_name} files...")
    
    source_dir = os.path.join(download_dir, source_name)
    os.makedirs(source_dir, exist_ok=True)
        
    # Get schema based on source type
    if source_name == "aggTrades":
        schema = [col["name"] for col in agg_trades_schema]
        merge_regular_files(source_dir, source_name, schema)
    elif source_name == "trades":
        schema = [col["name"] for col in trades_schema]
        merge_regular_files(source_dir, source_name, schema)
    elif source_name == "klines":
        schema = [col["name"] for col in klines_schema]
        merge_klines_files(source_dir, schema)
    else:
        print(f"Unknown source type: {source_name}")
        return

def merge_regular_files(source_dir, source_name, schema):
    all_data = []
    
    # Iterate through files in source directory
    for file in sorted(os.listdir(source_dir)):
        if file.endswith('.csv'):
            file_path = os.path.join(source_dir, file)
            try:
                # Read CSV without headers since we'll add them later
                df = pd.read_csv(file_path, header=None)
                # Verify column count matches schema
                if len(df.columns) != len(schema):
                    print(f"Warning: {file} has {len(df.columns)} columns but schema has {len(schema)} columns. Skipping file.")
                    continue
                all_data.append(df)
                print(f"Processed {file}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                
    if all_data:
        # Concatenate all dataframes
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Add column headers
        merged_df.columns = schema
        
        # Create source directory in output_dir if it doesn't exist
        source_output_dir = os.path.join(output_dir, source_name)
        os.makedirs(source_output_dir, exist_ok=True)
        
        # Save merged file
        output_file = os.path.join(source_output_dir, "merged.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"Saved merged file to {output_file}")
    else:
        print(f"No data found for {source_name}")

def merge_klines_files(source_dir, schema):
    # Group files by interval
    interval_files = {}
    for file in sorted(os.listdir(source_dir)):
        if file.endswith('.csv'):
            # Extract interval from filename (e.g. BTCUSDC-30m-2024-01-01.csv)
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
                df = pd.read_csv(file_path, header=None)
                if len(df.columns) != len(schema):
                    print(f"Warning: {file} has {len(df.columns)} columns but schema has {len(schema)} columns. Skipping file.")
                    continue
                all_data.append(df)
                print(f"Processed {file}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            merged_df.columns = schema
            
            # Create interval-specific directory
            interval_dir = os.path.join(output_dir, "klines", interval)
            os.makedirs(interval_dir, exist_ok=True)
            
            # Save merged file for this interval
            output_file = os.path.join(interval_dir, "merged.csv")
            merged_df.to_csv(output_file, index=False)
            print(f"Saved merged file to {output_file}")
        else:
            print(f"No data found for interval {interval}")

for src in sources:
    print(f"\nProcessing {src['name']}")
    merge_files(download_dir, src["name"])
