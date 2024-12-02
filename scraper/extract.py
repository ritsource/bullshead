import os
import zipfile
import json

# Base URL for direct file downloads
base_url = "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDC/"

# Directory to save downloaded files
download_dir = "binance_data/downloads"
output_dir = "binance_data"

sources = json.load(open("sources.json"))
intervals = json.load(open("binance-intervals.json"))

# Create the directories if they don't exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def extract_files(file_name_prefix, download_dir, source_name):
    print(f"\nExtracting {source_name} files...")
    
    # Create source directory in output_dir
    extract_dir = os.path.join(output_dir, source_name)
    os.makedirs(extract_dir, exist_ok=True)
        
    for file in os.listdir(download_dir):
        matches = False
        if source_name == "klines":
            for interval in intervals:
                if file.startswith(file_name_prefix + interval) and file.endswith('.zip'):
                    matches = True
                    break
        else:
            if file.startswith(file_name_prefix) and file.endswith('.zip'):
                matches = True
                
        if matches:
            zip_path = os.path.join(download_dir, file)
            
            try:
                # Verify zip file integrity before extracting
                if not zipfile.is_zipfile(zip_path):
                    print(f"Failed to extract {file}: File is not a valid zip file")
                    # Remove corrupted file so it can be re-downloaded
                    os.remove(zip_path)
                    continue
                    
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Extracted {file}")
            except Exception as e:
                print(f"Failed to extract {file}: {str(e)}")
    
    print(f"Finished extracting {source_name} files")

for src in sources:
    print(f"\nProcessing {src['name']}")
    extract_files(src["file_name_prefix"], download_dir, src["name"])
