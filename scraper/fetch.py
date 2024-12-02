import os
import requests
import json
from datetime import datetime, timedelta

# Base URL for direct file downloads
base_url = "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDC/"

# Directory to save downloaded files
download_dir = "binance_data"

sources = json.load(open("sources.json"))
intervals = json.load(open("binance-intervals.json"))

# Create the directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

def download_files(name, base_url, file_name_prefix, download_dir):
    start_date = datetime(2024, 11, 15)
    end_date = datetime(2024, 11, 20)
    current_date = start_date

    file_urls = []
    
    if name == "klines":
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            for interval in intervals:
                file_name = f"BTCUSDC-{interval}-{date_str}.zip"
                file_url = f"{base_url}{interval}/{file_name}"
                file_urls.append(file_url)
            current_date += timedelta(days=1)
    else:
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            file_name = f"{file_name_prefix}{date_str}.zip"
            file_url = base_url + file_name
            file_urls.append(file_url)
            current_date += timedelta(days=1)
    
    print(f"Found {len(file_urls)} files to download.")
    
    # Create downloads subdirectory
    downloads_dir = os.path.join(download_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    
    for idx, file_url in enumerate(file_urls, start=1):
        file_name = file_url.split("/")[-1]
        file_path = os.path.join(downloads_dir, file_name)
        
        # Skip download if file already exists
        if os.path.exists(file_path):
            print(f"[{idx}/{len(file_urls)}] Skipping {file_name}, already exists.")
            continue
        
        print(f"[{idx}/{len(file_urls)}] Downloading {file_name}...")
        file_response = requests.get(file_url, stream=True)
        if file_response.status_code == 200:
            with open(file_path, "wb") as file:
                for chunk in file_response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"Downloaded {file_name}")
        else:
            print(f"Failed to download {file_name}")
    
    print("Download complete.")

for src in sources:
    print(f"\nProcessing {src['name']}")
    download_files(src["name"], src["base_url"], src["file_name_prefix"], download_dir)
