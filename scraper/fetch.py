import os
import requests
import json
from datetime import datetime, timedelta
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.json', help='Path to config file')
args = parser.parse_args()

config = json.load(open(args.config))

sources = config["sources"]
intervals = config["intervals"]

start_date = datetime.strptime(config["start_date"], "%Y-%m-%d")
end_date = datetime.strptime(config["end_date"], "%Y-%m-%d")

def build_file_url(base_url, aggrigate, symbol, interval, formatted_date):
    if aggrigate == "monthly":
        # For monthly files, only use YYYY-MM format
        formatted_date = formatted_date[:7]
    return f"{base_url}/{interval}/{symbol}-{interval}-{formatted_date}.zip"

def download_files(export_dir, type, aggrigate, symbol, base_url):
    # Create the directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    current_date = start_date

    file_urls = []
    
    if type == "klines":
        while current_date <= end_date:
            if aggrigate == "monthly":
                date_str = current_date.strftime("%Y-%m")
                # Skip if we've already processed this month
                if len(file_urls) > 0 and date_str in file_urls[-1]:
                    current_date += timedelta(days=1)
                    continue
            else:
                date_str = current_date.strftime("%Y-%m-%d")
                
            for interval in intervals:
                file_name = f"{symbol}-{interval}-{date_str}.zip"
                file_url = build_file_url(base_url, aggrigate, symbol, interval, date_str)
                file_urls.append(file_url)
            current_date += timedelta(days=1)
    else:
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            file_name = f"{file_name_prefix}{date_str}.zip"
            file_url = build_file_url(base_url, aggrigate, symbol, interval, date_str)
            file_urls.append(file_url)
            current_date += timedelta(days=1)
    
    print(f"Found {len(file_urls)} files to download.")
    
    # Create downloads subdirectory
    downloads_dir = os.path.join(export_dir, "downloads", "")
    os.makedirs(downloads_dir, exist_ok=True)
    
    for idx, file_url in enumerate(file_urls, start=1):
        file_name = file_url.split("/")[-1]
        file_path = os.path.join(downloads_dir, file_name)
        
        # Skip download if file already exists
        if os.path.exists(file_path):
            print(f"[{idx}/{len(file_urls)}] Skipping {file_name}, already exists.")
            continue
        
        print(f"[{idx}/{len(file_urls)}] Downloading {file_url}")
        try:
            file_response = requests.get(file_url, stream=True)
            if file_response.status_code == 200:
                with open(file_path, "wb") as file:
                    for chunk in file_response.iter_content(chunk_size=1024):
                        file.write(chunk)
                print(f"Downloaded {file_name}")
            else:
                print(f"Failed to download {file_name} (Status code: {file_response.status_code})")
                if file_response.headers.get('content-type') == 'application/xml':
                    print("File does not exist on server")
                else:
                    print(f"Error: {file_response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Network error while downloading {file_name}: {str(e)}")
    
    print("Download complete.")

for src in sources:
    print(f"\nProcessing {src['base_url']}")
    download_files(src["export_dir"], src["type"], src["aggrigate"], src["symbol"], src["base_url"])

# https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2023-08-09.zip
# https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-11.zip