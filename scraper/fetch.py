import os
import requests
import json
from datetime import datetime, timedelta

# Directory to save downloaded files
download_dir = "binance_data"

config = json.load(open("config.json"))
sources = config["sources"]
intervals = config["intervals"]

start_date = datetime.strptime(config["start_date"], "%Y-%m-%d")
end_date = datetime.strptime(config["end_date"], "%Y-%m-%d")

# Create the directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

def build_file_url(base_url, symbol, interval, date):
    return f"{base_url}/{interval}/{symbol}-{interval}-{date}.zip"

def download_files(name, base_url, symbol, file_name_prefix, download_dir):
    current_date = start_date

    file_urls = []
    
    if name == "klines":
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            for interval in intervals:
                file_name = f"{symbol}-{interval}-{date_str}.zip"
                file_url = build_file_url(base_url, symbol, interval, date_str)
                file_urls.append(file_url)
            current_date += timedelta(days=1)
    else:
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            file_name = f"{file_name_prefix}{date_str}.zip"
            file_url = build_file_url(base_url, symbol, interval, date_str)
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
    print(f"\nProcessing {src['name']}")
    download_files(src["name"], src["base_url"], src["symbol"], src["file_name_prefix"], download_dir)