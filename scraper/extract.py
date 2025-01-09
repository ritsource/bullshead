import os
import zipfile
import json
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.json', help='Path to config file')
args = parser.parse_args()

config = json.load(open(args.config))
sources = config["sources"]
intervals = config["intervals"]

def extract_files(export_dir, type):
    print(f"\nExtracting {type} files...")
    
    # Get downloads directory
    downloads_dir = os.path.join(export_dir, "downloads")
    
    # Create extract directory
    extract_dir = os.path.join(export_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
        
    for file in os.listdir(downloads_dir):
        if file.endswith('.zip'):
            zip_path = os.path.join(downloads_dir, file)
            
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
    
    print(f"Finished extracting {type} files")

for src in sources:
    print(f"\nProcessing {src['base_url']}")
    extract_files(src["export_dir"], src["type"])
