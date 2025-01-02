#!/bin/bash

# pyenv
source ./dev/bin/activate

# Run the data pipeline scripts
python3 scraper/fetch.py
python3 scraper/extract.py 
python3 scraper/merge.py
