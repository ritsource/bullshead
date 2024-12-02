#!/bin/bash

# pyenv
source ./dev/bin/activate

# Run the data pipeline scripts
python3 fetch.py
python3 extract.py 
python3 merge.py
