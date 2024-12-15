#!/bin/bash

# Change to the project directory
cd "$(dirname "$0")"

# Activate Python environment if needed
# source /path/to/your/venv/bin/activate  # Uncomment and modify if using virtualenv

# Run the update script
python3 data_collection/automated_updates.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "NFL data update completed successfully"
else
    echo "Error updating NFL data"
    exit 1
fi
