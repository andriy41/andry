#!/bin/bash

echo "Starting NFL prediction continuous testing..."
echo "Will run for 1 hour with confidence validation"
echo "Results will be logged to continuous_test.log"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the continuous testing script
python3 continuous_test.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Testing completed successfully"
    echo "Check continuous_test.log for detailed results"
else
    echo "Testing terminated with errors"
    echo "Check continuous_test.log for error details"
fi
