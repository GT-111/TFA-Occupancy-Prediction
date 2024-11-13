#!/bin/bash

# Check if the logs directory exists
if [ -d "./logs" ]; then
    # Remove all files in the logs directory
    rm -r ./logs/*
    echo "Logs directory has been reset."
else
    echo "Logs directory does not exist."
fi