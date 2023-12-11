#!/bin/bash

# Define the name of the zip file
zip_file="train2014.zip"

# Function to unzip data
unzip_data() {
    local file=$1

    # Check if the zip file exists
    if [ -f "$file" ]; then
        echo "Unzipping the file $file"
        unzip "$file"
        echo "Train Unzipping complete."
    else
        echo "Error: File $file not found."
    fi
}

# Unzip the dataset zip file
unzip_data $zip_file

