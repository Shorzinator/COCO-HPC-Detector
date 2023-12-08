#!/bin/bash

# Define the URL for the COCO train dataset
dataset_url="http://images.cocodataset.org/zips/test2014.zip"

# Function to download and unzip data
download_and_unzip() {
    local url=$1
    # shellcheck disable=SC2155
    local zip_file=$(basename "$url")

    # Download
    echo "Downloading data from $url"
    wget "$url"

    # Unzip
    echo "Unzipping the dataset"
    unzip "$zip_file"

    # Removing zip file
    rm "$zip_file"  # Remove the zip file after extraction
    echo "Download and extraction complete."
}

# Download and unzip the dataset in the current directory
download_and_unzip $dataset_url

echo "Dataset setup complete."
