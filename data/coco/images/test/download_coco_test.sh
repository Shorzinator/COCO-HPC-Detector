#!/bin/bash

# Define the URL for the COCO dataset
dataset_url="http://images.cocodataset.org/zips/test2014.zip"

# Function to download data
download_data() {
    local url=$1

    # Download
    echo "Downloading data from $url"
    wget $url
    echo "Download complete."
}

# Download the dataset zip file
download_data $dataset_url

echo "Test Dataset zip file downloaded."

