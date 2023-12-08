#!/bin/bash

# Define the URl from where the validation dat is being downloaded
dataset_url="http://images.cocodataset.org/zips/val2014.zip"

# Function to download data
download_data() {
	local url=$1
	
	# Download
	echo "Downloading data from $url"
	wget $url
	echo "val Download complete."
}

# Download the dataset zip file
download_data $dataset_url

echo "Dataset zip file downloaded."
