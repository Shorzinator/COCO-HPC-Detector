#!/bin/bash

# Define the base directory containing the 'data' folder
base_directory_path="/u/s/m/smaheshwari7/Group4/COCO-HPC-Detector"

pwd

# Change to the base directory
# shellcheck disable=SC2164
cd "$base_directory_path"

# Define the URLs for the COCO dataset
train_dataset_url="http://images.cocodataset.org/zips/train2014.zip"
val_dataset_url="http://images.cocodataset.org/zips/val2014.zip"
test_dataset_url="http://images.cocodataset.org/zips/test2014.zip"

# Main directory for the dataset
main_dir="data/coco/images"

# Create main dataset directory
mkdir -p $main_dir

# Define the number of subdirectories
num_parts=6

# Function to download and unzip data
download_and_unzip() {
    local url=$1
    local output_dir=$2
    local zip_file=$(basename $url)

    # Download
    echo "Downloading data from $url"
    wget $url -P $output_dir

    # Unzip
    echo "Unzipping the dataset in $output_dir"
    unzip "${output_dir}/${zip_file}" -d $output_dir
    rm "${output_dir}/${zip_file}"  # Remove the zip file after extraction
    echo "Download and extraction complete for $output_dir."
}

# Function to create subdirectories and distribute data
distribute_data() {
    local dataset_dir=$1
    local num_files
    local i=1

    echo "Creating subdirectories and distributing data in $dataset_dir"
    mkdir -p "${dataset_dir}/part"{1..$num_parts}

    # Get a list of files
    files=($(ls $dataset_dir))

    num_files=${#files[@]}
    files_per_part=$((num_files / num_parts))

    for file in "${files[@]}"; do
        if [ $(( (i - 1) % files_per_part )) -eq 0 ] && [ $((i / files_per_part)) -le $num_parts ]; then
            part=$((i / files_per_part))
            part=$((part==0?1:part))
        fi
        mv "${dataset_dir}/${file}" "${dataset_dir}/part${part}/"
        let i++
    done
}

# Start downloading and unzipping datasets in the background
echo "Starting to download and unzip datasets"
download_and_unzip $train_dataset_url "${main_dir}/train" &
download_and_unzip $val_dataset_url "${main_dir}/val" &
download_and_unzip $test_dataset_url "${main_dir}/test" &

# Wait for all background processes (downloads and unzipping) to finish
wait
echo "All datasets downloaded and unzipped"

# Distribute data in subdirectories for train, val, and test
distribute_data "${main_dir}/train"
distribute_data "${main_dir}/val"
distribute_data "${main_dir}/test"

echo "Dataset setup complete."
