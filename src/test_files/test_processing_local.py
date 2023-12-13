import os

import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

# Import the preprocess functions
from src.preprocessing.preprocess import generate_masks
from src.utility.path_utils import get_path_from_root

# Constants
IMAGE_DIR = get_path_from_root('data', 'coco', 'images', 'train')
MASK_DIR = get_path_from_root('data', 'mask', 'train')  # Ensure this matches with preprocess.py
ANNOTATION_FILE = get_path_from_root('data', 'coco', 'annotations', 'train&val', 'instances_train2014.json')
TARGET_SIZE = (128, 128)

# Initialize COCO API
coco = COCO(ANNOTATION_FILE)

# List of specific image files to process
specific_images = ['COCO_val2014_000000037149.jpg',
                   'COCO_val2014_000000186147.jpg',
                   'COCO_val2014_000000324670.jpg',
                   'COCO_val2014_000000464263.jpg']


def test_preprocessing():
    for image_file in specific_images:
        image_path = os.path.join(IMAGE_DIR, image_file)
        image = Image.open(image_path)
        processed_image = image.resize(TARGET_SIZE)

        # Extracting image ID from file name
        img_id = int(image_file.split('_')[-1].split('.')[0])

        # Generate masks for the specific image
        generate_masks(coco, [img_id], MASK_DIR, IMAGE_DIR, ['person'])

        # Load the generated mask
        mask_file = os.path.join(MASK_DIR, f"COCO_train2014_{img_id:012d}.jpg")
        if os.path.exists(mask_file):
            mask = Image.open(mask_file).resize(TARGET_SIZE)
        else:
            print(f"Mask not found for {image_file}")
            continue

        # Display processed image and mask
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(processed_image)
        plt.title(f"Processed Image: {image_file}")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Generated Mask: {image_file}")

        plt.show()


if __name__ == "__main__":
    test_preprocessing()
