import os

import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

from src.preprocessing.preprocess import generate_masks  # Ensure this function is correctly imported
# Import your preprocessing and utility scripts
from src.utility.path_utils import get_path_from_root

# Constants
IMAGE_DIR = get_path_from_root('data', 'coco', 'images', 'train')
MASK_DIR = get_path_from_root('data', 'mask', 'train')  # Ensure this path is correct
ANNOTATION_FILE = get_path_from_root('data', 'coco', 'annotations', 'train&val', 'instances_train2014.json')
SAMPLE_SIZE = 5  # Number of images to process and display
TARGET_SIZE = (128, 128)  # Update this if different

# Initialize COCO API
coco = COCO(ANNOTATION_FILE)


def test_preprocessing():
    # Filter images based on categories (e.g., 'person')
    catIds = coco.getCatIds(catNms=['food'])  # Change 'food' to the category of interest if needed
    imgIds = coco.getImgIds(catIds=catIds)
    selected_imgIds = imgIds[:SAMPLE_SIZE]

    for img_id in selected_imgIds:
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(IMAGE_DIR, img_info['file_name'])
        image = Image.open(image_path)
        processed_image = image.resize(TARGET_SIZE)

        # Generate masks for the specific image
        generate_masks(coco, [img_id], MASK_DIR, IMAGE_DIR, ['food'])  # Match the category with catIds

        # Load the generated mask
        mask_file = os.path.join(MASK_DIR, f"COCO_train2014_{img_id:012d}.jpg")
        if os.path.exists(mask_file):
            mask = Image.open(mask_file).resize(TARGET_SIZE)
        else:
            print(f"Mask not found for {img_info['file_name']}")
            continue

        # Display processed image and mask
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(processed_image)
        plt.title(f"Processed Image: {img_info['file_name']}")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Generated Mask: {img_info['file_name']}")
        # plt.savefig(f"Generated Mask: {img_info['file_name']}.png")
        plt.show()


if __name__ == "__main__":
    test_preprocessing()
