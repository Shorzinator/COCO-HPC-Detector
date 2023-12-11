import json
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


def load_coco_data_custom(image_dir, instances_ann_file, keypoints_ann_file=None, captions_ann_file=None):
    """
    Load COCO dataset images and annotations (instances, keypoints, captions).

    :param image_dir: Directory containing COCO images.
    :param instances_ann_file: JSON file containing instance annotations.
    :param keypoints_ann_file: JSON file containing person keypoints annotations.
    :param captions_ann_file: JSON file containing captions annotations.
    :return: List of tuples, each containing an image and its annotations.
    """
    # Load instance annotations
    with open(instances_ann_file, 'r') as f:
        instances_annotations = json.load(f)

    keypoints_annotations = None
    if keypoints_ann_file:
        with open(keypoints_ann_file, 'r') as f:
            keypoints_annotations = json.load(f)

    captions_annotations = None
    if captions_ann_file:
        with open(captions_ann_file, 'r') as f:
            captions_annotations = json.load(f)

    # List of image file names in the image_dir
    image_files = set(os.listdir(image_dir))

    # Process each image
    data = []
    for ann in instances_annotations['images']:
        if ann['file_name'] in image_files:
            image_path = os.path.join(image_dir, ann['file_name'])

            try:
                image = Image.open(image_path)
                image_data = np.array(image)

                # Extract instance annotations for this image
                image_instances = [a for a in instances_annotations['annotations'] if a['image_id'] == ann['id']]

                # Extract keypoints annotations for this image (if available)
                image_keypoints = []
                if keypoints_annotations:
                    image_keypoints = [a for a in keypoints_annotations['annotations'] if a['image_id'] == ann['id']]

                # Extract captions for this image (if available)
                image_captions = []
                if captions_annotations:
                    image_captions = [a for a in captions_annotations['annotations'] if a['image_id'] == ann['id']]

                data.append((image_data, image_instances, image_keypoints, image_captions))

            except FileNotFoundError:
                continue

    return data


def load_coco_data(image_dir, annotation_file):
    """
    Load COCO dataset images and annotations using pycocotools.

    :param image_dir: Directory containing COCO images.
    :param annotation_file: JSON file containing COCO annotations.
    :return: List of tuples, each containing an image and its annotations.
    """
    # Initialize COCO api for instance annotations
    coco = COCO(annotation_file)

    # Get IDs of all images in the dataset
    img_ids = coco.getImgIds()

    # List of image file names in the image_dir
    available_files = set(os.listdir(image_dir))

    data = []
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        if img_info['file_name'] in available_files:
            img_path = os.path.join(image_dir, img_info['file_name'])
            try:
                image = Image.open(img_path)
                image_data = np.array(image)

                # Load annotations
                ann_ids = coco.getAnnIds(imgIds=img_id)
                annotations = coco.loadAnns(ann_ids)

                data.append((image_data, annotations))
            except FileNotFoundError:
                # Handle missing files if necessary
                continue

    return data
