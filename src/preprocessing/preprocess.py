import numpy as np
from PIL import Image


def resize_image(image, target_size):
    """
    Resize an image to the target size.

    :param image: The input image as a numpy array.
    :param target_size: A tuple (width, height) for the target size.
    :return: Resized image as a numpy array.
    """
    resized_image = Image.fromarray(image).resize(target_size)
    return np.array(resized_image)


def normalize_image(image):
    """
    Normalize the image pixel values to the range [0, 1].

    :param image: The input image as a numpy array.
    :return: Normalized image as a numpy array.
    """
    return image / 255.0


def resize_annotations(image, image_instances, old_size, new_size):
    """
    Resize bounding box annotations to match the new image size.

    :param image: The input image as a numpy array.
    :param image_instances: List of instance annotations (bounding boxes).
    :param old_size: Tuple (old_width, old_height) of the image.
    :param new_size: Tuple (new_width, new_height) of the image.
    :return: List of resized instance annotations.
    """
    old_width, old_height = old_size
    new_width, new_height = new_size

    # Scaling factors
    x_scale = new_width / old_width
    y_scale = new_height / old_height

    resized_instances = []
    for instance in image_instances:
        x, y, width, height = instance['bbox']
        resized_bbox = [x * x_scale, y * y_scale, width * x_scale, height * y_scale]
        resized_instances.append({**instance, 'bbox': resized_bbox})

    return resized_instances


def resize_keypoints(image_keypoints, old_size, new_size):
    """
    Resize keypoint annotations to match the new image size.

    :param image_keypoints: List of keypoint annotations.
    :param old_size: Tuple (old_width, old_height) of the image.
    :param new_size: Tuple (new_width, new_height) of the image.
    :return: List of resized keypoint annotations.
    """
    old_width, old_height = old_size
    new_width, new_height = new_size

    # Scaling factors
    x_scale = new_width / old_width
    y_scale = new_height / old_height

    resized_keypoints = []
    for keypoints in image_keypoints:
        # Keypoints are typically stored as a flat list [x1, y1, v1, x2, y2, v2, ...]
        # where v is the visibility flag.
        resized_kp = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i:i + 3]
            resized_x = x * x_scale
            resized_y = y * y_scale
            resized_kp.extend([resized_x, resized_y, v])
        resized_keypoints.append(resized_kp)

    return resized_keypoints


def preprocess_data(data, target_size=(224, 224)):
    """
    Apply preprocessing steps to the loaded data.

    :param data: List of tuples containing image data and annotations.
    :param target_size: A tuple (width, height) for the target size of images.
    :return: List of tuples with preprocessed images and annotations.
    """
    preprocessed_data = []
    for image_data, image_instances, image_keypoints, image_captions in data:
        # Resize and normalize the image
        resized_image = resize_image(image_data, target_size)
        normalized_image = normalize_image(resized_image)

        # Resize annotations
        old_size = (image_data.shape[1], image_data.shape[0])  # (width, height)
        resized_instances = resize_annotations(image_data, image_instances, old_size, target_size)
        resized_keypoints = resize_keypoints(image_keypoints, old_size, target_size)

        # TODO: Add annotation processing if necessary

        preprocessed_data.append((normalized_image, resized_instances, image_keypoints, image_captions))

    return preprocessed_data
