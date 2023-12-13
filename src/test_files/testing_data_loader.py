from matplotlib import patches, pyplot as plt

from src.utility.data_loader import load_coco_data, load_coco_data_custom
from src.utility.path_utils import get_path_from_root


# Function to visualize an image with its bounding boxes
def visualize_image_with_boxes(image_data, image_instances):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_data)

    # Add bounding boxes
    for instance in image_instances:
        # Assuming bbox format is [x, y, width, height]
        bbox = instance['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def main():
    # Path to your sample images and annotation files
    image_dir = get_path_from_root("data", "extra_data")
    instances_ann_file = get_path_from_root("data", "coco", "annotations", "train&val", "instances_val2014.json")
    keypoints_ann_file = get_path_from_root("data", "coco", "annotations", "train&val", "person_keypoints_val2014.json")
    captions_ann_file = get_path_from_root("data", "coco", "annotations", "train&val", "captions_val2014.json")

    # Load the data
    # sample_data = load_coco_data_custom(image_dir, instances_ann_file, keypoints_ann_file, captions_ann_file)
    sample_data = load_coco_data(image_dir, instances_ann_file)

    print("sample data:", sample_data)
    # Print out some basic information to check if data is loaded correctly
    for image_data, annotations in sample_data:
        image_instances = [ann for ann in annotations if 'bbox' in ann]
        image_keypoints = [ann for ann in annotations if 'keypoints' in ann]
        image_captions = [ann for ann in annotations if 'caption' in ann]

        print("Image shape:", image_data.shape)
        print("Number of instances:", len(image_instances))
        print("Number of keypoints annotations:", len(image_keypoints))
        print("Number of captions:", len(image_captions))
        print("-----------------------------------------")

        # Optional: Visualize the first image in the sample data
        visualize_image_with_boxes(image_data, image_instances)


if __name__ == "__main__":
    main()
