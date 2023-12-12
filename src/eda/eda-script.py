from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np

# Initialize COCO API
dataDir = "/Users/kanishksaxena/Downloads/coco"
dataType = 'train2017'  # Specify the dataset type (e.g., train2017, val2017, test2017)
annFile = f'{dataDir}/annotations/instances_{dataType}.json'
coco = COCO(annFile)

# Load category information
categories = coco.loadCats(coco.getCatIds())
category_names = [category['name'] for category in categories]

# Explore category information
category_ids = coco.getCatIds(catNms=category_names)
category_images_count = {category: len(coco.getImgIds(catIds=[cat_id])) for cat_id, category in zip(category_ids, category_names)}
category_instance_count = {category: sum([len(coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])) for img_id in coco.getImgIds(catIds=[cat_id])]) for cat_id, category in zip(category_ids, category_names)}

# Visualize category distribution - Images and Instances
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(category_names, list(category_images_count.values()), color='skyblue')
plt.xlabel('Number of Images')
plt.ylabel('Categories')
plt.title('Image Distribution by Category')
plt.xticks(np.arange(0, max(list(category_images_count.values())), step=5000))
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.barh(category_names, list(category_instance_count.values()), color='salmon')
plt.xlabel('Number of Instances')
plt.ylabel('Categories')
plt.title('Instance Distribution by Category')
plt.xticks(np.arange(0, max(list(category_instance_count.values())), step=2000))
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Distribution of object instances (bounding box area)
ann_ids = coco.getAnnIds()
annotations = coco.loadAnns(ann_ids)
areas = [ann['area'] for ann in annotations]
plt.figure(figsize=(8, 6))
plt.hist(areas, bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('Object Area')
plt.ylabel('Frequency')
plt.title('Distribution of Object Instances by Area')
plt.grid(True)
plt.show()

# Explore segmentation types (polygon, bounding box, etc.)
segmentation_types = [ann['segmentation']['size'] for ann in annotations]
segmentation_types_count = {size: segmentation_types.count(size) for size in set(segmentation_types)}
sizes = list(segmentation_types_count.keys())
count = list(segmentation_types_count.values())

plt.figure(figsize=(8, 6))
plt.bar(sizes, count, color='orange')
plt.xlabel('Segmentation Type')
plt.ylabel('Count')
plt.title('Distribution of Segmentation Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()