#!/usr/bin/env python3
"""
Convert VOC format annotations to COCO format JSON
"""
import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

# VOC classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def convert_voc_to_coco(voc_root, split='train'):
    """
    Convert VOC annotations to COCO format

    Args:
        voc_root: Path to VOC root directory (e.g., VOCdevkit/VOC2007)
        split: 'train', 'val', 'trainval', or 'test'
    """
    # Create category mapping
    categories = []
    for idx, class_name in enumerate(VOC_CLASSES, start=1):
        categories.append({
            'id': idx,
            'name': class_name,
            'supercategory': 'none'
        })

    # Read image list for this split
    split_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{split}.txt')
    with open(split_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]

    print(f"Processing {len(image_ids)} images for {split} split...")

    # Initialize COCO format data
    coco_data = {
        'info': {
            'description': 'PASCAL VOC 2007 in COCO format',
            'version': '1.0',
            'year': 2007
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': categories
    }

    annotation_id = 1

    # Process each image
    for img_id, image_id in enumerate(image_ids, start=1):
        # Parse XML annotation
        ann_file = os.path.join(voc_root, 'Annotations', f'{image_id}.xml')
        tree = ET.parse(ann_file)
        root = tree.getroot()

        # Get image info
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        filename = root.find('filename').text

        # Add image info
        coco_data['images'].append({
            'id': img_id,
            'file_name': filename,
            'width': width,
            'height': height
        })

        # Process objects
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in VOC_CLASSES:
                continue

            category_id = VOC_CLASSES.index(class_name) + 1

            # Get bounding box
            bbox_elem = obj.find('bndbox')
            xmin = float(bbox_elem.find('xmin').text)
            ymin = float(bbox_elem.find('ymin').text)
            xmax = float(bbox_elem.find('xmax').text)
            ymax = float(bbox_elem.find('ymax').text)

            # Convert to COCO format (x, y, width, height)
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            area = bbox[2] * bbox[3]

            # Add annotation
            coco_data['annotations'].append({
                'id': annotation_id,
                'image_id': img_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0
            })

            annotation_id += 1

    print(f"Converted {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")

    return coco_data


if __name__ == '__main__':
    voc_root = 'VOCdevkit/VOC2007'
    output_dir = 'data'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert trainval and test splits
    for split in ['trainval', 'test']:
        print(f"\nConverting {split} split...")
        coco_data = convert_voc_to_coco(voc_root, split)

        # Save to JSON
        output_file = os.path.join(output_dir, f'{split}.json')
        with open(output_file, 'w') as f:
            json.dump(coco_data, f)

        print(f"Saved to {output_file}")

    print("\nConversion complete!")
