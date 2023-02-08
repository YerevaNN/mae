import os
from tqdm import tqdm
import json
from PIL import Image
import pandas as pd


nightowls_dir = '/home/ani/nightowls_stage_3/'

def coco_dict():
    coco_format = {}
    coco_format['images'] = []
    coco_format['annotations'] = []
    coco_format['categories'] = [
        {
            'id': 1, 
            'name': 'pedestrian',
            'supercategory': 'pedestrian'
        },
        {
            'id': 2, 
            'name': 'motorbike driver',
            'supercategory': 'motorbike driver'
        },
        {
            'id': 3, 
            'name': 'motorbike driver',
            'supercategory': 'motorbike driver'
        }
    ]
    
    return coco_format


def nightowls_annotations(label_filename):
    annotations = pd.read_csv(label_filename, header=None, sep=' ')
    img_boxes = annotations.iloc[:, 1:].values  # [x1, y1, x2, y2]
    img_labels = annotations.iloc[:, 0].values  # label
    return img_boxes, img_labels


def create_coco(path, img_dir, label_dir):
    image_names = os.listdir(os.path.join(path, img_dir)) # returns list of img names without absolute path
    image_names = [x for x in image_names if '.png' in x]
    max_img_size, min_img_size = -1, 100000000
    nightowls_coco_format = coco_dict()
    
    for img_id, img_name in tqdm(enumerate(image_names), total=len(image_names)):
        img_filename = os.path.join(path, img_dir, img_name)
#         print(img_filename)
#         if '58c58167bc260130acfebf96' in img_filename:
        label_filename = os.path.join(path, label_dir, img_name.replace('.png', '.txt'))
        img = Image.open(img_filename)

        width, height = img.size
        max_img_size = max(max_img_size, height, width)
        min_img_size = min(min_img_size, height, width)
        tmp_img_dct = {
            'file_name': img_filename, 
            'height': height,
            'width': width,
            'id': img_id
        }

        nightowls_coco_format['images'].append(tmp_img_dct)

        img_boxes, img_labels = nightowls_annotations(label_filename)

        bbox_id = 0
        for boxes, label in zip(img_boxes, img_labels):
            bbox_width, bbox_height = boxes[2] - boxes[0], boxes[3] - boxes[1]
#             print(bbox_width, bbox_height)
            tmp_annotation_dct = {
                'image_id': img_id,
                'category_id': int(label), 
                'bbox': [int(boxes[0]), int(boxes[1]), int(bbox_width), int(bbox_height)],
                'id': bbox_id,
                'iscrowd': 0,
                'area': int(bbox_width * bbox_height)
            }
            nightowls_coco_format['annotations'].append(tmp_annotation_dct)
            bbox_id += 1

    return nightowls_coco_format, max_img_size, min_img_size


if __name__ == "__main__":
    nightowls_train, max_, min_ = create_coco(nightowls_dir, './', './')

    with open('./annotations/few_shot_8_nightowls.json', 'w') as no:
        json.dump(nightowls_train, no)