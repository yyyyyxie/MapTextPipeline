import argparse
import json
import os

import cv2
import numpy as np

from tools.utils.bezier_utils import polygon_to_bezier_pts
from tools.utils.geometry import *
from tools.utils.utils import index_list_to_word, word_to_index_list


def convert_to_bezier_points(vertices, img):
    # 确保顶点数量为偶数
    vertices = ensure_even_vertices(vertices)
    bezier_pts = polygon_to_bezier_pts(np.array(vertices), img)
    return np.round(bezier_pts, 6).flatten().tolist()

# 转换为COCO格式的函数
def convert_to_coco(original_annotations):
    # 初始化COCO数据结构
    coco_format = {
        "licenses": [],
        "info": {},
        "categories": [],
        "images": [],
        "annotations": []
    }

    # 假设我们已经填充了licenses和categories信息
    coco_format['licenses'].append({
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    })

    coco_format['categories'].append({
        "supercategory": "none",
        "id": 1,
        "name": "text"
    })

    image_id = 0
    annotation_id = 0
    image_id_to_filename = {}  # 新增映射字典

    for item in original_annotations:
        img_path = 'datasets/' + item['image']
        img = cv2.imread(img_path)  # 读取图像

        # 保存image_id到文件名的映射
        image_id_to_filename[image_id] = item['image']

        # 添加图像信息
        coco_format['images'].append({
            "license": 0,
            "file_name": item['image'],
            "coco_url": "",
            "height": img.shape[0], 
            "width": img.shape[1], 
            "date_captured": "",
            "id": image_id
        })

        # 对于每个文本实体，添加注释
        for group in item['groups']:
            for word in group:
                # bezier_pts = np.round(polygon_to_bezier_pts(np.array(word['vertices']), img), 2).flatten().tolist()
                polys = interpolate_to_fixed_points(word['vertices']) 
                text = word_to_index_list(word['text'])
                bezier_pts = convert_to_bezier_points(polys, img)
                # bezier_pts = np.round(polygon_to_bezier_pts(np.array(polys).reshape(-1, 2), img), 2).flatten().tolist()
                area = calculate_polygon_area(polys)
                bbox = calculate_bounding_box(polys)
                coco_format['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    # "segmentation": [],
                    "area": area,  # 根据实际情况计算多边形面积
                    "bbox": bbox,  # 根据实际情况计算包围盒[x_min, y_min, width, height]
                    "iscrowd": 0,
                    "polys":polys,
                    "bezier_pts":bezier_pts,
                    "rec": text,
                    "illegible": word['illegible'],
                    "truncated": word['truncated']
                })
                if len(word['vertices']) > 40:
                    print(item['image'])
                annotation_id += 1

        image_id += 1

    return coco_format, image_id_to_filename
 


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Process input and output file paths for JSON data.")
    parser.add_argument('--input_json', type=str, default='datasets/rumsey/rumsey_val.json', help='Input JSON file path')
    parser.add_argument('--output_json', type=str,default='datasets/rumsey/rumsey_val_148voc.json', help='Output JSON file path')
    parser.add_argument('--output_image_id_json', type=str,default='datasets/rumsey/rumsey_val_image_id.json', help='Output JSON file path')

    
    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数中的路径
    json_file_path = args.input_json
    converted_json_path = args.output_json
    output_image_id = args.output_image_id_json

    # 从json文件中加载数据
    with open(json_file_path, 'r', encoding='utf-8') as file:
        original_annotations = json.load(file)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(converted_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_image_id), exist_ok=True)


    # 剩余的功能函数和代码
    converted_coco, image_id_to_file = convert_to_coco(original_annotations)

    # 将转换后的JSON数据写入文件
    with open(converted_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(converted_coco, json_file, ensure_ascii=False)

    # 将转换后的JSON数据写入文件
    with open(output_image_id, 'w', encoding='utf-8') as json_file:
        json.dump(image_id_to_file, json_file, ensure_ascii=False)

    print("Conversion complete.")

if __name__ == '__main__':
    main()
