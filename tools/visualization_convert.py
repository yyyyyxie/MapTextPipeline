# Visualization of the converted images
import json
import cv2
import os
import numpy as np

def compute_bezier_curve(bezier_pts, num_points=100):
    """根据贝塞尔控制点计算贝塞尔曲线上的点。"""
    n = len(bezier_pts) - 1
    t = np.linspace(0, 1, num_points)
    curve_points = np.zeros((num_points, 2))
    for i in range(n + 1):
        binomial_coeff = np.math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve_points += np.outer(binomial_coeff, bezier_pts[i])
    return curve_points

# 加载数据集
with open('datasets/voc148/tc25synth_train_5461voc.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取图像和注释
images = {img['id']: img['file_name'] for img in data['images']}
annotations_by_image = {img_id: [] for img_id in images.keys()}
for ann in data['annotations']:
    if ann['image_id'] in annotations_by_image:
        annotations_by_image[ann['image_id']].append(ann)

# 创建一个输出目录
output_dir = 'visual_tw_synth'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历每张图像，绘制所有相关的贝塞尔曲线
for img_id, img_path in images.items():
    full_image_path = os.path.join('datasets', img_path)
    print(f"Loading image from: {full_image_path}")
    image = cv2.imread(full_image_path)
    
    if image is None:
        print(f"Image not found: {full_image_path}")
        continue

    # 对于每个图像，处理所有相关的注释
    for ann in annotations_by_image[img_id]:
        bezier_pts = ann.get('bezier_pts', [])
        if bezier_pts:
            bezier_pts = np.array(bezier_pts).reshape(-1, 2)
            # 分割控制点为上部和下部
            upper_curve_pts = bezier_pts[:4]
            lower_curve_pts = bezier_pts[4:]
            
            # 计算曲线
            upper_curve = compute_bezier_curve(upper_curve_pts)
            lower_curve = compute_bezier_curve(lower_curve_pts)
            
            # 绘制上部曲线
            for i in range(1, len(upper_curve)):
                pt1 = tuple(np.int32(upper_curve[i - 1]))
                pt2 = tuple(np.int32(upper_curve[i]))
                cv2.line(image, pt1, pt2, (255, 0, 0), thickness=2)
            
            # 绘制下部曲线
            for i in range(1, len(lower_curve)):
                pt1 = tuple(np.int32(lower_curve[i - 1]))
                pt2 = tuple(np.int32(lower_curve[i]))
                cv2.line(image, pt1, pt2, (0, 255, 0), thickness=2)
            
            # 连接上部曲线的结束点与下部曲线的起始点
            cv2.line(image, tuple(np.int32(upper_curve[-1])), tuple(np.int32(lower_curve[0])), (0, 0, 255), thickness=2)
# 连接下部曲线的结束点与上部曲线的起始点
            cv2.line(image, tuple(np.int32(lower_curve[-1])), tuple(np.int32(upper_curve[0])), (0, 0, 255), thickness=2)
    # 保存修改后的图片
    output_filepath = os.path.join(output_dir, os.path.basename(full_image_path))
    if cv2.imwrite(output_filepath, image):
        print(f"Processed and saved: {output_filepath}")
    else:
        print(f"Failed to save: {output_filepath}")
