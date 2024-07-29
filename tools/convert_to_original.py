import json
import argparse
from shapely.geometry import Polygon, MultiPolygon
import shapely.geometry

def simplify_polygon(poly, tolerance=0.01):
    return poly.simplify(tolerance, preserve_topology=True)

def ensure_correct_orientation(poly):
    if isinstance(poly, Polygon):
        if not shapely.geometry.polygon.orient(poly, sign=-1.0).is_empty:
            return shapely.geometry.polygon.orient(poly, sign=-1.0)
    return poly

def remove_small_polygons(geom, min_area=400):
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([poly for poly in geom.geoms if poly.area >= min_area])
    elif isinstance(geom, Polygon):
        if geom.area >= min_area:
            return geom
    return None

def process_polygon(poly):
    poly = ensure_correct_orientation(poly)  # 确保方向正确
    # poly = simplify_polygon(poly)            # 简化多边形
    poly = remove_small_polygons(poly)       # 移除过小的多边形
    return poly

def get_vertices(geometry):
    vertices = []
    if isinstance(geometry, Polygon):
        # 处理单个多边形
        vertices.append([list(coord) for coord in geometry.exterior.coords[:-1]])
    elif isinstance(geometry, MultiPolygon):
        # 处理多多边形
        for poly in geometry.geoms:
            vertices.extend(get_vertices(poly))  # 递归调用以处理每个子多边形
    return vertices

# 只保留小数点后 6位
def convert_from_coco(coco_annotations, image_id_to_filename):
    # 初始化原始数据结构
    original_format = []
    grouped_by_image_id = {}

    # 按image_id组织数据
    for annotation in coco_annotations:
        image_id = annotation['image_id']
        if image_id not in grouped_by_image_id:
            grouped_by_image_id[image_id] = []
        grouped_by_image_id[image_id].append(annotation)

    # 恢复图像和组信息
    for image_id, annotations in grouped_by_image_id.items():
        str_image_id = str(image_id)  # 将image_id转换为字符串
        image_entry = {
            "image": image_id_to_filename[str_image_id],
            "groups": []
        }
        # 组织为groups
        current_group = []
        for anno in annotations:
            # 小数点保留后6位
            formatted_vertices = [[round(x, 6), round(y, 6)] for x, y in anno['polys']]
            # 每个 polys 单独作为一个 group
            # decoded_text = ctc_decode(anno['rec'])
            current_group = [{
                "vertices": formatted_vertices,
                "text": anno['rec']
            }]
            # 将每个 group 加入到对应的 image entry 中
            image_entry['groups'].append(current_group)
        
        # if current_group:
        #     image_entry['groups'].append(current_group)
        original_format.append(image_entry)

    return original_format


def fix(raw_data):
    ppp = 0
    data = []
    for item in raw_data:
        image_data = {"image": item["image"], "groups": []}
        for group in item["groups"]:
            fixed_group = []
            for word in group:
                poly = Polygon(word["vertices"])
                if not poly.is_valid:
                    ppp += 1
                    print(ppp)
                    poly = poly.buffer(0)
                poly = process_polygon(poly)
                if poly:
                    fixed_vertices_list = get_vertices(poly)
                    for fixed_vertices in fixed_vertices_list:
                        fixed_group.append({"vertices": fixed_vertices, "text": word["text"]})
            image_data["groups"].append(fixed_group)
        data.append(image_data)
    return data

def main():
    parser = argparse.ArgumentParser(description="Process input and output file paths for JSON data.")
    parser.add_argument('--input_json', type=str, default='text_results_val.json', help='Input JSON file path')
    parser.add_argument('--input_image_id_json', type=str, default='datasets/rumsey/rumsey_val_image_id.json', help='Input JSON file path')
    parser.add_argument('--output_json', type=str,default='output_results.json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    json_file_path = args.input_json
    input_image_id = args.input_image_id_json
    output_path = args.output_json

    with open(input_image_id, 'r', encoding='utf-8') as file:
        image_id = json.load(file)

    with open(json_file_path, 'r', encoding='utf-8') as file:
        pred_results = json.load(file)

    converted_data = convert_from_coco(pred_results, image_id)
    converted_pred_data = fix(converted_data)

    # 保存转换后的数据到文件
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(converted_pred_data, outfile, ensure_ascii=False)
    print("Conversion complete and data saved to" + output_path)


if __name__ == '__main__':
    main()