import numpy as np
import cv2
import json
import pickle

def read_Chinese(file):
    ans = []
    with open(file, "rb") as fp:
        words = pickle.load(fp)
    for code in words:
        try:
            # 转换为 Unicode 字符
            char = chr(int(code))  # 注意 code 必须是整数
            ans.append(char)
        except (TypeError, ValueError) as e:
            print(f"无效的编码值: {code} ({str(e)})")
    return ans


CTLABELS = read_Chinese("./chn_cls_list")  # 替换为你的 txt 文件路径



def word_to_index_list(word, max_length=25):
 # 创建索引字典
    index_dict = {char: idx for idx, char in enumerate(CTLABELS)}

    # 转换单词到索引列表
    indices = [index_dict.get(char, len(CTLABELS) + 1) for char in word]

    # 如果单词长度小于 max_length，使用 len(CTlabels) + 1 填充剩余部分
    if len(indices) < max_length:
        indices.extend([len(CTLABELS) + 1] * (max_length - len(indices)))

    return indices[:max_length]

def index_list_to_word(index_list):
    # 填充值
    padding_value = len(CTLABELS) + 1

    # 转换索引回字符，忽略填充值
    characters = [CTLABELS[i] for i in index_list if i != padding_value]

    # 将字符列表转换成字符串
    return ''.join(characters)

def compute_bezier_curve(control_points, num_points=8):
    """根据控制点计算贝塞尔曲线上的点。"""
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    curve_points = np.zeros((num_points, 2))

    for i in range(n + 1):
        binomial_coeff = np.math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve_points += np.outer(binomial_coeff, control_points[i])

    return curve_points

def compute_bezier_points(control_points, num_points=8):
    """计算给定控制点的贝塞尔曲线上的点，假设控制点数组可以分为两部分。"""
    mid_index = len(control_points) // 2

    # 计算每一部分的贝塞尔曲线点
    points1 = compute_bezier_curve(control_points[:mid_index], num_points)
    points2 = compute_bezier_curve(control_points[mid_index:], num_points)

    # 将两部分的曲线点连接起来
    return np.concatenate((points1, points2), axis=0)

