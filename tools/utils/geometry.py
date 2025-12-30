import numpy as np


def ensure_even_vertices(vertices):
    # 检查顶点数量是否为偶数，如果不是，则复制最后一个顶点
    if len(vertices) % 2 != 0:
        vertices.append(vertices[-1])
    return vertices


def calculate_polygon_area(vertices):
    """
    计算多边形的面积，使用顶点数组。
    检查vertices的结构：
    - 如果vertices已经是[n, 2]格式，直接使用。
    - 如果不是，假设vertices是一维列表，转换为[n, 2]格式。
    """
    # 检查是否需要转换vertices格式
    if len(vertices) == 0 or not isinstance(vertices[0], (list, tuple)) or len(vertices[0]) != 2:
        vertices = reshape_vertices(vertices)  # 使用前面定义的reshape_vertices函数

    n = len(vertices)
    if n < 3:
        raise ValueError("A polygon must have at least three vertices.")
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2.0
    return area

def reshape_vertices(vertices):
    if len(vertices) % 2 != 0:
        raise ValueError("The total number of elements in vertices must be even (x and y pairs).")
    reshaped_vertices = []
    for i in range(0, len(vertices), 2):
        reshaped_vertices.append([vertices[i], vertices[i+1]])
    return reshaped_vertices


def calculate_bounding_box(vertices, img_width=None, img_height=None):
    """
    计算多边形的边界框
    :param vertices: 顶点列表
    :param img_width: 图像宽度（可选，用于裁剪边界）
    :param img_height: 图像高度（可选，用于裁剪边界）
    :return: [x_min, y_min, width, height]
    """
    # 检查是否需要转换vertices格式
    if len(vertices) == 0 or not isinstance(vertices[0], (list, tuple)) or len(vertices[0]) != 2:
        vertices = reshape_vertices(vertices)

    vertices_np = np.array(vertices)

    # 计算最小和最大 x、y 坐标
    x_min = np.min(vertices_np[:, 0])
    y_min = np.min(vertices_np[:, 1])
    x_max = np.max(vertices_np[:, 0])
    y_max = np.max(vertices_np[:, 1])

    # 如果提供了图像尺寸，裁剪到图像边界内
    if img_width is not None:
        x_min = max(0, x_min)
        x_max = min(img_width, x_max)
    if img_height is not None:
        y_min = max(0, y_min)
        y_max = min(img_height, y_max)

    # 计算宽度和高度
    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height]


def _interpolate_edge(points, num_output):
    """
    沿一条边（由多个点组成）进行插值，生成固定数量的点
    :param points: 边上的点序列，shape (n, 2)
    :param num_output: 输出点的数量
    :return: 插值后的点列表
    """
    points = np.array(points)
    if len(points) < 2:
        return [points[0].tolist()] * num_output

    # 计算每段的长度
    lengths = np.sqrt(np.sum(np.square(points[:-1] - points[1:]), axis=1))
    total_length = np.sum(lengths)

    if total_length == 0:
        return [points[0].tolist()] * num_output

    # 累积长度
    cumulative_lengths = np.concatenate([[0], np.cumsum(lengths)])

    # 生成均匀分布的目标位置
    target_positions = np.linspace(0, total_length, num_output)

    result = []
    for target in target_positions:
        # 找到目标位置所在的边段
        idx = np.searchsorted(cumulative_lengths, target, side='right') - 1
        idx = max(0, min(idx, len(points) - 2))

        # 计算在该边段上的插值比例
        segment_start = cumulative_lengths[idx]
        segment_length = lengths[idx]

        if segment_length == 0:
            t = 0
        else:
            t = (target - segment_start) / segment_length

        t = max(0, min(1, t))

        # 线性插值
        interpolated = (1 - t) * points[idx] + t * points[idx + 1]
        result.append(interpolated.tolist())

    return result


def interpolate_to_fixed_points(vertices, l=16):
    """
    将多边形顶点插值为固定数量的点，分别对上边界和下边界进行插值

    假设输入顶点格式为：[上边界点..., 下边界点...]
    - 对于4点四边形：[左上, 右上, 右下, 左下]，前2点为上边界，后2点为下边界
    - 对于多点多边形：前半部分为上边界，后半部分为下边界

    :param vertices: 顶点列表
    :param l: 输出总点数（上下边界各 l/2 个点）
    :return: 插值后的点列表 [上边界点..., 下边界点...]
    """
    vertices = np.array(vertices)
    num_vertices = len(vertices)

    # 对于4点或8点，输出8个点
    if num_vertices == 4 or num_vertices == 8:
        l = 8

    half_l = l // 2  # 上下边界各输出的点数
    mid_idx = num_vertices // 2  # 上下边界的分界点

    # 分离上边界和下边界
    # 上边界：从左到右
    top_boundary = vertices[:mid_idx]
    # 下边界：从右到左（保持原顺序）
    bottom_boundary = vertices[mid_idx:]

    # 分别对上下边界进行插值
    top_interpolated = _interpolate_edge(top_boundary, half_l)
    bottom_interpolated = _interpolate_edge(bottom_boundary, half_l)

    # 合并结果
    return top_interpolated + bottom_interpolated