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


def calculate_bounding_box(vertices):
    # 将顶点列表转换为 NumPy 数组以便处理

    # 检查是否需要转换vertices格式
    if len(vertices) == 0 or not isinstance(vertices[0], (list, tuple)) or len(vertices[0]) != 2:
        vertices = reshape_vertices(vertices)  # 使用前面定义的reshape_vertices函数

    vertices_np = np.array(vertices)
    
    # 计算最小和最大 x、y 坐标
    x_min = np.min(vertices_np[:, 0])
    y_min = np.min(vertices_np[:, 1])
    x_max = np.max(vertices_np[:, 0])
    y_max = np.max(vertices_np[:, 1])
    
    # 计算宽度和高度
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]


def interpolate_to_fixed_points(vertices, l=16):
    vertices = np.array(vertices)
    num_vertices = len(vertices)
    if num_vertices == 4 or num_vertices == 8:
        l = 8

    # 计算所有边的长度
    lengths = np.sqrt(np.sum(np.square(vertices[:-1] - vertices[1:]), axis=1))
    total_length = np.sum(lengths)

    # 根据总长度确定每个插入点的目标间隔
    interval = total_length / (l - 1)  # 减1是为了让间隔计算略微向内收缩，避免最后一个点过早出现

    # 创建插值点数组，初始化为第一个顶点
    new_vertices = [vertices[0].tolist()]

    # 累计长度和当前边的索引
    accum_length = 0
    vertex_index = 0

    # 循环生成除第一个顶点外的l-1个顶点
    while len(new_vertices) < l - 1:
        current_edge_length = lengths[vertex_index]
        start = vertices[vertex_index]
        end = vertices[(vertex_index + 1) % num_vertices]

        # 累积距离足够插入下一个点
        while accum_length + current_edge_length < interval:
            accum_length += current_edge_length
            vertex_index = (vertex_index + 1) % num_vertices
            current_edge_length = lengths[vertex_index]
            start = vertices[vertex_index]
            end = vertices[(vertex_index + 1) % num_vertices]

        # 插值生成新的顶点
        remaining = interval - accum_length
        t = remaining / current_edge_length
        interpolated_point = (1 - t) * start + t * end
        new_vertices.append(interpolated_point.tolist())
        
        # 更新下一个插值点的目标距离
        interval += total_length / (l - 1)

    # 确保最后一个点是原始顶点数组的最后一个点
    new_vertices.append(vertices[-1].tolist())

    return new_vertices