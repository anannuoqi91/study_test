import math

l = 5.0
w = 2.0

k = 1


def cal_p(k, b, l, p1_x, p1_y):
    # 计算直线的单位方向向量
    norm = math.sqrt(1 + k**2)  # 直线方向向量的模长
    unit_vector = (1 / norm, k / norm)  # 单位方向向量 (dx, dy)

    # 计算两个新点
    point1 = (p1_x + l * unit_vector[0], p1_y + l * unit_vector[1])
    point2 = (p1_x - l * unit_vector[0], p1_y - l * unit_vector[1])

    return point1, point2


def calculate_intersection(k1, b1, k2, b2):
    # 检查平行线

    if k1 == k2:
        if b1 == b2:
            return "The lines are coincident (infinite intersection points)."
        else:
            return "The lines are parallel (no intersection point)."

    # 计算交点
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1  # 或者使用 y = k2 * x + b2

    return (x, y)


p1, p2 = cal_p(1, 0, 2.5, 0, 0)

k1 = -1
k2 = -1
k3 = 1
k4 = 1
b1 = p1[0] + p1[1]
b2 = p2[0] + p2[1]
b3 = math.sqrt(2)
b4 = -math.sqrt(2)
print(calculate_intersection(k1, b1, k3, b3))
print(calculate_intersection(k1, b1, k4, b4))
print(calculate_intersection(k2, b2, k3, b3))
print(calculate_intersection(k2, b2, k4, b4))
