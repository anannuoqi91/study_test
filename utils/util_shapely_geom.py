from shapely import geometry as geom


def create_polygon(points: list):
    return geom.Polygon(points)


def union_area(polygon1: geom.polygon, polygon2: geom.polygon):
    intersection = polygon1.intersection(polygon2)
    if intersection:
        return polygon1.intersection(polygon2).area
    else:
        return 0.0
