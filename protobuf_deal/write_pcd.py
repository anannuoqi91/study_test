def write_pcd_ietsfss(filename, points: list):
    if not points or len(points[0]) != 10:
        return
    header = """
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z intensity elongation timestamp sub_id flags scan_id scan_idx
    SIZE 4 4 4 4 4 8 4 4 4 4
    TYPE F F F F F I I I I I
    COUNT 1 1 1 1 1 1 1 1 1 1
    WIDTH {}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS {}
    DATA ascii
    """
    num_points = len(points)
    header = header.format(num_points, num_points)

    with open(filename, 'w') as f:
        f.write(header)
        for pt in points:
            line = f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {pt[3]:.6f} {pt[4]:.6f} {int(pt[5])} {int(pt[6])} {int(pt[7])} {int(pt[8])} {int(pt[9])}'
            f.write(line + '\n')
