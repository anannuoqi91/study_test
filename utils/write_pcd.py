def check_headers(headers):
    if not isinstance(headers, list):
        raise ValueError("Headers must be list of strings.")
    if not headers:
        raise ValueError("Headers cannot be empty.")
    if "x:F" not in headers or "y:F" not in headers or "z:F" not in headers:
        raise ValueError(
            "Headers must include 'x:F', 'y:F', 'z:F' .")


def generate_headers(headers: list):
    check_headers(headers)
    size_map = {
        'F': '4',
        'I64': '8',
        'I32': '4',
    }
    type_map = {
        'F': 'F',
        'I64': 'I',
        'I32': 'I',
    }
    fields = "FIELDS "
    size = "SIZE "
    type_ = "TYPE "
    count = "COUNT "
    type_list = []
    for header in headers:
        f_size = header.split(":")
        if len(f_size) != 2:
            raise ValueError(
                f"Invalid header: {header}, must be 'fieldname:type' .")
        fields += f"{f_size[0]} "
        if f_size[1] not in size_map:
            raise ValueError(
                f"Invalid type: {f_size[1]}. must in {size_map.keys().tolist()}")
        size += f"{size_map[f_size[1]]} "
        type_ += f"{type_map[f_size[1]]} "
        count += f"{1} "
        type_list.append(type_map[f_size[1]])
    fields = fields.rstrip()
    size = size.rstrip()
    type_ = type_.rstrip()
    count = count.rstrip()
    header_str = "# .PCD v0.7 - Point Cloud Data file format\n"
    header_str += "VERSION 0.7\n"
    header_str += f"{fields}\n"
    header_str += f"{size}\n"
    header_str += f"{type_}\n"
    header_str += f"{count}\n"
    header_str += "WIDTH {}\n"
    header_str += "HEIGHT 1\n"
    header_str += "VIEWPOINT 0 0 0 1 0 0 0\n"
    header_str += "POINTS {}\n"
    header_str += "DATA ascii\n"
    return header_str, type_list


def write_pcd(filename, points: list, headers: list):
    header, type_list = generate_headers(headers)
    if not points or len(points[0]) != len(headers):
        raise ValueError("points value not attached to headers .")
    num_points = len(points)
    pt_len = len(headers)
    header = header.format(num_points, num_points)

    with open(filename, 'w') as f:
        f.write(header)
        for pt in points:
            line = ""
            for i in range(pt_len):
                if type_list[i] == 'F':
                    line += f"{pt[i]:.6f} "
                else:
                    line += f"{int(pt[i])} "
            line = line.rstrip()
            f.write(line + '\n')
