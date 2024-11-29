import pandas as pd
import numpy as np


class GroundFinder:
    def __init__(self, points, **kwargs):
        self.points = pd.DataFrame(points, columns=['x', 'y', 'z'])
        self._bounds()
        self._init_params(kwargs)

    def _init_params(self, kwargs):
        self._grid_size = kwargs.get('grid_size', 0.5)
        self._height_diff = kwargs.get('height_diff', 1.0)
        self._voxel_size = kwargs.get('voxel_size', 0.1)
        self._max_iterations = kwargs.get('max_iterations', 100)

    def find_ground(self):
        self._calculate_height_differences()

    def _bounds(self):
        self._x_min = self.points['x'].min()
        self._x_max = self.points['x'].max()
        self._y_min = self.points['y'].min()
        self._y_max = self.points['y'].max()
        self._z_min = self.points['z'].min()
        self._z_max = self.points['z'].max()

    def _filter_by_height(self):
        self.points = self.points.sort_values(by='x')  # 假设 'x' 列表示高度
        cutoff_index = int(len(self.points) * 0.85)  # 计算保留的 85% 的索引
        self.points = self.points.iloc[:cutoff_index]  # 去除最后的 15%
        self.points = self.points.reset_index(drop=True)  # 重置索引

    def _calculate_height_differences(self):
        points1 = []
        points2 = []
        # 创建一个新的 DataFrame 来存储网格的高度信息
        grid_height = {}
        grid_points = {}

        # 将点云划分到网格中
        for _, point in self.points.iterrows():
            grid_z = int(point['z'] // self._grid_size)
            grid_y = int(point['y'] // self._grid_size)
            grid_key = (grid_z, grid_y)

            if grid_key not in grid_height:
                grid_height[grid_key] = []
                grid_points[grid_key] = []
            grid_height[grid_key].append(point['x'])
            grid_points[grid_key].append([point['x'], point['y'], point['z']])

        # 计算每个网格的高度差
        height_differences = {}
        for grid_key, heights in grid_height.items():
            if heights:
                min_height = min(heights)
                max_height = max(heights)
                height_diff = max_height - min_height
                height_differences[grid_key] = {
                    'height_diff': height_diff,
                    'min_height': min_height,
                    'max_height': max_height,
                    'std': np.std(np.array(heights)),
                    # 'confidence': 1.0 if height_diff < self._height_diff else 0.0,
                    # 'neighbor_confidence': 0.0
                }
        for grid_key in height_differences:
            z = grid_key[0] * self._grid_size
            y = grid_key[1] * self._grid_size
            if height_differences[grid_key]['std'] < 0.3 and height_differences[grid_key]['height_diff'] < 1:
                points2.extend(grid_points[grid_key])
            else:
                mean_h = 0
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [0, -1, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        neighbor_key = (grid_key[0] + dx, grid_key[1] + dy)
                        if neighbor_key in height_differences and height_differences[neighbor_key]['height_diff'] < 1 and height_differences[neighbor_key]['std'] < 0.3:
                            mean_h += height_differences[neighbor_key]['max_height']
                            count += 1
                if count > 0:
                    mean_h /= count
                    for i_pt in grid_points[grid_key]:
                        if i_pt[0] < mean_h:
                            points2.append(i_pt)
            points1.append([0, y, z, height_differences[grid_key]
                            ['std'], height_differences[grid_key]['height_diff']])
        from write_pcd import write_pcd
        write_pcd('2_ground_height_diff.pcd', points1, [
                  'x:F', 'y:F', 'z:F', 'height_std:F', 'height_diff:F'])
        write_pcd('2_ground_height_diff-g.pcd', points2, ['x:F', 'y:F', 'z:F'])

        # # 计算与8个方向上网格的高度差
        # for grid_key in list(height_differences.keys()):
        #     for dx in [-1, 0, 1]:
        #         for dy in [-1, 0, 1]:
        #             if dx == 0 and dy == 0:
        #                 continue  # 跳过自身网格
        #             neighbor_key = (grid_key[0] + dx, grid_key[1] + dy)
        #             if neighbor_key in height_differences:
        #                 neighbor_diff = abs(height_differences[neighbor_key]['min_height'] -
        #                                     height_differences[grid_key]['min_height'])
        #                 if neighbor_diff < self._height_diff:
        #                     height_differences[grid_key]['neighbor_confidence'] += 1

        # return height_differences

    @property
    def bounds(self):
        return {'x': [self._x_min, self._x_max],
                'y': [self._y_min, self._y_max],
                'z': [self._z_min, self._z_max]}


if __name__ == "__main__":
    import open3d as o3d
    pcd_path = "/home/demo/Documents/code/omnisense/launch/calib_matrices/intermediate_lidar_01-shoufeizhan/orign_ground/2.pcd"

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = pcd.points

    gf = GroundFinder(points)
    gf.find_ground()
