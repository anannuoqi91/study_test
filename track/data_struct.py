from enum import Enum
import copy
import math
from utils.util_angle import norm_angle_radius


class ObjectType(Enum):
    PEDESTRIAN = 0
    CYCLIST = 1
    CAR = 2
    TRUCK = 3
    BUS = 4
    UNKNOWN = 5


class PointXYZ:
    def __init__(self, x, y, z) -> None:
        self.x = x  # height
        self.y = y  # right
        self.z = z  # front


class ObjectBox:
    # up -x front - z  right -y
    def __init__(self, **kwargs):
        self._id = kwargs['id']
        self._length = kwargs['length']
        self._width = kwargs['width']
        self._height = kwargs['height']
        self._time_idx = getattr(kwargs, 'time_idx', 0)
        self._is_valid = True if self._time_idx > 0 else False
        self._type = getattr(kwargs, 'type', ObjectType.UNKNOWN)
        self._time_idx = getattr(kwargs, 'time_idx', 0)
        # N-z E-y U-x
        self._heading_azimuth_rad = norm_angle_radius(
            math.radians(kwargs['heading']))
        self._velocity = {'x': kwargs['speed_x'],
                          'y': kwargs['speed_y'], 'z': kwargs['speed_z']}
        self._center = PointXYZ(
            kwargs['center_x'], kwargs['center_y'], kwargs['center_z'])
        self._corners = self._cal_corners()
        self._covariance = []   # front right height v_z v_y v_z
        self._prob_match = 0
        self._age = 0

    @property
    def is_valid(self):
        return self._is_valid

    def set_position_center(self, x, y, z):
        self._center = {'x': x, 'y': y, 'z': z}

    def deep_copy_object(self):
        return copy.deepcopy(self)

    def set_covariance(self, covariance):
        self._covariance = covariance

    def bev_polygon(self):
        front_left = [self._corners['upper_front_left'].z,
                      self._corners['upper_front_left'].y]
        front_right = [self._corners['upper_front_right'].z,
                       self._corners['upper_front_right'].y]
        back_left = [self._corners['upper_back_left'].z,
                     self._corners['upper_back_left'].y]
        back_right = [self._corners['upper_back_right'].z,
                      self._corners['upper_back_right'].y]
        return [front_left, back_left, back_right, front_right]

    def _cal_corners(self):
        out = {}
        # front-left, back-left, back-right, front-right
        cosval = math.cos(self._heading_azimuth_rad)
        sinval = math.sin(self._heading_azimuth_rad)
        dz_fl = 0.5 * self._length * cosval - 0.5 * self._width * sinval
        dy_fl = 0.5 * self._length * sinval + 0.5 * self._width * cosval

        dz_bl = -0.5 * self._length * cosval - 0.5 * self._width * sinval
        dy_bl = -0.5 * self._length * sinval + 0.5 * self._width * cosval

        out['upper_front_left'] = PointXYZ(
            0.5 * self._height, self._center.y + dy_fl, self._center.z + dz_fl)
        out['upper_back_left'] = PointXYZ(
            0.5 * self._height, self._center.y + dy_bl, self._center.z + dz_bl)
        out['upper_back_right'] = PointXYZ(
            0.5 * self._height, self._center.y - dy_fl, self._center.z - dz_fl)
        out['upper_front_right'] = PointXYZ(
            0.5 * self._height, self._center.y - dy_bl, self._center.z - dz_bl)
        out['lower_front_left'] = PointXYZ(
            -0.5 * self._height, self._center.y + dy_fl, self._center.z + dz_fl)
        out['lower_back_left'] = PointXYZ(
            -0.5 * self._height, self._center.y + dy_bl, self._center.z + dz_bl)
        out['lower_back_right'] = PointXYZ(
            -0.5 * self._height, self._center.y - dy_fl, self._center.z - dz_fl)
        out['lower_front_right'] = PointXYZ(
            -0.5 * self._height, self._center.y - dy_bl, self._center.z - dz_bl)
        return out

    @property
    def position_center(self):
        return copy.deepcopy(self._center)

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def length(self):
        return self._length

    @property
    def type(self):
        return self._type

    @property
    def id(self):
        return self._id

    @property
    def time_idx(self):
        return self._time_idx

    @property
    def velocity(self):
        return self._velocity

    @property
    def heading_rad(self):
        return self._heading_azimuth_rad

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    def set_prob_match(self, prob):
        self._prob_match = prob

    @property
    def prob_match(self):
        return self._prob_match

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, age):
        self._age = age
