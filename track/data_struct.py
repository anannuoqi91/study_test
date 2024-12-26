from enum import Enum
import copy
import math
import warnings
from ..utils.util_angle import norm_angle_radius
from ..utils.util_logger import UtilLogger


class ObjectType(Enum):
    PEDESTRIAN = 0
    CYCLIST = 1
    CAR = 2
    TRUCK = 3
    BUS = 4
    UNKNOWN = 5

    @staticmethod
    def values():
        return [e.value for e in ObjectType]


class SourceDetect(Enum):
    AI = 1
    CL = 2

    @staticmethod
    def values():
        return [e.value for e in SourceDetect]


class PointXYZ:
    def __init__(self, x, y, z) -> None:
        self.x = x  # height
        self.y = y  # right
        self.z = z  # front


class ObjectBox:
    """
    The coordinate system of a single object:
    the center point as the origin, 
    the forward direction as the z-axis, 
    the right direction as the y-axis, 
    the upward direction as the x-axis. 
    """

    def __init__(self, **kwargs):
        self._init_logger(**kwargs)
        self._init_attributes(**kwargs)
        self._init_movement(**kwargs)
        self._init_match()
        self._corners_global = self._cal_corners()

    def _init_logger(self, **kwargs):
        self._logger = getattr(kwargs, 'logger', None)
        if self._logger is not None and not isinstance(self._logger, UtilLogger):
            warnings.warn("logger should be UtilLogger instance.")
            self._logger = None

    def _init_attributes(self, **kwargs):
        if self._logger is not None:
            self._logger.logger.debug("ObjectBox init attributes.")

        self._id = kwargs['id']
        self._length = kwargs['length']
        self._width = kwargs['width']
        self._height = kwargs['height']
        self._time_idx = getattr(kwargs, 'time_idx', 0)   # timestamp
        self._is_valid = True if self._time_idx > 0 else False
        self._type = getattr(kwargs, 'type', ObjectType.UNKNOWN)
        self._center_global = PointXYZ(
            kwargs['center_up'], kwargs['center_right'], kwargs['center_front'])
        self._source_detect = getattr(kwargs, 'type', SourceDetect.CL)

    def _init_movement(self, **kwargs):
        if self._logger is not None:
            self._logger.logger.debug("ObjectBox init movement.")

        self._heading_azimuth_rad = norm_angle_radius(
            math.radians(kwargs['heading']))
        self._velocity = {'up': kwargs['speed_up'],
                          'right': kwargs['speed_right'],
                          'front': kwargs['speed_front']}

    def _init_match(self):
        if self._logger is not None:
            self._logger.logger.debug("ObjectBox init match.")

        self._covariance = []   # front right height v_z v_y v_z
        self._prob_match = 0
        self._age = 0

    def yaw_rad(self):
        return norm_angle_radius(math.arctan2(self._center_global.z, self._center_global.y))

    def set_source_detect(self, source_detect):
        self._source_detect = source_detect

    def set_position_center(self, up, right, front):
        self._center_global = PointXYZ(up, right, front)

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

    def set_length(self, length):
        self._length = length

    def set_width(self, width):
        self._width = width

    def set_height(self, height):
        self._height = height

    def extend(self, front, back, left, right):
        if self._logger is not None:
            self._logger.logger.debug("ObjectBox extend.")

        out = copy.deepcopy()
        delta_vert = 0.5 * (front - back)
        delta_hori = 0.5 * (left - right)
        delta_x = delta_vert * \
            math.cos(out.heading_rad) - delta_hori * math.sin(out.heading_rad)
        delta_y = delta_vert * \
            math.sin(out.heading_rad) + delta_hori * math.cos(out.heading_rad)
        out.set_lenght(out.length + front + back)
        out.set_width(out.width + left + right)
        out.set_position_center(
            out.position_center['x'], out.position_center['y'] + delta_y, out.position_center['z'] + delta_x)
        return out

    def _cal_corners(self):
        if self._logger is not None:
            self._logger.logger.debug("ObjectBox cal_corners.")

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

    def set_id(self, id):
        self._id = id

    def set_prob_match(self, prob):
        self._prob_match = prob

    def set_age(self, age):
        self._age = age

    @property
    def prob_match(self):
        return self._prob_match

    @property
    def age(self):
        return self._age

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
        return copy.deepcopy(self._velocity)

    @property
    def heading_rad(self):
        return self._heading_azimuth_rad

    @property
    def id(self):
        return self._id

    @property
    def is_valid(self):
        return self._is_valid

    @property
    def source_detect(self):
        return self._source_detect
