from pydantic import BaseModel
from typing import Optional


class PointCloud2(BaseModel):
    x: float
    y: float
    z: float
    intensity: float
    scan_id: Optional[int] = 0
    scan_idx: Optional[int] = 0
    flags: Optional[int]

    @staticmethod
    def create_from_protobuf(protobuf):
        tmp = {
            'x': protobuf.x,
            'y': protobuf.y,
            'z': protobuf.z,
            'intensity': protobuf.intensity
        }
        try:
            tmp['scan_id'] = protobuf.scan_id
        except BaseException:
            pass
        try:
            tmp['scan_idx'] = protobuf.scan_idx
        except BaseException:
            pass
        try:
            tmp['flags'] = protobuf.flags
        except BaseException:
            pass
        return PointCloud2(**tmp)


class PointXYZI(BaseModel):
    x: float
    y: float
    z: float
    intensity: float


class PointXYZ(BaseModel):
    x: float
    y: float
    z: float
