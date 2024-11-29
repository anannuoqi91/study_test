from pydantic import BaseModel
from typing import Optional
from enum import Enum


class Boxes(BaseModel):
    track_id: int


class BoxType(Enum):
    UNKNOWN = 0
    PEDESTRIAN = 1
    CYCLIST = 2
    CAR = 3
    TRUCK = 4
    BUS = 5
