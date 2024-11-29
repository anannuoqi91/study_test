# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/omnisense/common/proto/pose.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from modules.common.proto import header_pb2 as modules_dot_common_dot_proto_dot_header__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/omnisense/common/proto/pose.proto',
  package='omnisense.drivers.seyond',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n)modules/omnisense/common/proto/pose.proto\x12\x18omnisense.drivers.seyond\x1a!modules/common/proto/header.proto\"8\n\x08PointENU\x12\x0e\n\x01x\x18\x01 \x01(\x01:\x03nan\x12\x0e\n\x01y\x18\x02 \x01(\x01:\x03nan\x12\x0c\n\x01z\x18\x03 \x01(\x01:\x01\x30\"P\n\nQuaternion\x12\x0f\n\x02qx\x18\x01 \x01(\x01:\x03nan\x12\x0f\n\x02qy\x18\x02 \x01(\x01:\x03nan\x12\x0f\n\x02qz\x18\x03 \x01(\x01:\x03nan\x12\x0f\n\x02qw\x18\x04 \x01(\x01:\x03nan\"\x9e\x01\n\x04Pose\x12%\n\x06header\x18\x01 \x01(\x0b\x32\x15.apollo.common.Header\x12\x34\n\x08position\x18\x02 \x01(\x0b\x32\".omnisense.drivers.seyond.PointENU\x12\x39\n\x0borientation\x18\x03 \x01(\x0b\x32$.omnisense.drivers.seyond.Quaternion\"5\n\x05Poses\x12,\n\x04pose\x18\x01 \x03(\x0b\x32\x1e.omnisense.drivers.seyond.Pose\"\xb2\x01\n\x07\x43\x61nData\x12%\n\x06header\x18\x01 \x01(\x0b\x32\x15.apollo.common.Header\x12\x1f\n\x10is_yawrate_valid\x18\x02 \x01(\x08:\x05\x66\x61lse\x12\x12\n\x07yawrate\x18\x03 \x01(\x02:\x01\x30\x12 \n\x11is_velocity_valid\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x13\n\x08velocity\x18\x05 \x01(\x02:\x01\x30\x12\x14\n\ttimestamp\x18\x06 \x01(\x03:\x01\x30\"\xc7\x01\n\x0cSurfaceParam\x12\r\n\x05param\x18\x01 \x03(\x02\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12\r\n\x05min_z\x18\x03 \x01(\x02\x12\r\n\x05max_z\x18\x04 \x01(\x02\x12\x15\n\rdetach_thresh\x18\x05 \x03(\x02\x12\x0e\n\x06mean_y\x18\x06 \x01(\x02\x12\x0e\n\x06mean_z\x18\x07 \x01(\x02\x12\x11\n\tvar_y_inv\x18\x08 \x01(\x02\x12\x11\n\tvar_z_inv\x18\t \x01(\x02\x12\x0c\n\x04mean\x18\n \x01(\x02\x12\x0b\n\x03std\x18\x0b \x01(\x02\"\x8a\x03\n\x0bRoadSurface\x12%\n\x06header\x18\x01 \x01(\x0b\x32\x15.apollo.common.Header\x12\x37\n\x07surface\x18\x02 \x03(\x0b\x32&.omnisense.drivers.seyond.SurfaceParam\x12\x14\n\tsource_id\x18\x03 \x01(\r:\x01\x30\x12\x0e\n\x03idx\x18\x04 \x01(\x04:\x01\x30\x12\x14\n\x0chas_boundary\x18\x05 \x01(\x08\x12\x18\n\x10road_cloud_index\x18\x06 \x03(\r\x12\x36\n\x0eposes_in_frame\x18\x07 \x03(\x0b\x32\x1e.omnisense.drivers.seyond.Pose\x12@\n\x10surface_opposite\x18\x08 \x03(\x0b\x32&.omnisense.drivers.seyond.SurfaceParam\x12\x18\n\x10\x64ist_range_start\x18\t \x03(\x02\x12\x16\n\x0e\x64ist_range_end\x18\n \x03(\x02\x12\x19\n\x11min_detach_thresh\x18\x0b \x03(\x02'
  ,
  dependencies=[modules_dot_common_dot_proto_dot_header__pb2.DESCRIPTOR,])




_POINTENU = _descriptor.Descriptor(
  name='PointENU',
  full_name='omnisense.drivers.seyond.PointENU',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='omnisense.drivers.seyond.PointENU.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=(1e10000 * 0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='omnisense.drivers.seyond.PointENU.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=(1e10000 * 0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='z', full_name='omnisense.drivers.seyond.PointENU.z', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=106,
  serialized_end=162,
)


_QUATERNION = _descriptor.Descriptor(
  name='Quaternion',
  full_name='omnisense.drivers.seyond.Quaternion',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='qx', full_name='omnisense.drivers.seyond.Quaternion.qx', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=(1e10000 * 0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='qy', full_name='omnisense.drivers.seyond.Quaternion.qy', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=(1e10000 * 0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='qz', full_name='omnisense.drivers.seyond.Quaternion.qz', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=(1e10000 * 0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='qw', full_name='omnisense.drivers.seyond.Quaternion.qw', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=(1e10000 * 0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=164,
  serialized_end=244,
)


_POSE = _descriptor.Descriptor(
  name='Pose',
  full_name='omnisense.drivers.seyond.Pose',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='omnisense.drivers.seyond.Pose.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='position', full_name='omnisense.drivers.seyond.Pose.position', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='orientation', full_name='omnisense.drivers.seyond.Pose.orientation', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=247,
  serialized_end=405,
)


_POSES = _descriptor.Descriptor(
  name='Poses',
  full_name='omnisense.drivers.seyond.Poses',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='pose', full_name='omnisense.drivers.seyond.Poses.pose', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=407,
  serialized_end=460,
)


_CANDATA = _descriptor.Descriptor(
  name='CanData',
  full_name='omnisense.drivers.seyond.CanData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='omnisense.drivers.seyond.CanData.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='is_yawrate_valid', full_name='omnisense.drivers.seyond.CanData.is_yawrate_valid', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='yawrate', full_name='omnisense.drivers.seyond.CanData.yawrate', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='is_velocity_valid', full_name='omnisense.drivers.seyond.CanData.is_velocity_valid', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='velocity', full_name='omnisense.drivers.seyond.CanData.velocity', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='omnisense.drivers.seyond.CanData.timestamp', index=5,
      number=6, type=3, cpp_type=2, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=463,
  serialized_end=641,
)


_SURFACEPARAM = _descriptor.Descriptor(
  name='SurfaceParam',
  full_name='omnisense.drivers.seyond.SurfaceParam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='param', full_name='omnisense.drivers.seyond.SurfaceParam.param', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='omnisense.drivers.seyond.SurfaceParam.confidence', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='min_z', full_name='omnisense.drivers.seyond.SurfaceParam.min_z', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='max_z', full_name='omnisense.drivers.seyond.SurfaceParam.max_z', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='detach_thresh', full_name='omnisense.drivers.seyond.SurfaceParam.detach_thresh', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mean_y', full_name='omnisense.drivers.seyond.SurfaceParam.mean_y', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mean_z', full_name='omnisense.drivers.seyond.SurfaceParam.mean_z', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='var_y_inv', full_name='omnisense.drivers.seyond.SurfaceParam.var_y_inv', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='var_z_inv', full_name='omnisense.drivers.seyond.SurfaceParam.var_z_inv', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mean', full_name='omnisense.drivers.seyond.SurfaceParam.mean', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='std', full_name='omnisense.drivers.seyond.SurfaceParam.std', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=644,
  serialized_end=843,
)


_ROADSURFACE = _descriptor.Descriptor(
  name='RoadSurface',
  full_name='omnisense.drivers.seyond.RoadSurface',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='omnisense.drivers.seyond.RoadSurface.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='surface', full_name='omnisense.drivers.seyond.RoadSurface.surface', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='source_id', full_name='omnisense.drivers.seyond.RoadSurface.source_id', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='idx', full_name='omnisense.drivers.seyond.RoadSurface.idx', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='has_boundary', full_name='omnisense.drivers.seyond.RoadSurface.has_boundary', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='road_cloud_index', full_name='omnisense.drivers.seyond.RoadSurface.road_cloud_index', index=5,
      number=6, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='poses_in_frame', full_name='omnisense.drivers.seyond.RoadSurface.poses_in_frame', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='surface_opposite', full_name='omnisense.drivers.seyond.RoadSurface.surface_opposite', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dist_range_start', full_name='omnisense.drivers.seyond.RoadSurface.dist_range_start', index=8,
      number=9, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dist_range_end', full_name='omnisense.drivers.seyond.RoadSurface.dist_range_end', index=9,
      number=10, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='min_detach_thresh', full_name='omnisense.drivers.seyond.RoadSurface.min_detach_thresh', index=10,
      number=11, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=846,
  serialized_end=1240,
)

_POSE.fields_by_name['header'].message_type = modules_dot_common_dot_proto_dot_header__pb2._HEADER
_POSE.fields_by_name['position'].message_type = _POINTENU
_POSE.fields_by_name['orientation'].message_type = _QUATERNION
_POSES.fields_by_name['pose'].message_type = _POSE
_CANDATA.fields_by_name['header'].message_type = modules_dot_common_dot_proto_dot_header__pb2._HEADER
_ROADSURFACE.fields_by_name['header'].message_type = modules_dot_common_dot_proto_dot_header__pb2._HEADER
_ROADSURFACE.fields_by_name['surface'].message_type = _SURFACEPARAM
_ROADSURFACE.fields_by_name['poses_in_frame'].message_type = _POSE
_ROADSURFACE.fields_by_name['surface_opposite'].message_type = _SURFACEPARAM
DESCRIPTOR.message_types_by_name['PointENU'] = _POINTENU
DESCRIPTOR.message_types_by_name['Quaternion'] = _QUATERNION
DESCRIPTOR.message_types_by_name['Pose'] = _POSE
DESCRIPTOR.message_types_by_name['Poses'] = _POSES
DESCRIPTOR.message_types_by_name['CanData'] = _CANDATA
DESCRIPTOR.message_types_by_name['SurfaceParam'] = _SURFACEPARAM
DESCRIPTOR.message_types_by_name['RoadSurface'] = _ROADSURFACE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PointENU = _reflection.GeneratedProtocolMessageType('PointENU', (_message.Message,), {
  'DESCRIPTOR' : _POINTENU,
  '__module__' : 'modules.omnisense.common.proto.pose_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.drivers.seyond.PointENU)
  })
_sym_db.RegisterMessage(PointENU)

Quaternion = _reflection.GeneratedProtocolMessageType('Quaternion', (_message.Message,), {
  'DESCRIPTOR' : _QUATERNION,
  '__module__' : 'modules.omnisense.common.proto.pose_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.drivers.seyond.Quaternion)
  })
_sym_db.RegisterMessage(Quaternion)

Pose = _reflection.GeneratedProtocolMessageType('Pose', (_message.Message,), {
  'DESCRIPTOR' : _POSE,
  '__module__' : 'modules.omnisense.common.proto.pose_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.drivers.seyond.Pose)
  })
_sym_db.RegisterMessage(Pose)

Poses = _reflection.GeneratedProtocolMessageType('Poses', (_message.Message,), {
  'DESCRIPTOR' : _POSES,
  '__module__' : 'modules.omnisense.common.proto.pose_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.drivers.seyond.Poses)
  })
_sym_db.RegisterMessage(Poses)

CanData = _reflection.GeneratedProtocolMessageType('CanData', (_message.Message,), {
  'DESCRIPTOR' : _CANDATA,
  '__module__' : 'modules.omnisense.common.proto.pose_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.drivers.seyond.CanData)
  })
_sym_db.RegisterMessage(CanData)

SurfaceParam = _reflection.GeneratedProtocolMessageType('SurfaceParam', (_message.Message,), {
  'DESCRIPTOR' : _SURFACEPARAM,
  '__module__' : 'modules.omnisense.common.proto.pose_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.drivers.seyond.SurfaceParam)
  })
_sym_db.RegisterMessage(SurfaceParam)

RoadSurface = _reflection.GeneratedProtocolMessageType('RoadSurface', (_message.Message,), {
  'DESCRIPTOR' : _ROADSURFACE,
  '__module__' : 'modules.omnisense.common.proto.pose_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.drivers.seyond.RoadSurface)
  })
_sym_db.RegisterMessage(RoadSurface)


# @@protoc_insertion_point(module_scope)
