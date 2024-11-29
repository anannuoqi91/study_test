# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/omnisense/common/proto/inno_web.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from modules.common.proto import header_pb2 as modules_dot_common_dot_proto_dot_header__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/omnisense/common/proto/inno_web.proto',
  package='omnisense.web',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n-modules/omnisense/common/proto/inno_web.proto\x12\romnisense.web\x1a!modules/common/proto/header.proto\".\n\nWebJsonStr\x12\x0e\n\x06module\x18\x01 \x01(\t\x12\x10\n\x08json_str\x18\x02 \x01(\t\"_\n\x02KV\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x14\n\x0cstring_value\x18\x02 \x01(\t\x12\x12\n\nbool_value\x18\x03 \x01(\x08\x12\x14\n\x0cnumber_value\x18\x04 \x01(\x02\x12\x0c\n\x04type\x18\x05 \x01(\r\"<\n\nModuleKeys\x12\x0e\n\x06module\x18\x01 \x01(\t\x12\x1e\n\x03kvs\x18\x02 \x03(\x0b\x32\x11.omnisense.web.KV\"9\n\x07HotKeys\x12.\n\x0bmodule_keys\x18\x01 \x03(\x0b\x32\x19.omnisense.web.ModuleKeys'
  ,
  dependencies=[modules_dot_common_dot_proto_dot_header__pb2.DESCRIPTOR,])




_WEBJSONSTR = _descriptor.Descriptor(
  name='WebJsonStr',
  full_name='omnisense.web.WebJsonStr',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='module', full_name='omnisense.web.WebJsonStr.module', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='json_str', full_name='omnisense.web.WebJsonStr.json_str', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=99,
  serialized_end=145,
)


_KV = _descriptor.Descriptor(
  name='KV',
  full_name='omnisense.web.KV',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='omnisense.web.KV.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='string_value', full_name='omnisense.web.KV.string_value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bool_value', full_name='omnisense.web.KV.bool_value', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='number_value', full_name='omnisense.web.KV.number_value', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='omnisense.web.KV.type', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=147,
  serialized_end=242,
)


_MODULEKEYS = _descriptor.Descriptor(
  name='ModuleKeys',
  full_name='omnisense.web.ModuleKeys',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='module', full_name='omnisense.web.ModuleKeys.module', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='kvs', full_name='omnisense.web.ModuleKeys.kvs', index=1,
      number=2, type=11, cpp_type=10, label=3,
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
  serialized_start=244,
  serialized_end=304,
)


_HOTKEYS = _descriptor.Descriptor(
  name='HotKeys',
  full_name='omnisense.web.HotKeys',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='module_keys', full_name='omnisense.web.HotKeys.module_keys', index=0,
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
  serialized_start=306,
  serialized_end=363,
)

_MODULEKEYS.fields_by_name['kvs'].message_type = _KV
_HOTKEYS.fields_by_name['module_keys'].message_type = _MODULEKEYS
DESCRIPTOR.message_types_by_name['WebJsonStr'] = _WEBJSONSTR
DESCRIPTOR.message_types_by_name['KV'] = _KV
DESCRIPTOR.message_types_by_name['ModuleKeys'] = _MODULEKEYS
DESCRIPTOR.message_types_by_name['HotKeys'] = _HOTKEYS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

WebJsonStr = _reflection.GeneratedProtocolMessageType('WebJsonStr', (_message.Message,), {
  'DESCRIPTOR' : _WEBJSONSTR,
  '__module__' : 'modules.omnisense.common.proto.inno_web_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.web.WebJsonStr)
  })
_sym_db.RegisterMessage(WebJsonStr)

KV = _reflection.GeneratedProtocolMessageType('KV', (_message.Message,), {
  'DESCRIPTOR' : _KV,
  '__module__' : 'modules.omnisense.common.proto.inno_web_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.web.KV)
  })
_sym_db.RegisterMessage(KV)

ModuleKeys = _reflection.GeneratedProtocolMessageType('ModuleKeys', (_message.Message,), {
  'DESCRIPTOR' : _MODULEKEYS,
  '__module__' : 'modules.omnisense.common.proto.inno_web_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.web.ModuleKeys)
  })
_sym_db.RegisterMessage(ModuleKeys)

HotKeys = _reflection.GeneratedProtocolMessageType('HotKeys', (_message.Message,), {
  'DESCRIPTOR' : _HOTKEYS,
  '__module__' : 'modules.omnisense.common.proto.inno_web_pb2'
  # @@protoc_insertion_point(class_scope:omnisense.web.HotKeys)
  })
_sym_db.RegisterMessage(HotKeys)


# @@protoc_insertion_point(module_scope)
