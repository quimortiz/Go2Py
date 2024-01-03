"""
  Generated by Eclipse Cyclone DDS idlc Python Backend
  Cyclone DDS IDL version: v0.11.0
  Module: msgs
  IDL file: LowCmd.idl

"""

from enum import auto
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

# root module import for resolving types
import msgs


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowCmd(idl.IdlStruct, typename="msgs.LowCmd"):
    q: types.array[types.float32, 12]
    dq: types.array[types.float32, 12]
    tau_ff: types.array[types.float32, 12]
    kp: types.array[types.float32, 12]
    kv: types.array[types.float32, 12]
    e_stop: types.uint8


