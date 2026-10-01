"""Versioned cache representation, independent of native byte order."""

from array import array
import math
import struct
import sys
from .common import BridgeError

ENCODING = "f32le/1"


def decode_vector(data, dim):
    if type(dim) is not int or not 1 <= dim <= 65536 or len(data) != dim * 4:
        raise BridgeError("cached vector has invalid byte length or dimension")
    result = array("f")
    if result.itemsize != 4:
        result.extend(struct.unpack("<" + str(dim) + "f", data))
    else:
        result.frombytes(data)
        if sys.byteorder != "little":
            result.byteswap()
    if not all(math.isfinite(v) for v in result) or not any(v != 0 for v in result):
        raise BridgeError("cached vector is non-finite or zero")
    return result


def encode_vector(values):
    try:
        data = struct.pack("<" + str(len(values)) + "f", *values)
    except (ValueError, TypeError, OverflowError, struct.error) as error:
        raise BridgeError("invalid float32 vector") from error
    decode_vector(data, len(values))
    return data
