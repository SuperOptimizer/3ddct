"""Thin numpy-facing wrapper over the cffi `_dct3d` extension.

One block == one 16^3 chunk. `encode_block` takes a C-contiguous 16^3 numpy
array of a supported dtype and returns the compressed `bytes`; `decode_block`
reverses it into a fresh array of the requested dtype/shape. All dtype dispatch
and buffer marshalling lives here so the zarr codec class stays declarative.
"""

from __future__ import annotations

import numpy as np

from ._dct3d import ffi, lib  # type: ignore[import-not-found]

N = lib.DCT3D_N  # 16
N3 = lib.DCT3D_N3  # 4096
MAX_BYTES = lib.DCT3D_MAX_BYTES  # worst-case encoded size for one block

BLOCK_SHAPE = (N, N, N)

# numpy dtype -> (encode fn, decode fn, C element type string for ffi.cast).
# Only these seven dtypes exist in the C library; anything else is rejected
# loudly rather than silently mis-cast.
_DISPATCH = {
    np.dtype("uint8"): (lib.dct3d_encode_u8, lib.dct3d_decode_u8, "uint8_t *"),
    np.dtype("uint16"): (lib.dct3d_encode_u16, lib.dct3d_decode_u16, "uint16_t *"),
    np.dtype("uint32"): (lib.dct3d_encode_u32, lib.dct3d_decode_u32, "uint32_t *"),
    np.dtype("int8"): (lib.dct3d_encode_s8, lib.dct3d_decode_s8, "int8_t *"),
    np.dtype("int16"): (lib.dct3d_encode_s16, lib.dct3d_decode_s16, "int16_t *"),
    np.dtype("int32"): (lib.dct3d_encode_s32, lib.dct3d_decode_s32, "int32_t *"),
    np.dtype("float32"): (lib.dct3d_encode_f32, lib.dct3d_decode_f32, "float *"),
}

SUPPORTED_DTYPES = tuple(str(d) for d in _DISPATCH)


class Dct3dError(ValueError):
    """Raised on an unsupported dtype, wrong-shaped block, or a malformed blob."""


def _resolve(dtype: np.dtype):
    # dct3d's core is native-endian float; the C entry points take native ints.
    # Normalize to the native byte order for lookup so e.g. '>u2' still maps to
    # the u16 path (the caller is responsible for having native-order data — the
    # zarr codec always hands us native arrays).
    key = np.dtype(dtype).newbyteorder("=") if dtype.byteorder not in ("=", "|") else np.dtype(dtype)
    try:
        return _DISPATCH[np.dtype(key)]
    except KeyError:
        raise Dct3dError(
            f"dct3d supports {SUPPORTED_DTYPES}, not {dtype!r}"
        ) from None


def encode_block(block: np.ndarray, quality: float, max_error: float, tau: float) -> bytes:
    """Compress one 16^3 block. Returns the encoded blob as bytes."""
    if block.size != N3:
        raise Dct3dError(f"dct3d block must be {N3} voxels ({BLOCK_SHAPE}), got {block.shape}")
    enc, _dec, cptr = _resolve(block.dtype)
    src = np.ascontiguousarray(block).reshape(N3)
    in_ptr = ffi.cast(cptr, ffi.from_buffer(src))
    out = ffi.new("uint8_t[]", MAX_BYTES)
    n = enc(in_ptr, float(quality), float(max_error), float(tau), out)
    if n <= 0 or n > MAX_BYTES:
        raise Dct3dError(f"dct3d encode produced an invalid length {n}")
    return bytes(ffi.buffer(out, n))


def decode_block(blob: bytes, dtype: np.dtype) -> np.ndarray:
    """Decompress one 16^3 block of `dtype` from `blob`. Returns a (16,16,16) array."""
    _enc, dec, cptr = _resolve(np.dtype(dtype))
    out = np.empty(N3, dtype=np.dtype(dtype).newbyteorder("="))
    out_ptr = ffi.cast(cptr, ffi.from_buffer(out))
    ok = dec(blob, len(blob), out_ptr)
    if not ok:
        raise Dct3dError("dct3d decode failed: malformed, truncated, or dtype-mismatched blob")
    return out.reshape(BLOCK_SHAPE)
