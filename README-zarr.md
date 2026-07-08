# dct3d-zarr — a Zarr v3 codec for dct3d

`dct3d-zarr` exposes the [dct3d](./README.md) 16³ lossy volume codec as a
**Zarr v3 codec** named `dct3d`. Install it and stock `zarr` reads and writes
dct3d-compressed arrays with no further wiring:

```sh
pip install dct3d-zarr
```

```python
import numpy as np
import zarr
from zarr.codecs import ShardingCodec
from dct3d_zarr import Dct3dCodec

# One dct3d block == one 16³ zarr chunk. Aggregate many into a shard so a whole
# scroll isn't billions of tiny files — sharding is zarr-native; dct3d only ever
# sees a single 16³ block.
arr = zarr.create_array(
    store="scroll.zarr",
    shape=(1024, 1024, 1024),
    chunks=(512, 512, 512),                       # shard (the on-disk object)
    dtype="uint8",
    serializer=ShardingCodec(
        chunk_shape=(16, 16, 16),                 # inner chunk = one dct3d block
        codecs=[Dct3dCodec(quality=1.0, tau=2.0)],
    ),
    fill_value=0,
)
arr[:] = my_volume
```

Reading needs **no import at all** — the codec registers via a `zarr.codecs`
entry point, so any process that opens the array resolves `dct3d` automatically:

```python
import zarr
back = zarr.open_array(store="scroll.zarr", mode="r")[:]   # just works
```

## The design in one paragraph

The codec is deliberately geometry-fixed: it compresses exactly one **16³** block
per call — the atomic unit dct3d already operates on — and each encoded block is
fully self-contained (it carries its own value range and quantizer step), so it is
**independently decodable**. That makes it a clean Zarr v3 `array→bytes` codec.
Anything larger than 16³ is handled by zarr's native `sharding_indexed` codec,
which packs many 16³ inner chunks into one shard object with an index; dct3d and
sharding compose without either knowing about the other.

## Codec parameters

`Dct3dCodec(quality=1.0, max_error=0.0, tau=0.0)`

| param | meaning |
|-------|---------|
| `quality` | quantizer coarseness; smaller = higher fidelity, larger output. Auto-normalized per chunk so one value means comparable *relative* fidelity across dtypes. |
| `max_error` | optional *relative* per-voxel error bound in `[0, 1)` (fraction of the chunk's value range). `0` disables. |
| `tau` | optional *absolute* per-voxel error bound in raw input units. `0` disables. If both are set, the tighter wins. |

Supported dtypes: `uint8 uint16 uint32 int8 int16 int32 float32` (the seven dct3d
types). It is a **lossy** codec — reconstruction is within the requested bound, not
bit-exact, and (fast-math float) not bit-reproducible across builds.

## Requirements

- Python ≥ 3.10, `zarr` ≥ 3.0, `numpy`, `cffi`.
- Prebuilt wheels ship the compiled C extension (no toolchain needed). On a
  platform without a wheel, `pip` builds from the sdist, which needs a C23 compiler.

## Building from source

```sh
pip install -e ".[test]"   # editable install (compiles the cffi extension)
pytest -q                  # round-trip + entry-point discovery tests
```
