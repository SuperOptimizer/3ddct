# dct3d

A tiny, dependency-free lossy codec for **16×16×16 voxel chunks**. A 3D float
DCT-II, a dead-zone quantizer, and an adaptive binary range coder, packaged as a
single `.c` / `.h` pair.

Extracted and generalized from the block codec in
[SuperOptimizer/matter-compressor](https://github.com/SuperOptimizer/matter-compressor)
(the Vesuvius Challenge scroll compressor).

- **One dependency:** the C standard library. Nothing else.
- **Seven dtypes:** `u8 u16 u32 s8 s16 s32 f32`.
- **Auto-normalized quality:** one `quality` number means comparable *relative*
  fidelity for any dtype and any data scale — the codec measures each chunk's
  own value range and quantizes relative to it.
- **Two error bounds:** a *relative* `max_error` and an *absolute* `tau`, either
  of which enables a sparse exact-correction pass.
- **Float-only core.** Every value/transform/quant/normalization computation is
  `float`. Integers appear only where they must: inside the range coder (a
  bit-exact arithmetic coder), byte I/O, and loop indices.
- **Reentrant, lock-free, thread-free.** No shared mutable state on the hot path,
  no `pthread`, no globals to guard. Call it from as many of *your own* threads
  as you like; the library starts none.
- **C23, pure scalar, no SIMD** — written to auto-vectorize under `-O2/-O3
  -ffast-math`.

## Build

```sh
cc -std=c23 -O3 -ffast-math -c dct3d.c
```

Just drop `dct3d.c` and `dct3d.h` into your project. No build system, no config.

## API

One encode/decode pair per dtype:

```c
size_t dct3d_encode_u16(const uint16_t *chunk, float quality,
                        float max_error, float tau, uint8_t *out);
int    dct3d_decode_u16(const uint8_t *blob, size_t len, uint16_t *chunk);
```

…and likewise `_u8 _u32 _s8 _s16 _s32 _f32`.

| arg | meaning |
|-----|---------|
| `chunk` | `DCT3D_N3` (=4096) voxels, z-major: index `(z*16 + y)*16 + x` |
| `quality` | quantizer coarseness, `> 0`. Smaller = higher fidelity, bigger blob. Auto-normalized. |
| `max_error` | optional **relative** bound in `[0,1)`: caps per-voxel error at `max_error × chunk_value_range`. `0` disables. |
| `tau` | optional **absolute** bound in raw input units (`tau=2` → within ±2). `0` disables. |
| `out` | output buffer, capacity `>= DCT3D_MAX_BYTES` |

If both `max_error` and `tau` are set, the tighter one wins. `encode` returns the
blob length; `decode` returns `1` on success, `0` on a malformed/truncated blob
(it never reads or writes out of bounds, and rejects a blob decoded with the
wrong-typed function).

## Example

```c
#include "dct3d.h"

uint16_t chunk[DCT3D_N3];              // your 16^3 data
// ... fill chunk ...

uint8_t blob[DCT3D_MAX_BYTES];
size_t  n = dct3d_encode_u16(chunk, /*quality*/ 2.0f,
                             /*max_error*/ 0.0f, /*tau*/ 4.0f, blob);
// n bytes in `blob`; every reconstructed voxel will be within ±4 of the input.

uint16_t back[DCT3D_N3];
if (!dct3d_decode_u16(blob, n, back)) { /* corrupt blob */ }
```

Chunk your own volume however you like (tile a 128³ into 8³ blocks of 16³, etc.)
and call the codec per block — that stays embarrassingly parallel across your
threads.

## Real-world results

Measured on **dense-center scroll chunks** from the Vesuvius Challenge
`PHercParis4` 2.4 µm masked volume
(`20260411134726-2.400um-0.2m-78keV-masked.zarr`), the same data this codec was
tuned on. Six 128³ chunks near the scroll core were pulled from S3, each tiled
into 512 blocks of 16³ (u8). All six are **100 % material** (no air), mean
intensity 60–100, spanning the full 0–255 range.

Averaged over all six chunks (3M voxels), quality swept `1…64`, each alone and
with `tau = 2·quality`:

```
setting        ratio   b/vox   PSNR    MAE    SSIM     err p90/p95/p99/max
------------------------------------------------------------------------
q1              6.1x   0.168  50.14  0.515  0.9997     1 /  2 /  2 /  7
q1  tau=2       5.2x   0.196  50.67  0.483  0.9997     1 /  1 /  2 /  2
q2              9.7x   0.106  47.00  0.814  0.9994     2 /  2 /  3 /  9
q2  tau=4       9.2x   0.111  47.11  0.806  0.9994     2 /  2 /  3 /  4
q4             15.3x   0.066  43.93  1.208  0.9988     3 /  3 /  4 / 14
q4  tau=8      15.2x   0.067  43.95  1.207  0.9988     3 /  3 /  4 /  8
q8             24.3x   0.042  40.84  1.746  0.9975     4 /  5 /  7 / 22
q8  tau=16     24.2x   0.042  40.84  1.746  0.9975     4 /  5 /  7 / 16
q16            38.6x   0.026  37.77  2.486  0.9948     5 /  7 / 10 / 35
q16 tau=32     38.6x   0.026  37.77  2.486  0.9948     5 /  7 / 10 / 30
q32            59.8x   0.017  34.81  3.478  0.9894     7 / 10 / 14 / 50
q32 tau=64     59.8x   0.017  34.81  3.478  0.9894     7 / 10 / 14 / 50
q64            87.4x   0.011  32.01  4.763  0.9793    10 / 14 / 20 / 76
q64 tau=128    87.4x   0.011  32.01  4.763  0.9793    10 / 14 / 20 / 76
```

- **ratio** = raw ÷ compressed; **b/vox** = bytes per voxel (raw is 1.0).
- **PSNR** in dB, **MAE** mean absolute error, **SSIM** global structural
  similarity, and the **p90/p95/p99/max** columns are percentiles of the
  per-voxel absolute error.

### Reading the numbers

- **q1 ≈ visually lossless:** ~6× smaller, 50 dB, SSIM 0.9997, 99 % of voxels
  off by ≤2. **q8 ≈ 24×** still holds 41 dB / 0.997 SSIM.
- **Graceful high end:** q64 reaches ~87× at 32 dB and SSIM still 0.98 — a
  usable preview at ~0.01 bytes/voxel.
- **`tau` does what it says.** The `max` error column collapses to exactly `tau`
  wherever the bound bites (`q1 tau=2` → max 2, `q2 tau=4` → max 4, …). Because
  the correction residuals are coded as small integer deltas (not raw floats),
  the bound is nearly free: `q2 tau=4` costs ~5 % over plain `q2`, and by q≥16
  the natural error is already under `2q`, so `tau=2q` is a no-op (identical
  rows).

### Cross-dtype sanity

Auto-normalization means the same `quality` gives the same *relative* fidelity
regardless of dtype. Synthetic blocks at `q=1.0`, RMSE as a fraction of the
value range: u8 ≈ 0.36 %, u16 ≈ 0.4 %, u32 ≈ 0.4 %, f32 ≈ 0.4 %. Absolute `tau`
is honored exactly for every type (e.g. f32 `tau=0.00998` → max error 0.00998).

## Performance

Single-threaded throughput on an **Apple M4** (Homebrew clang 22,
`-O3 -ffast-math -march=native`), measured over the dense-center scroll blocks
above (u8, one 16³ block = 4096 voxels = 4 KB raw):

```
  q      encode              decode
         MB/s   ns/block      MB/s   ns/block
  ----------------------------------------------
  q1     104     39300        87     46900
  q4     146     28100       139     29400
  q16    175     23400       181     22700
  q64    197     20800       213     19200
```

Throughput rises with `quality` because there are fewer significant coefficients
to range-code. MB/s is of raw voxel data; a block is ~20–47 µs each way.

Because the codec is **thread-free and every block is independent**, it scales
out trivially — the same M4 across its 10 cores (OpenMP, one block per work
item) hits **~624 MB/s encode at q1 and ~1.04 GB/s at q16**, ~6× the
single-thread rate. There are no locks or shared writable state to contend on;
the only shared read is the static cosine table.

## Notes & caveats

- **Lossy.** Reconstruction is close, not bit-exact.
- **Same-build codec.** f32 + fast-math is not bit-reproducible across ISAs / opt
  levels / FMA contraction, so an encoder and decoder compiled together agree
  exactly (and the `tau` bound holds), but a decoder built differently may drift
  by a small delta. If you need a hard cross-machine bound, compile both sides
  the same way (and pin `-ffp-contract=off`).
- **Fixed 16³ block.** The transform, scan order, and contexts are specialized to
  `DCT3D_N = 16`. Tile larger volumes yourself.
- **Robust to corruption.** Truncated or bit-flipped blobs are rejected or safely
  bounded — fuzzed with truncations and random bit flips, never crashes.

## License

Derived from matter-compressor; see that project for upstream licensing.
