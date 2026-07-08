# dct3d_export — stream a scroll volume to a sharded dct3d Zarr v3 over SFTP

A bespoke, multithreaded C tool that recompresses an OME-Zarr scroll volume from
the Vesuvius open-data S3 bucket into a dct3d-compressed, sharded **Zarr v3**
volume and uploads it to an SFTP server — with **no Python / numcodecs in the
loop**. Each worker thread owns one shard end to end: fetch → encode → upload →
free.

## Pipeline (per shard, one thread each)

1. **Fetch** the source 128³ u8 chunks covering a `SHARD_VOX`³ (1024³) region via
   libs3 batched ranged GETs, assembled into a dense volume (zero-padded past the
   real extent).
2. **Encode** the 64³ inner 16³ blocks with `dct3d_encode_u8` at the level's
   (quality, tau), packed into a Zarr v3 `sharding_indexed` shard (inner 16³,
   index_location=end, crc32c — byte-compatible with the `dct3d-zarr` reader).
3. **Upload** the shard to `<root>/<scroll>/<vol>.zarr/<L>/<z>/<y>/<x>` over SFTP.
4. **Free** the shard; move to the next.

Array dims are padded up to a whole multiple of 1024, so every shard is a full
1024³ (edge shards zero-filled). All-air inner chunks are stored as the sharding
"missing" sentinel (no DCT, read back as fill_value=0).

## Air oracle (fast empty-region skip)

A level-L voxel downsampled 16× lives in level L+4. Because the volume is masked
(air==0) and downsampling averages, a coarse voxel is 0 **iff** every fine voxel
under it is 0 — a safe air signal (never a false skip; verified against real
data). Per shard the tool fetches the small 64³ oracle region (**L0←L4, L1←L5**;
deeper levels have no 16×-coarser level and fetch directly):

- whole-shard: if the oracle region is all zero, skip the shard's ~512 source
  GETs and emit the all-sentinel shard directly.
- per-source-chunk: a 128³ source chunk maps to an 8³ oracle region; if that is
  zero, skip fetching that chunk.

`--no-oracle` disables it (output is byte-identical either way — the oracle only
skips provable air).

## Quality ladder

`tau = 2·quality`; each coarser level halves both; quality floors at 1. L0
quality is picked from the volume's voxel size (parsed from the `--um` arg):

| voxel µm | L0 quality |
|---|---|
| ~0.5, ~1.1 | 64 |
| ~2.2, 2.4  | 32 |
| ~4.3       | 16 |
| ~7.9, 9.4  | 8  |
| 45+        | 2  |

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # needs libcurl (+SSH), pthreads
cmake --build build -j
```

`dct3d.c` is compiled from the repo root; libs3 is vendored under `vendor/`.

## Usage

```sh
dct3d_export \
  --base   https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis3/volumes/<vol>.zarr \
  --scroll PHercParis3 \
  --vol    <vol>.zarr \
  --um     2.4 \
  --levels 0-5 \
  --threads 6 \
  --remote-root exports \
  --sftp-host dl.ash2txt.org --sftp-port 9238 \
  --sftp-user U --sftp-pass X --known-hosts ~/.ssh/known_hosts
```

Uploads land at `<remote-root>/<scroll>/<vol>/<L>/<z>/<y>/<x>` with a per-level
`zarr.json` (dct3d sharding codec) and a group `zarr.json`.

Flags: `--dry-run` (fetch+encode, no upload), `--limit-shards N` (first N),
`--only-shard z,y,x` (a single shard, for testing), `--no-oracle`.

## Memory

Each worker holds one 1 GiB dense volume + its shard output (~1.3 GiB peak), so
`--threads` × ~1.3 GiB must fit RAM (≈6 threads on a 16 GB box).
