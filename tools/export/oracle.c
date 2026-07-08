// oracle.c — see oracle.h.

#include "oracle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SRC_CHUNK 128
#define SRC_CHUNK3 ((size_t)SRC_CHUNK * SRC_CHUNK * SRC_CHUNK)

// The oracle region for a shard is ORACLE_EDGE=64 voxels/axis. At oracle level
// those 64 voxels span oracle-voxel range [ooz, ooz+64) where ooz = oz/16.
// They come from source 128^3 oracle chunks, so a 64^3 region touches at most
// 1-2 oracle chunks per axis. We fetch those and scatter into out->vox.

static void scatter(uint8_t *dst, const uint8_t *chunk, int64_t lz, int64_t ly, int64_t lx) {
    const int64_t E = ORACLE_EDGE;
    for (int z = 0; z < SRC_CHUNK; ++z) {
        int64_t vz = lz + z; if (vz < 0 || vz >= E) continue;
        for (int y = 0; y < SRC_CHUNK; ++y) {
            int64_t vy = ly + y; if (vy < 0 || vy >= E) continue;
            int64_t x0 = lx < 0 ? 0 : lx, x1 = (lx + SRC_CHUNK) > E ? E : (lx + SRC_CHUNK);
            if (x1 <= x0) continue;
            memcpy(dst + ((size_t)vz * E + vy) * E + x0,
                   chunk + ((size_t)z * SRC_CHUNK + y) * SRC_CHUNK + (x0 - lx),
                   (size_t)(x1 - x0));
        }
    }
}

int oracle_fetch(s3_client *client, const char *oracle_base, int oracle_level,
                 const int64_t oshape[3], int64_t oz, int64_t oy, int64_t ox,
                 oracle_region *out) {
    memset(out, 0, sizeof(*out));
    if (oracle_level < 0) { out->valid = 0; return 0; }

    // Oracle-voxel origin of this shard's region.
    int64_t ooz = oz / ORACLE_SCALE, ooy = oy / ORACLE_SCALE, oox = ox / ORACLE_SCALE;
    const int64_t E = ORACLE_EDGE;

    int64_t cz0 = ooz / SRC_CHUNK, cz1 = (ooz + E - 1) / SRC_CHUNK;
    int64_t cy0 = ooy / SRC_CHUNK, cy1 = (ooy + E - 1) / SRC_CHUNK;
    int64_t cx0 = oox / SRC_CHUNK, cx1 = (oox + E - 1) / SRC_CHUNK;
    int64_t ncz = (oshape[0] + SRC_CHUNK - 1) / SRC_CHUNK;
    int64_t ncy = (oshape[1] + SRC_CHUNK - 1) / SRC_CHUNK;
    int64_t ncx = (oshape[2] + SRC_CHUNK - 1) / SRC_CHUNK;

    size_t maxn = (size_t)(cz1 - cz0 + 1) * (cy1 - cy0 + 1) * (cx1 - cx0 + 1);
    typedef struct { int64_t cz, cy, cx; } coord;
    coord *coords = malloc(maxn * sizeof(coord));
    char (*urls)[1280] = malloc(maxn * sizeof(*urls));
    uint8_t *bufs = malloc(maxn * SRC_CHUNK3);
    s3_range_req *reqs = calloc(maxn, sizeof(s3_range_req));
    s3_response *resp = calloc(maxn, sizeof(s3_response));
    if (!coords || !urls || !bufs || !reqs || !resp) {
        free(coords); free(urls); free(bufs); free(reqs); free(resp); return -1;
    }

    size_t n = 0;
    for (int64_t cz = cz0; cz <= cz1; ++cz) {
        if (cz < 0 || cz >= ncz) continue;
        for (int64_t cy = cy0; cy <= cy1; ++cy) {
            if (cy < 0 || cy >= ncy) continue;
            for (int64_t cx = cx0; cx <= cx1; ++cx) {
                if (cx < 0 || cx >= ncx) continue;
                coords[n] = (coord){cz, cy, cx};
                snprintf(urls[n], sizeof(urls[n]), "%s/%d/%lld/%lld/%lld",
                         oracle_base, oracle_level,
                         (long long)cz, (long long)cy, (long long)cx);
                reqs[n] = (s3_range_req){urls[n], 0, SRC_CHUNK3, bufs + n * SRC_CHUNK3};
                n++;
            }
        }
    }

    int rc = 0;
    if (n > 0) {
        s3_status st = s3_get_batch(client, reqs, n, 0, resp);
        if (st != S3_OK && st != S3_ERR_HTTP) { rc = -1; }
        else {
            for (size_t i = 0; i < n && rc == 0; ++i) {
                long s = resp[i].status;
                if (s == 404) continue;  // air
                if (s < 200 || s >= 300 || resp[i].body_len != SRC_CHUNK3) {
                    fprintf(stderr, "oracle: %s -> HTTP %ld / %zu bytes (hard error)\n",
                            urls[i], s, resp[i].body_len);
                    rc = -1; break;
                }
                scatter(out->vox, bufs + i * SRC_CHUNK3,
                        coords[i].cz * SRC_CHUNK - ooz,
                        coords[i].cy * SRC_CHUNK - ooy,
                        coords[i].cx * SRC_CHUNK - oox);
            }
        }
    }
    for (size_t i = 0; i < n; ++i) s3_response_free(&resp[i]);
    free(coords); free(urls); free(bufs); free(reqs); free(resp);

    if (rc != 0) return -1;
    out->valid = 1;
    out->all_zero = 1;
    for (size_t i = 0; i < (size_t)E * E * E; ++i)
        if (out->vox[i]) { out->all_zero = 0; break; }
    return 0;
}

int oracle_chunk_is_air(const oracle_region *o, int64_t lz, int64_t ly, int64_t lx) {
    if (!o->valid) return 0;
    // A source 128^3 chunk at export-level offset (lz,ly,lx) into the shard maps
    // to oracle voxels [lz/16, lz/16 + 8) etc (128/16 = 8).
    const int64_t E = ORACLE_EDGE;
    int64_t vz0 = lz / ORACLE_SCALE, vy0 = ly / ORACLE_SCALE, vx0 = lx / ORACLE_SCALE;
    for (int64_t z = vz0; z < vz0 + SRC_CHUNK / ORACLE_SCALE; ++z) {
        if (z < 0 || z >= E) continue;
        for (int64_t y = vy0; y < vy0 + SRC_CHUNK / ORACLE_SCALE; ++y) {
            if (y < 0 || y >= E) continue;
            for (int64_t x = vx0; x < vx0 + SRC_CHUNK / ORACLE_SCALE; ++x) {
                if (x < 0 || x >= E) continue;
                if (o->vox[((size_t)z * E + y) * E + x]) return 0;
            }
        }
    }
    return 1;
}
