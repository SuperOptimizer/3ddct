// fetch.c — see fetch.h.

#include "fetch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oracle.h"

#define SRC_CHUNK 128
#define SRC_CHUNK3 ((size_t)SRC_CHUNK * SRC_CHUNK * SRC_CHUNK)

// Scatter a fetched 128^3 source chunk into the dense SHARD_VOX^3 vol, clipped to
// the shard bounds (a source chunk straddling the shard edge contributes only
// its in-shard part). (lz,ly,lx) is the chunk origin relative to the shard.
static void scatter_chunk(uint8_t *vol, const uint8_t *chunk,
                          int64_t lz, int64_t ly, int64_t lx) {
    const int64_t S = SHARD_VOX;
    for (int z = 0; z < SRC_CHUNK; ++z) {
        int64_t vz = lz + z;
        if (vz < 0 || vz >= S) continue;
        for (int y = 0; y < SRC_CHUNK; ++y) {
            int64_t vy = ly + y;
            if (vy < 0 || vy >= S) continue;
            // x run: intersect [lx, lx+128) with [0, S)
            int64_t x0 = lx < 0 ? 0 : lx;
            int64_t x1 = (lx + SRC_CHUNK) > S ? S : (lx + SRC_CHUNK);
            if (x1 <= x0) continue;
            const uint8_t *src = chunk + ((size_t)z * SRC_CHUNK + y) * SRC_CHUNK + (x0 - lx);
            uint8_t *dst = vol + (((size_t)vz * S + vy) * S + x0);
            memcpy(dst, src, (size_t)(x1 - x0));
        }
    }
}

int fetch_shard_region(s3_client *client, const src_level *lvl,
                       int64_t oz, int64_t oy, int64_t ox, uint8_t *vol,
                       const struct oracle_region *oracle) {
    memset(vol, 0, (size_t)SHARD_VOX * SHARD_VOX * SHARD_VOX);

    // Source chunk index range covering [o, o+SHARD_VOX) on each axis, clipped to
    // the real level extent (chunks past `shape` don't exist -> stay zero).
    const int64_t *shp = lvl->shape;
    int64_t cz0 = oz / SRC_CHUNK, cz1 = (oz + SHARD_VOX - 1) / SRC_CHUNK;
    int64_t cy0 = oy / SRC_CHUNK, cy1 = (oy + SHARD_VOX - 1) / SRC_CHUNK;
    int64_t cx0 = ox / SRC_CHUNK, cx1 = (ox + SHARD_VOX - 1) / SRC_CHUNK;
    int64_t ncz = (shp[0] + SRC_CHUNK - 1) / SRC_CHUNK;
    int64_t ncy = (shp[1] + SRC_CHUNK - 1) / SRC_CHUNK;
    int64_t ncx = (shp[2] + SRC_CHUNK - 1) / SRC_CHUNK;

    // Build the list of in-range chunk coords + URLs.
    size_t maxn = (size_t)(cz1 - cz0 + 1) * (cy1 - cy0 + 1) * (cx1 - cx0 + 1);
    typedef struct { int64_t cz, cy, cx; } coord;
    coord *coords = (coord *)malloc(maxn * sizeof(coord));
    char (*urls)[1280] = malloc(maxn * sizeof(*urls));
    uint8_t *bufs = (uint8_t *)malloc(maxn * SRC_CHUNK3);
    s3_range_req *reqs = (s3_range_req *)calloc(maxn, sizeof(s3_range_req));
    s3_response *resp = (s3_response *)calloc(maxn, sizeof(s3_response));
    if (!coords || !urls || !bufs || !reqs || !resp) {
        free(coords); free(urls); free(bufs); free(reqs); free(resp);
        return -1;
    }

    size_t n = 0;
    for (int64_t cz = cz0; cz <= cz1; ++cz) {
        if (cz < 0 || cz >= ncz) continue;
        for (int64_t cy = cy0; cy <= cy1; ++cy) {
            if (cy < 0 || cy >= ncy) continue;
            for (int64_t cx = cx0; cx <= cx1; ++cx) {
                if (cx < 0 || cx >= ncx) continue;
                // Oracle air-skip: if the coarse level says this 128^3 chunk is
                // all air, don't fetch it (vol stays zero there).
                if (oracle && oracle_chunk_is_air(
                        oracle, cz * SRC_CHUNK - oz, cy * SRC_CHUNK - oy,
                        cx * SRC_CHUNK - ox))
                    continue;
                coords[n].cz = cz; coords[n].cy = cy; coords[n].cx = cx;
                snprintf(urls[n], sizeof(urls[n]), "%s/%d/%lld/%lld/%lld",
                         lvl->base, lvl->level,
                         (long long)cz, (long long)cy, (long long)cx);
                reqs[n].url = urls[n];
                reqs[n].offset = 0;
                reqs[n].length = SRC_CHUNK3;      // whole raw chunk
                reqs[n].dst = bufs + n * SRC_CHUNK3;  // zero-copy into our buffer
                n++;
            }
        }
    }

    int rc = 0;
    if (n > 0) {
        // Batched concurrent GET. Transport-level failures -> S3_OK still, per-req
        // status inspected below.
        s3_status st = s3_get_batch(client, reqs, n, 0, resp);
        if (st != S3_OK && st != S3_ERR_HTTP) {
            fprintf(stderr, "fetch: s3_get_batch failed: %s\n", s3_status_str(st));
            rc = -1;
        } else {
            for (size_t i = 0; i < n && rc == 0; ++i) {
                long status = resp[i].status;
                if (status == 404) {
                    // omitted == air; leave zeros
                    continue;
                }
                if (status < 200 || status >= 300) {
                    fprintf(stderr, "fetch: %s -> HTTP %ld (not 404) — hard error\n",
                            urls[i], status);
                    rc = -1;
                    break;
                }
                if (resp[i].body_len != SRC_CHUNK3) {
                    fprintf(stderr, "fetch: %s -> %zu bytes (want %zu) — hard error\n",
                            urls[i], resp[i].body_len, SRC_CHUNK3);
                    rc = -1;
                    break;
                }
                int64_t lz = coords[i].cz * SRC_CHUNK - oz;
                int64_t ly = coords[i].cy * SRC_CHUNK - oy;
                int64_t lx = coords[i].cx * SRC_CHUNK - ox;
                scatter_chunk(vol, bufs + i * SRC_CHUNK3, lz, ly, lx);
            }
        }
    }

    for (size_t i = 0; i < n; ++i) s3_response_free(&resp[i]);
    free(coords); free(urls); free(bufs); free(reqs); free(resp);
    return rc;
}
