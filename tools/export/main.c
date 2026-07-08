// dct3d_export — stream a source OME-zarr volume level to a dct3d-compressed,
// sharded Zarr v3 volume on an SFTP server.
//
// Per shard (SHARD_VOX^3, one worker thread each): batched-fetch the source
// 128^3 chunks from S3/HTTP, assemble a dense volume (zero-padded past the real
// extent), dct3d-encode its 16^3 inner chunks into a sharding_indexed shard,
// SFTP it to optimized/<scroll>/<vol>.zarr/<L>/<z>/<y>/<x>, then free it. The
// array dims are padded up to a whole multiple of SHARD_VOX, so every shard is
// a full SHARD_VOX^3 (edge shards are zero-filled past the volume).
//
// Usage:
//   dct3d_export --base <https-zarr-root> --scroll <PHercId> --vol <name.zarr>
//                --um <voxel-um> --levels 0-5 --threads N
//                --sftp-host H --sftp-port P --sftp-user U --sftp-pass X
//                [--known-hosts path] [--remote-root optimized]
//                [--limit-shards N] [--dry-run]

#include <curl/curl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fetch.h"
#include "ladder.h"
#include "oracle.h"
#include "shard.h"
#include "upload.h"
#include "vendor/libs3.h"

// ---- source metadata (.zarray) ---------------------------------------------

// Minimal ".zarray" scrape: pull shape[3]. Source levels are known raw u8 128^3;
// we only need the shape to size the grid.
static int parse_shape(const char *json, int64_t shape[3]) {
    const char *s = strstr(json, "\"shape\"");
    if (!s) return -1;
    s = strchr(s, '[');
    if (!s) return -1;
    return sscanf(s, "[ %lld , %lld , %lld",
                  (long long *)&shape[0], (long long *)&shape[1],
                  (long long *)&shape[2]) == 3
               ? 0
               : -1;
}

static int fetch_level_shape(s3_client *c, const char *base, int level,
                             int64_t shape[3]) {
    char url[1280];
    snprintf(url, sizeof(url), "%s/%d/.zarray", base, level);
    s3_response r = {0};
    s3_status st = s3_get(c, url, &r);
    int rc = -1;
    if ((st == S3_OK || st == S3_ERR_HTTP) && r.status == 200 && r.body)
        rc = parse_shape((const char *)r.body, shape);
    s3_response_free(&r);
    return rc;
}

// ---- work queue ------------------------------------------------------------

typedef struct {
    int64_t sz, sy, sx;  // shard grid coordinate
} shard_coord;

typedef struct {
    shard_coord *items;
    size_t n, next;
    pthread_mutex_t lock;
    atomic_size_t done, failed, skipped;
} work_queue;

typedef struct {
    s3_client *client;
    src_level lvl;
    float quality, tau;
    const char *remote_root;   // "exports"
    const char *scroll;        // PHercId
    const char *vol;           // <name>.zarr
    sftp_target sftp;
    work_queue *q;
    int dry_run;
    int worker_id;
    // Air oracle (L0<-L4, L1<-L5); oracle_level < 0 => disabled.
    int oracle_level;
    char oracle_base[1024];
    int64_t oracle_shape[3];
    atomic_size_t *air_shards;  // count of whole-shard skips (shared)
} worker_ctx;

static int next_shard(work_queue *q, shard_coord *out) {
    pthread_mutex_lock(&q->lock);
    int have = 0;
    if (q->next < q->n) { *out = q->items[q->next++]; have = 1; }
    pthread_mutex_unlock(&q->lock);
    return have;
}

static void *worker_main(void *arg) {
    worker_ctx *w = (worker_ctx *)arg;
    uint8_t *vol = (uint8_t *)malloc((size_t)SHARD_VOX * SHARD_VOX * SHARD_VOX);
    if (!vol) { fprintf(stderr, "[w%d] OOM vol\n", w->worker_id); return NULL; }

    shard_coord sc;
    while (next_shard(w->q, &sc)) {
        int64_t oz = sc.sz * SHARD_VOX, oy = sc.sy * SHARD_VOX, ox = sc.sx * SHARD_VOX;

        // Fast path 1: a shard whose origin is at or past the real (unpadded)
        // extent on any axis is entirely in the zero-padded region — every
        // source chunk would 404 to air. Skip fetch, emit the all-sentinel shard.
        const int64_t *shp = w->lvl.shape;
        int fully_air = (oz >= shp[0] || oy >= shp[1] || ox >= shp[2]);

        oracle_region orc;
        orc.valid = 0;
        if (!fully_air && w->oracle_level >= 0) {
            if (oracle_fetch(w->client, w->oracle_base, w->oracle_level,
                             w->oracle_shape, oz, oy, ox, &orc) != 0) {
                // Oracle fetch failed hard — fall back to fetching everything.
                orc.valid = 0;
            } else if (orc.valid && orc.all_zero) {
                // Fast path 2: the coarse oracle says the whole shard is air.
                fully_air = 1;
                atomic_fetch_add(w->air_shards, 1);
            }
        }

        if (fully_air) {
            memset(vol, 0, (size_t)SHARD_VOX * SHARD_VOX * SHARD_VOX);
        } else if (fetch_shard_region(w->client, &w->lvl, oz, oy, ox, vol,
                                      orc.valid ? &orc : NULL) != 0) {
            fprintf(stderr, "[w%d] fetch failed shard %lld/%lld/%lld L%d\n",
                    w->worker_id, (long long)sc.sz, (long long)sc.sy,
                    (long long)sc.sx, w->lvl.level);
            atomic_fetch_add(&w->q->failed, 1);
            continue;
        }

        uint8_t *shard = NULL;
        size_t shard_len = 0;
        if (shard_encode_u8(vol, w->quality, w->tau, &shard, &shard_len) != 0) {
            fprintf(stderr, "[w%d] encode OOM shard %lld/%lld/%lld\n",
                    w->worker_id, (long long)sc.sz, (long long)sc.sy, (long long)sc.sx);
            atomic_fetch_add(&w->q->failed, 1);
            continue;
        }

        char remote[1536];
        snprintf(remote, sizeof(remote), "/%s/%s/%s/%d/%lld/%lld/%lld",
                 w->remote_root, w->scroll, w->vol, w->lvl.level,
                 (long long)sc.sz, (long long)sc.sy, (long long)sc.sx);

        int ok = 0;
        if (w->dry_run) {
            ok = 1;
        } else {
            ok = sftp_upload(&w->sftp, remote, shard, shard_len) == 0;
        }
        free(shard);

        if (ok) {
            atomic_fetch_add(&w->q->done, 1);
        } else {
            fprintf(stderr, "[w%d] upload failed %s\n", w->worker_id, remote);
            atomic_fetch_add(&w->q->failed, 1);
        }
    }
    free(vol);
    return NULL;
}

// ---- zarr.json emission ----------------------------------------------------

// Emit the per-level array zarr.json (padded shape, dct3d codec) and upload it.
static int upload_level_metadata(const worker_ctx *base, int64_t padded[3],
                                 float quality, float tau) {
    char zj[2048];
    snprintf(zj, sizeof(zj),
             "{\n"
             "  \"zarr_format\": 3,\n"
             "  \"node_type\": \"array\",\n"
             "  \"shape\": [%lld, %lld, %lld],\n"
             "  \"data_type\": \"uint8\",\n"
             "  \"chunk_grid\": {\"name\": \"regular\", \"configuration\": "
             "{\"chunk_shape\": [%d, %d, %d]}},\n"
             "  \"chunk_key_encoding\": {\"name\": \"default\", \"configuration\": "
             "{\"separator\": \"/\"}},\n"
             "  \"fill_value\": 0,\n"
             "  \"codecs\": [{\"name\": \"sharding_indexed\", \"configuration\": {\n"
             "    \"chunk_shape\": [%d, %d, %d],\n"
             "    \"codecs\": [{\"name\": \"dct3d\", \"configuration\": "
             "{\"quality\": %g, \"max_error\": 0.0, \"tau\": %g}}],\n"
             "    \"index_codecs\": [{\"name\": \"bytes\", \"configuration\": "
             "{\"endian\": \"little\"}}, {\"name\": \"crc32c\"}],\n"
             "    \"index_location\": \"end\"\n"
             "  }}]\n"
             "}\n",
             (long long)padded[0], (long long)padded[1], (long long)padded[2],
             SHARD_VOX, SHARD_VOX, SHARD_VOX, INNER, INNER, INNER,
             (double)quality, (double)tau);

    char remote[1536];
    snprintf(remote, sizeof(remote), "/%s/%s/%s/%d/zarr.json",
             base->remote_root, base->scroll, base->vol, base->lvl.level);
    if (base->dry_run) {
        printf("[dry-run] would write %s (%zu B)\n", remote, strlen(zj));
        return 0;
    }
    return sftp_upload(&base->sftp, remote, (const uint8_t *)zj, strlen(zj));
}

// ---- CLI -------------------------------------------------------------------

typedef struct {
    const char *base, *scroll, *vol, *remote_root, *known_hosts;
    const char *sftp_host, *sftp_user, *sftp_pass;
    int sftp_port, threads, lvl_lo, lvl_hi, dry_run;
    long limit_shards;
    double um;
    int no_oracle;
    int64_t only[3];   // if only[0]>=0, process just this one shard coord
} args_t;

static void parse_levels(const char *s, int *lo, int *hi) {
    if (sscanf(s, "%d-%d", lo, hi) == 2) return;
    if (sscanf(s, "%d", lo) == 1) { *hi = *lo; return; }
    *lo = 0; *hi = 5;
}

static const char *argval(int argc, char **argv, int *i) {
    if (*i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[*i]); exit(2); }
    return argv[++(*i)];
}

int main(int argc, char **argv) {
    args_t a = {0};
    a.remote_root = "exports";
    a.threads = 6;
    a.lvl_lo = 0; a.lvl_hi = 5;
    a.sftp_port = 22;
    a.limit_shards = -1;
    a.only[0] = -1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--base")) a.base = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--scroll")) a.scroll = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--vol")) a.vol = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--um")) a.um = atof(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--levels")) parse_levels(argval(argc, argv, &i), &a.lvl_lo, &a.lvl_hi);
        else if (!strcmp(argv[i], "--threads")) a.threads = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--remote-root")) a.remote_root = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--sftp-host")) a.sftp_host = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--sftp-port")) a.sftp_port = atoi(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--sftp-user")) a.sftp_user = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--sftp-pass")) a.sftp_pass = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--known-hosts")) a.known_hosts = argval(argc, argv, &i);
        else if (!strcmp(argv[i], "--limit-shards")) a.limit_shards = atol(argval(argc, argv, &i));
        else if (!strcmp(argv[i], "--no-oracle")) a.no_oracle = 1;
        else if (!strcmp(argv[i], "--only-shard")) {
            const char *v = argval(argc, argv, &i);
            if (sscanf(v, "%lld,%lld,%lld", (long long *)&a.only[0],
                       (long long *)&a.only[1], (long long *)&a.only[2]) != 3) {
                fprintf(stderr, "--only-shard wants z,y,x\n"); return 2;
            }
        }
        else if (!strcmp(argv[i], "--dry-run")) a.dry_run = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!a.base || !a.scroll || !a.vol || a.um <= 0) {
        fprintf(stderr, "required: --base --scroll --vol --um (and sftp opts unless --dry-run)\n");
        return 2;
    }
    if (!a.dry_run && (!a.sftp_host || !a.sftp_user || !a.sftp_pass)) {
        fprintf(stderr, "--sftp-host/user/pass required unless --dry-run\n");
        return 2;
    }

    sftp_global_init();
    s3_config cfg = {0};
    cfg.transfer_timeout_s = 60;
    cfg.max_retries = 4;
    s3_client *client = s3_client_new(&cfg);
    if (!client) { fprintf(stderr, "s3_client_new failed\n"); return 1; }

    sftp_target sftp = {a.sftp_host, a.sftp_port, a.sftp_user, a.sftp_pass, a.known_hosts};

    for (int level = a.lvl_lo; level <= a.lvl_hi; ++level) {
        int64_t shape[3];
        if (fetch_level_shape(client, a.base, level, shape) != 0) {
            fprintf(stderr, "L%d: no .zarray (stopping levels)\n", level);
            break;
        }
        // Pad each dim up to a whole multiple of SHARD_VOX.
        int64_t padded[3], nshard[3];
        for (int d = 0; d < 3; ++d) {
            nshard[d] = (shape[d] + SHARD_VOX - 1) / SHARD_VOX;
            padded[d] = nshard[d] * SHARD_VOX;
        }
        float quality, tau;
        ladder_for_level(a.um, level, &quality, &tau);

        size_t total = (size_t)nshard[0] * nshard[1] * nshard[2];
        printf("L%d shape [%lld,%lld,%lld] -> padded [%lld,%lld,%lld] "
               "shards %lldx%lldx%lld = %zu | q=%g tau=%g\n",
               level, (long long)shape[0], (long long)shape[1], (long long)shape[2],
               (long long)padded[0], (long long)padded[1], (long long)padded[2],
               (long long)nshard[0], (long long)nshard[1], (long long)nshard[2],
               total, (double)quality, (double)tau);

        // Resolve the air oracle level (L0<-L4, L1<-L5) if it exists.
        int oracle_level = a.no_oracle ? -1 : oracle_level_for(level);
        int64_t oracle_shape[3] = {0, 0, 0};
        if (oracle_level >= 0) {
            if (fetch_level_shape(client, a.base, oracle_level, oracle_shape) != 0) {
                fprintf(stderr, "L%d: oracle L%d .zarray missing — oracle disabled\n",
                        level, oracle_level);
                oracle_level = -1;
            } else {
                printf("L%d: using air oracle L%d [%lld,%lld,%lld]\n", level,
                       oracle_level, (long long)oracle_shape[0],
                       (long long)oracle_shape[1], (long long)oracle_shape[2]);
            }
        }
        atomic_size_t air_shards = 0;

        work_queue q = {0};
        q.items = (shard_coord *)malloc(total * sizeof(shard_coord));
        pthread_mutex_init(&q.lock, NULL);
        size_t k = 0;
        if (a.only[0] >= 0) {
            q.items[k++] = (shard_coord){a.only[0], a.only[1], a.only[2]};
        } else {
            for (int64_t z = 0; z < nshard[0]; ++z)
                for (int64_t y = 0; y < nshard[1]; ++y)
                    for (int64_t x = 0; x < nshard[2]; ++x)
                        q.items[k++] = (shard_coord){z, y, x};
        }
        q.n = k;
        if (a.limit_shards >= 0 && (size_t)a.limit_shards < q.n) q.n = a.limit_shards;

        worker_ctx base = {0};
        base.client = client;
        base.lvl.level = level;
        base.lvl.src_chunk = 128;
        snprintf(base.lvl.base, sizeof(base.lvl.base), "%s", a.base);
        base.lvl.shape[0] = shape[0]; base.lvl.shape[1] = shape[1]; base.lvl.shape[2] = shape[2];
        base.quality = quality; base.tau = tau;
        base.remote_root = a.remote_root; base.scroll = a.scroll; base.vol = a.vol;
        base.sftp = sftp; base.q = &q; base.dry_run = a.dry_run;
        base.oracle_level = oracle_level;
        snprintf(base.oracle_base, sizeof(base.oracle_base), "%s", a.base);
        base.oracle_shape[0] = oracle_shape[0];
        base.oracle_shape[1] = oracle_shape[1];
        base.oracle_shape[2] = oracle_shape[2];
        base.air_shards = &air_shards;

        // Write the level metadata first so a reader sees a valid array.
        if (upload_level_metadata(&base, padded, quality, tau) != 0)
            fprintf(stderr, "L%d: metadata upload failed (continuing)\n", level);

        int nthreads = a.threads;
        pthread_t *tids = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
        worker_ctx *ctxs = (worker_ctx *)malloc(nthreads * sizeof(worker_ctx));
        for (int t = 0; t < nthreads; ++t) {
            ctxs[t] = base;
            ctxs[t].worker_id = t;
            pthread_create(&tids[t], NULL, worker_main, &ctxs[t]);
        }
        for (int t = 0; t < nthreads; ++t) pthread_join(tids[t], NULL);
        free(tids); free(ctxs);

        printf("L%d done: %zu ok, %zu failed of %zu (%zu whole-shard air-skips)\n",
               level, atomic_load(&q.done), atomic_load(&q.failed), q.n,
               atomic_load(&air_shards));
        free(q.items);
        pthread_mutex_destroy(&q.lock);
    }

    // Upload the OME group zarr.json once (references levels as multiscales).
    // Minimal group marker; a full multiscales block can be added later.
    if (!a.dry_run) {
        const char *group =
            "{\n  \"zarr_format\": 3,\n  \"node_type\": \"group\",\n"
            "  \"attributes\": {}\n}\n";
        char remote[1536];
        snprintf(remote, sizeof(remote), "/%s/%s/%s/zarr.json",
                 a.remote_root, a.scroll, a.vol);
        sftp_upload(&sftp, remote, (const uint8_t *)group, strlen(group));
    }

    s3_client_free(client);
    return 0;
}
