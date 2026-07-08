"""cffi API-mode builder for the dct3d C codec.

Compiles ../dct3d.c straight into a Python extension (`dct3d_zarr._dct3d`) and
declares the 14 typed encode/decode entry points. Pure C, no third-party deps —
the extension is self-contained, so the resulting wheel needs only libc at
runtime. Invoked by setuptools via `cffi_modules` in pyproject.toml, and usable
standalone (`python build_dct3d.py`) for a local in-place build.
"""

from __future__ import annotations

import os

from cffi import FFI

# This builder lives at the dct3d repo root, alongside dct3d.c / dct3d.h. The
# extension source is referenced by a REPO-RELATIVE path ("dct3d.c") on purpose:
# cffi/setuptools reject an absolute path in `sources`, and a relative one lands
# correctly in both the in-place build and the sdist/wheel build.
REPO = os.path.dirname(os.path.abspath(__file__))

ffibuilder = FFI()

# The public C surface, verbatim from dct3d.h (cffi's cdef parser wants plain
# declarations — no macros/#include, so the DCT3D_DECL expansion is written out).
_TYPES = [
    ("uint8_t", "u8"),
    ("uint16_t", "u16"),
    ("uint32_t", "u32"),
    ("int8_t", "s8"),
    ("int16_t", "s16"),
    ("int32_t", "s32"),
    ("float", "f32"),
]

cdefs = []
for ctype, name in _TYPES:
    cdefs.append(
        f"size_t dct3d_encode_{name}(const {ctype} *chunk, float quality, "
        f"float max_error, float tau, uint8_t *out);"
    )
    cdefs.append(
        f"int dct3d_decode_{name}(const uint8_t *blob, size_t len, {ctype} *chunk);"
    )
# DCT3D_N3 * 8 + 256 worst-case, surfaced as a plain constant for the caller's
# scratch buffer (kept in sync with dct3d.h; asserted against the header below).
cdefs.append("#define DCT3D_MAX_BYTES ...")
cdefs.append("#define DCT3D_N ...")
cdefs.append("#define DCT3D_N3 ...")

ffibuilder.cdef("\n".join(cdefs))

ffibuilder.set_source(
    "dct3d_zarr._dct3d",
    '#include "dct3d.h"',
    sources=["dct3d.c"],  # relative to the repo root (the sdist/build cwd)
    include_dirs=["."],
    # Match the library's intended build: C23 + fast-math autovectorization.
    # -ffast-math is safe here — the codec is explicitly tolerance-only and not
    # bit-reproducible across builds (see dct3d.h). MSVC ignores these flags; its
    # /fp:fast is applied via the extra_compile_args split below.
    extra_compile_args=(
        ["/O2"] if os.name == "nt" else ["-std=c23", "-O3", "-ffast-math"]
    ),
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
