"""Generate the tiny deterministic COPC fixture used by copcTileSource tests.

The committed ``fixture.copc.laz`` is what the test suite actually reads, so
you only need this script to regenerate it. It depends on a ``write_copc``
helper (a copclib octree writer) that is not vendored in this repo. Point
``COPC_WRITER_DIR`` at a directory containing a ``copc_writer`` module that
exposes ``write_copc(path, xyz, rgb16, *, scale, offset, capacity, seed)``:

    COPC_WRITER_DIR=/path/to/writer \
      python test/fixtures/generate_fixture.py

The output is small enough to commit (< 100 KB).
"""

import os
import pathlib
import sys

import numpy as np

WRITER_DIR = pathlib.Path(os.environ.get("COPC_WRITER_DIR", ".")).resolve()
sys.path.insert(0, str(WRITER_DIR))

from copc_writer import write_copc  # noqa: E402

OUT = pathlib.Path(__file__).parent / "fixture.copc.laz"

POINTS = 2000
rng = np.random.default_rng(42)
# A 40 m cube of full-range-colored points around a non-trivial origin.
xyz = rng.uniform([100.0, 200.0, 50.0], [140.0, 240.0, 90.0], size=(POINTS, 3))
rgb16 = rng.integers(0, 65536, size=(POINTS, 3), dtype=np.uint16)

node_count, elapsed = write_copc(
    str(OUT),
    xyz,
    rgb16,
    scale=[0.001, 0.001, 0.001],
    offset=[100.0, 200.0, 50.0],
    capacity=600,  # forces at least two hierarchy levels at 2000 points
    seed=7,
)
print(f"wrote {OUT} nodes={node_count} bytes={OUT.stat().st_size}")
