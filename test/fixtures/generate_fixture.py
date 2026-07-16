"""Generate the tiny deterministic COPC fixture used by copcTileSource tests.

Run from the repo root with the telesculptor spike environment:

    PYTHONPATH=<scratch>/copc-spike/pylibs \
      <app-venv>/bin/python test/fixtures/generate_fixture.py

It reuses write_copc from plans/cloud/spike/spike_copc_common.py (copclib
octree writer). The output is small enough to commit (< 100 KB).
"""

import pathlib
import sys

import numpy as np

SPIKE_DIR = pathlib.Path("/home/paulhax/src/tele/plans/cloud/spike")
sys.path.insert(0, str(SPIKE_DIR))

from spike_copc_common import write_copc  # noqa: E402

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
