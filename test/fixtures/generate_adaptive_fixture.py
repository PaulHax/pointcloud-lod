"""Generate the deterministic COPC fixture for adaptive browser tests.

The ordinary committed fixtures contain only 2,000 and 3,000 points. That is
deliberately small for the full browser matrix, but it cannot exercise an
adaptive policy whose minimum track budget is 200,000 points: demand clamps
the track to its floor before a synthetic slow or fast frame can move it.

This fixture is used only by ``adaptive.spec.ts``. It has enough visible demand
to demonstrate both reduction and growth without making every browser scenario
render hundreds of thousands of points.

Regenerate it with the copclib environment used by the multipage fixture:

    /home/paulhax/src/tele/telesculptor-web/reg-ui/.venv/bin/python \
      test/fixtures/generate_adaptive_fixture.py
"""

import collections
import pathlib

import copclib
import numpy as np

OUT = pathlib.Path(__file__).parent / "adaptive.copc.laz"

SEED = 84
POINTS = 350_000
CAPACITY = 50_000
MIN = np.array([100.0, 200.0, 50.0])
MAX = np.array([140.0, 240.0, 90.0])
SCALE = [0.001, 0.001, 0.001]
POINT_FORMAT = 7

CENTER = (MIN + MAX) / 2.0
HALFSIZE = float(np.max(MAX - MIN)) / 2.0
CUBE_SIZE = 2.0 * HALFSIZE
ROOT = (0, 0, 0, 0)


def voxel_key(point, depth):
    """Return the COPC octree key containing ``point`` at ``depth``."""
    cell = CUBE_SIZE / 2**depth
    xyz = np.floor((point - (CENTER - HALFSIZE)) / cell).astype(int)
    return (depth, *(int(v) for v in np.clip(xyz, 0, 2**depth - 1)))


def build_octree(xyz, rng):
    """Keep up to CAPACITY points per node and distribute the remainder."""
    nodes = {}
    pending = [(ROOT, np.arange(len(xyz)))]
    while pending:
        key, members = pending.pop()
        if len(members) <= CAPACITY:
            nodes[key] = members
            continue
        shuffled = rng.permutation(members)
        nodes[key] = shuffled[:CAPACITY]
        children = collections.defaultdict(list)
        for index in shuffled[CAPACITY:]:
            children[voxel_key(xyz[index], key[0] + 1)].append(index)
        pending.extend((key, np.array(indices)) for key, indices in children.items())
    return nodes


def main():
    rng = np.random.default_rng(SEED)
    xyz = rng.uniform(MIN, MAX, size=(POINTS, 3))
    rgb16 = rng.integers(0, 65536, size=(POINTS, 3), dtype=np.uint16)
    nodes = build_octree(xyz, rng)

    config = copclib.CopcConfigWriter(POINT_FORMAT, SCALE, list(MIN))
    config.las_header.min = [float(value) for value in xyz.min(axis=0)]
    config.las_header.max = [float(value) for value in xyz.max(axis=0)]
    config.copc_info.center_x, config.copc_info.center_y, config.copc_info.center_z = CENTER
    config.copc_info.halfsize = HALFSIZE
    config.copc_info.spacing = CUBE_SIZE / CAPACITY ** (1 / 3)

    writer = copclib.FileWriter(str(OUT), config)
    for key in sorted(nodes):
        points = copclib.Points(writer.copc_config.las_header)
        for index in nodes[key]:
            point = points.CreatePoint()
            point.x, point.y, point.z = xyz[index]
            point.rgb = tuple(int(value) for value in rgb16[index])
            points.AddPoint(point)
        writer.AddNode(
            copclib.VoxelKey(list(key)),
            points,
            copclib.VoxelKey(list(ROOT)),
        )
    writer.Close()
    print(
        f"wrote {OUT} points={POINTS} nodes={len(nodes)} "
        f"bytes={OUT.stat().st_size}"
    )


main()
