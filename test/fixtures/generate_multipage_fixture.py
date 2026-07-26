"""Generate the COPC fixture whose hierarchy spans several pages.

``multipage.copc.laz`` is the file the tests read; this script only exists to
regenerate it. It is the sibling of ``fixture.copc.laz`` (same 40 m cube of
randomly coloured points) with two differences that matter: a tiny node
capacity, so the octree has many nodes over several levels, and an explicit
page assignment, so those nodes are split across many hierarchy pages instead
of the single root page copclib writes by default. Every real COPC we have
ships one page, which leaves the page scheduler and the "block on a page we
have not fetched yet" branch of the selection pass untested.

Needs ``copclib``, which lives in the reg-ui virtualenv on this machine:

    /home/paulhax/src/tele/telesculptor-web/reg-ui/.venv/bin/python \
      test/fixtures/generate_multipage_fixture.py

The output is small enough to commit (< 100 KB). The points are seeded, so a
rerun writes the same octree; the hierarchy entries come back in whatever order
copclib hands them over, so the bytes after the point data may differ.
"""

import collections
import pathlib

import copclib
import numpy as np

OUT = pathlib.Path(__file__).parent / "multipage.copc.laz"

SEED = 42
POINTS = 3000
# A node this small needs four octree levels to hold 3000 points.
CAPACITY = 32
# A node lives on the page of its ancestor at the deepest of these depths (or
# on the root page if it is shallower than all of them). Two page depths give
# three levels of nesting: root page -> depth-1 pages -> depth-2 pages.
PAGE_DEPTHS = (1, 2)

# A 40 m cube around a non-trivial origin, as in fixture.copc.laz.
MIN = np.array([100.0, 200.0, 50.0])
MAX = np.array([140.0, 240.0, 90.0])
SCALE = [0.001, 0.001, 0.001]
POINT_FORMAT = 7  # copclib writes 6-8 only; 7 is the one with RGB

CENTER = (MIN + MAX) / 2.0
HALFSIZE = float(np.max(MAX - MIN)) / 2.0
CUBE_SIZE = 2.0 * HALFSIZE


def voxel_key(point, depth):
    """Octree key of a point at ``depth``, as the (d, x, y, z) tuple COPC uses."""
    cell = CUBE_SIZE / 2**depth
    xyz = np.floor((point - (CENTER - HALFSIZE)) / cell).astype(int)
    return (depth, *(int(v) for v in np.clip(xyz, 0, 2**depth - 1)))


def build_octree(xyz, rng):
    """Split point indices into nodes: each keeps CAPACITY, the rest sink down."""
    nodes = {}
    pending = [((0, 0, 0, 0), np.arange(len(xyz)))]
    while pending:
        key, members = pending.pop()
        if len(members) <= CAPACITY:
            nodes[key] = members
            continue
        shuffled = rng.permutation(members)
        nodes[key] = shuffled[:CAPACITY]
        children = collections.defaultdict(list)
        for i in shuffled[CAPACITY:]:
            children[voxel_key(xyz[i], key[0] + 1)].append(i)
        pending.extend((k, np.array(v)) for k, v in children.items())
    return nodes


def page_key(key):
    """The hierarchy page a node key belongs on -- this is what makes it multipage."""
    depth = max(d for d in (0,) + PAGE_DEPTHS if d <= key[0])
    shift = key[0] - depth
    return (depth, key[1] >> shift, key[2] >> shift, key[3] >> shift)


def main():
    rng = np.random.default_rng(SEED)
    xyz = rng.uniform(MIN, MAX, size=(POINTS, 3))
    rgb16 = rng.integers(0, 65536, size=(POINTS, 3), dtype=np.uint16)
    nodes = build_octree(xyz, rng)

    config = copclib.CopcConfigWriter(POINT_FORMAT, SCALE, list(MIN))
    config.las_header.min = [float(v) for v in xyz.min(axis=0)]
    config.las_header.max = [float(v) for v in xyz.max(axis=0)]
    config.copc_info.center_x, config.copc_info.center_y, config.copc_info.center_z = CENTER
    config.copc_info.halfsize = HALFSIZE
    # Point spacing of the root node: CAPACITY points spread over the cube.
    config.copc_info.spacing = CUBE_SIZE / CAPACITY ** (1 / 3)

    writer = copclib.FileWriter(str(OUT), config)
    for key in sorted(nodes):
        points = copclib.Points(writer.copc_config.las_header)
        for i in nodes[key]:
            point = points.CreatePoint()
            point.x, point.y, point.z = xyz[i]
            point.rgb = tuple(int(v) for v in rgb16[i])
            points.AddPoint(point)
        writer.AddNode(
            copclib.VoxelKey(list(key)),
            points,
            copclib.VoxelKey(list(page_key(key))),
        )
    writer.Close()

    pages = {page_key(k) for k in nodes}
    levels = {k[0] for k in nodes}
    print(
        f"wrote {OUT} points={POINTS} nodes={len(nodes)} "
        f"pages={len(pages)} levels={len(levels)} bytes={OUT.stat().st_size}"
    )


main()
