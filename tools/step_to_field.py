#!/usr/bin/env python3
"""Official RMUC2026 STEP -> MuJoCo-ready field, chunked per-solid export.

No decimation: every kept solid is meshed at collision-grade deflection and
exported as its own STL (auto-split to stay under MuJoCo's 200k-face cap).

Usage:
  python3 tools/step_to_field.py <input.step> <out.stl> \
      --min-volume 5e7 --top 60 --chunk-dir <dir>
"""
import argparse
import os
import sys

from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.StlAPI import StlAPI_Writer
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("out")
    ap.add_argument("--min-volume", type=float, default=1e7)
    ap.add_argument("--deflection", type=float, default=0.02)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--chunk-dir", type=str, default=None)
    ap.add_argument("--max-faces", type=int, default=190_000)
    args = ap.parse_args()

    r = STEPControl_Reader()
    if r.ReadFile(args.step) != IFSelect_RetDone:
        sys.exit("cannot read STEP")
    r.TransferRoots()
    shape = r.OneShape()

    downcast = getattr(TopoDS, "Solid_s", None) or TopoDS.Solid
    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        s = downcast(exp.Current())
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(s, props)
        solids.append((s, props.Mass()))
        exp.Next()
    solids.sort(key=lambda x: -x[1])
    print(f"total solids: {len(solids)}", flush=True)

    kept = solids if not args.top else solids[: args.top]
    kept = [(s, v) for s, v in kept if v >= args.min_volume]
    print(f"kept: {len(kept)}", flush=True)

    import trimesh
    import numpy as np
    import trimesh.transformations as T

    writer = StlAPI_Writer()
    os.makedirs(args.chunk_dir, exist_ok=True)
    chunk_meta = []
    for i, (s, vol) in enumerate(kept):
        BRepMesh_IncrementalMesh(s, args.deflection, False, 0.5, True)
        one = os.path.join(args.chunk_dir, f"part_{i:02d}.stl")
        writer.Write(s, one)
        m = trimesh.load(one)
        if len(m.faces) == 0 or len(m.vertices) == 0:
            os.remove(one)
            continue
        m.apply_scale(0.001)  # mm -> m
        chunk_meta.append((one, m))
        print(f"  part_{i:02d}: v={vol:.0f} faces={len(m.faces)}", flush=True)

    if not chunk_meta:
        sys.exit("no chunks produced")

    allv = np.vstack([m.vertices for _, m in chunk_meta])
    lo, hi = allv.min(0), allv.max(0)
    thin = int(np.argmin(hi - lo))
    print("extents(m):", np.round(hi - lo, 1), "thin axis:", thin, flush=True)
    rot = np.eye(4)
    if thin == 1:
        rot = T.rotation_matrix(np.pi / 2, [1, 0, 0])
    elif thin == 0:
        rot = T.rotation_matrix(np.pi / 2, [0, 1, 0])

    finals = []
    for one, m in chunk_meta:
        m.apply_transform(rot)
        finals.append((one, m))
    allv = np.vstack([m.vertices for _, m in finals])
    zfloor = float(allv[:, 2].min())
    cxy = (allv[:, :2].min(0) + allv[:, :2].max(0)) / 2

    index_lines = []
    total_faces = 0
    for one, m in finals:
        m.apply_translation([-cxy[0], -cxy[1], -zfloor])
        pieces = []
        if len(m.faces) <= args.max_faces:
            pieces = [m]
        else:
            parts = m.split(only_watertight=False)
            parts.sort(key=lambda x: -len(x.faces))
            buf, buf_faces = [], 0
            for part in parts:
                buf.append(part)
                buf_faces += len(part.faces)
                if buf_faces >= args.max_faces:
                    pieces.append(trimesh.util.concatenate(buf))
                    buf, buf_faces = [], 0
            if buf:
                pieces.append(trimesh.util.concatenate(buf))
        base = os.path.splitext(os.path.basename(one))[0]
        os.remove(one)
        for pi, piece in enumerate(pieces):
            name = f"{base}_{pi}.stl" if len(pieces) > 1 else f"{base}.stl"
            piece.export(os.path.join(args.chunk_dir, name))
            index_lines.append(name)
            total_faces += len(piece.faces)
            print(f"  -> {name}: {len(piece.faces)} faces", flush=True)

    open(os.path.join(args.chunk_dir, "index.txt"), "w").write("\n".join(index_lines))
    print(f"DONE chunks={len(index_lines)} total_faces={total_faces}")


if __name__ == "__main__":
    main()
