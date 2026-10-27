"""Periodic layered unit-cell mesh with gmsh (lengths in millimetres).

Layers along z: substrate [-h, 0] (PEC ground at -h), optional superstrate [0, h],
air [z_top, z_port]. The metal pattern is a set of plane faces embedded in z = 0, so the
tetrahedral mesh conforms to it. Side faces are periodic (gmsh ``setPeriodic``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gmsh
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .designs import Design
from .geometry import load_pattern, shift_pattern

REGION_SUB, REGION_SUP, REGION_AIR = 0, 1, 2

# Mesh-size presets (mm), tuned for order-2 elements (the default): h_edge at the metal
# edges, grown over ``grow`` mm to the layer cap h_diel (dielectrics) or h_air (air).
# h_air must stay below a/3 so no element spans half a period (see _check).
PRESETS = {
    "coarse": dict(h_edge=0.30, h_diel=0.60, h_air=2.0, grow=1.0),
    "medium": dict(h_edge=0.10, h_diel=0.60, h_air=2.0, grow=0.5),
    "fine": dict(h_edge=0.07, h_diel=0.40, h_air=2.0, grow=0.5),
}


@dataclass
class Mesh:
    nodes: np.ndarray  # (N, 3) mm
    tets: np.ndarray  # (T, 4) node indices
    tet_region: np.ndarray  # (T,) REGION_*
    tris: dict[str, np.ndarray]  # 'metal', 'port', 'ground', 'sheet' (all of z=0 interior)
    periodic: np.ndarray  # (N,) node -> its master node (identity if not a slave)
    z_port: float
    z_top: float
    info: dict = field(default_factory=dict)


def _add_polygon(occ, poly: Polygon) -> int:
    def loop(coords):
        coords = list(coords)[:-1]
        pts = [occ.addPoint(x, y, 0.0) for x, y in coords]
        lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        return occ.addCurveLoop(lines)

    loops = [loop(poly.exterior.coords)] + [loop(r.coords) for r in poly.interiors]
    return occ.addPlaneSurface(loops)


def build_mesh(
    design: Design,
    preset: str = "coarse",
    air_mm: float = 10.0,
    verbose: bool = False,
    shift: tuple[float, float] | None = None,
    **overrides,
) -> Mesh:
    p = dict(PRESETS[preset])
    p.update(overrides)
    a = design.period_mm
    h = design.h_mm
    z_top = design.z_top_mm
    z_port = z_top + air_mm
    polys = load_pattern(design.dxf, a) if design.dxf is not None else []
    if shift is not None and polys:
        polys = shift_pattern(polys, shift[0], shift[1], a)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("cell")
        occ = gmsh.model.occ
        vols = [(3, occ.addBox(0, 0, -h, a, a, h))]
        if design.superstrate:
            vols.append((3, occ.addBox(0, 0, 0, a, a, h)))
        vols.append((3, occ.addBox(0, 0, z_top, a, a, air_mm)))
        surfs = [(2, _add_polygon(occ, q)) for q in polys]
        occ.fragment(vols, surfs)
        occ.synchronize()
        model = gmsh.model
        tol = 1e-6

        def com(dim, tag):
            return np.array(occ.getCenterOfMass(dim, tag))

        def bbox(dim, tag):
            return np.array(model.getBoundingBox(dim, tag))

        # Volumes -> regions.
        region_of_vol = {}
        for _, v in model.getEntities(3):
            zc = com(3, v)[2]
            region_of_vol[v] = REGION_SUB if zc < 0 else (REGION_AIR if zc > z_top else REGION_SUP)

        # Faces: z=0 sheet plane, port (z=z_port), ground (z=-h). Metal triangles of
        # the sheet plane are picked after meshing by their centroids.
        sheet_faces, port_faces, ground_faces = [], [], []
        for _, s in model.getEntities(2):
            b = bbox(2, s)
            if abs(b[2]) < tol and abs(b[5]) < tol:
                sheet_faces.append(s)
            elif abs(b[2] - z_port) < tol and abs(b[5] - z_port) < tol:
                port_faces.append(s)
            elif abs(b[2] + h) < tol and abs(b[5] + h) < tol:
                ground_faces.append(s)

        # Periodic side faces: slave at x=a (y=a) copies master at x=0 (y=0).
        def side_faces(axis, value):
            out = []
            for _, s in model.getEntities(2):
                b = bbox(2, s)
                if abs(b[axis] - value) < tol and abs(b[axis + 3] - value) < tol:
                    out.append(s)
            return out

        for axis in (0, 1):
            tr = [0.0, 0.0, 0.0]
            tr[axis] = a
            affine = [1, 0, 0, tr[0], 0, 1, 0, tr[1], 0, 0, 1, tr[2], 0, 0, 0, 1]
            masters = side_faces(axis, 0.0)
            for s in side_faces(axis, a):
                bs = bbox(2, s)
                match = [m for m in masters if np.allclose(bbox(2, m) + np.r_[tr, tr], bs, atol=1e-5)]
                if len(match) != 1:
                    raise RuntimeError(f"no unique periodic master for face {s}")
                model.mesh.setPeriodic(2, [s], [match[0]], affine)

        # Size field: fine at the pattern edges, capped per layer.
        fields = []
        metal_curves = []
        for _, c in model.getEntities(1):
            b = bbox(1, c)
            on_side = any(abs(b[i] - v) < tol and abs(b[i + 3] - v) < tol for i in (0, 1) for v in (0.0, a))
            if abs(b[2]) < tol and abs(b[5]) < tol and not on_side:
                metal_curves.append(c)
        if metal_curves:
            fd = model.mesh.field.add("Distance")
            model.mesh.field.setNumbers(fd, "CurvesList", metal_curves)
            model.mesh.field.setNumber(fd, "Sampling", 200)
            ft = model.mesh.field.add("Threshold")
            model.mesh.field.setNumber(ft, "InField", fd)
            model.mesh.field.setNumber(ft, "SizeMin", p["h_edge"])
            model.mesh.field.setNumber(ft, "SizeMax", p["h_air"])
            model.mesh.field.setNumber(ft, "DistMin", 0.0)
            model.mesh.field.setNumber(ft, "DistMax", p["grow"])
            fields.append(ft)
        for v, reg in region_of_vol.items():
            fc = model.mesh.field.add("Constant")
            model.mesh.field.setNumber(fc, "VIn", p["h_air"] if reg == REGION_AIR else p["h_diel"])
            model.mesh.field.setNumbers(fc, "VolumesList", [v])
            fields.append(fc)
        fmin = model.mesh.field.add("Min")
        model.mesh.field.setNumbers(fmin, "FieldsList", fields)
        model.mesh.field.setAsBackgroundMesh(fmin)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        model.mesh.generate(3)

        tags, coords, _ = model.mesh.getNodes()
        nodes = coords.reshape(-1, 3)
        index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
        index[tags.astype(np.int64)] = np.arange(tags.size)

        tets, regs = [], []
        for v, reg in region_of_vol.items():
            types, _, conn = model.mesh.getElements(3, v)
            for t, c in zip(types, conn):
                if t != 4:
                    raise RuntimeError(f"unexpected 3-D element type {t}")
                c = index[c.astype(np.int64)].reshape(-1, 4)
                tets.append(c)
                regs.append(np.full(len(c), reg, dtype=np.int8))

        def tris_of(faces):
            out = [np.zeros((0, 3), dtype=np.int64)]
            for s in faces:
                types, _, conn = model.mesh.getElements(2, s)
                for t, c in zip(types, conn):
                    if t != 2:
                        raise RuntimeError(f"unexpected 2-D element type {t}")
                    out.append(index[c.astype(np.int64)].reshape(-1, 3))
            return np.concatenate(out)

        # Periodic node map, composed so corner/edge nodes reach the final master.
        master = np.arange(len(nodes))
        for dim in (0, 1, 2):
            for _, e in model.getEntities(dim):
                tag_m, nt, ntm, _ = model.mesh.getPeriodicNodes(dim, e)
                if tag_m != e and len(nt):
                    master[index[np.asarray(nt, dtype=np.int64)]] = index[np.asarray(ntm, dtype=np.int64)]
        for _ in range(4):
            master = master[master]

        sheet = tris_of(sheet_faces)
        cen = nodes[sheet].mean(axis=1)
        metal = np.zeros(len(sheet), dtype=bool)
        if polys:
            region = unary_union(polys)
            metal = np.array([region.contains(Point(x, y)) for x, y in cen[:, :2]], dtype=bool)

        mesh = Mesh(
            nodes=nodes,
            tets=np.concatenate(tets),
            tet_region=np.concatenate(regs),
            tris={
                "metal": sheet[metal],
                "sheet": sheet,
                "port": tris_of(port_faces),
                "ground": tris_of(ground_faces),
            },
            periodic=master,
            z_port=z_port,
            z_top=z_top,
            info=dict(preset=preset, air_mm=air_mm, shift=shift, **p),
        )
    finally:
        gmsh.finalize()
    _check(mesh, a)
    return mesh


def _check(mesh: Mesh, a: float) -> None:
    x = mesh.nodes
    slaves = np.flatnonzero(mesh.periodic != np.arange(len(x)))
    d = x[slaves] - x[mesh.periodic[slaves]]
    d[:, :2] = np.mod(d[:, :2] + 0.5 * a, a) - 0.5 * a
    if np.abs(d).max(initial=0) > 1e-6:
        raise RuntimeError("periodic node map is not a lattice translation")
    # Periodic merging identifies edges by their merged end nodes. That is only unique when
    # no edge spans half a period or more in x or y (else two different physical edges
    # collapse onto one key). Coarse air meshes (h > a/3) can violate it.
    ext = np.ptp(x[mesh.tets][:, :, :2], axis=1).max(axis=0)
    if np.any(ext >= 0.5 * a - 1e-9):
        raise RuntimeError(f"element extent {ext} >= a/2: mesh too coarse for periodic merging")
    on_side = (np.abs(x[:, 0] - a) < 1e-9) | (np.abs(x[:, 1] - a) < 1e-9)
    if np.any(mesh.periodic[on_side] == np.flatnonzero(on_side)):
        raise RuntimeError("some nodes on x=a or y=a have no periodic master")
