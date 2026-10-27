"""DXF unit-cell pattern -> metal polygons in millimetres."""

from __future__ import annotations

from pathlib import Path

import ezdxf
from shapely.geometry import Polygon
from shapely.ops import unary_union


def load_pattern(dxf: Path, period_mm: float = 10.0, simplify_mm: float = 0.01) -> list[Polygon]:
    """Metal polygons of a DXF unit cell.

    The first closed polyline spanning the drawing extents is the cell boundary;
    every other closed polyline is metal. Coordinates are scaled so the extents map
    onto ``period_mm`` (500 units -> 10 mm). Bulges (arcs) are replaced by chords.
    """
    doc = ezdxf.readfile(str(dxf))
    rings = []
    for e in doc.modelspace():
        if e.dxftype() == "POLYLINE" and e.is_closed:
            rings.append([(v.dxf.location.x, v.dxf.location.y) for v in e.vertices])
        elif e.dxftype() == "LWPOLYLINE" and e.closed:
            rings.append([(x, y) for x, y in e.get_points("xy")])
    if not rings:
        raise ValueError(f"{dxf}: no closed polylines")
    xs = [x for r in rings for x, _ in r]
    ys = [y for r in rings for _, y in r]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    scale = period_mm / (x1 - x0)
    if abs((y1 - y0) * scale - period_mm) > 1e-6 * period_mm:
        raise ValueError(f"{dxf}: drawing extents are not square")

    polys = []
    for ring in rings:
        rx = [x for x, _ in ring]
        ry = [y for _, y in ring]
        if min(rx) == x0 and max(rx) == x1 and min(ry) == y0 and max(ry) == y1:
            continue  # cell boundary
        p = Polygon([((x - x0) * scale, (y - y0) * scale) for x, y in ring])
        p = p.buffer(0).simplify(simplify_mm, preserve_topology=True)
        if not p.is_empty:
            polys.append(p)
    metal = unary_union(polys)
    parts = list(metal.geoms) if hasattr(metal, "geoms") else [metal]
    return [p for p in parts if p.area > 1e-6]


def metal_fraction(polys: list[Polygon], period_mm: float = 10.0) -> float:
    return sum(p.area for p in polys) / period_mm**2


def shift_pattern(polys: list[Polygon], dx: float, dy: float, period_mm: float = 10.0) -> list[Polygon]:
    """Translate the pattern by (dx, dy) and wrap it back into the cell (periodic copy)."""
    from shapely.affinity import translate
    from shapely.geometry import box

    cell = box(0, 0, period_mm, period_mm)
    out = []
    for p in polys:
        q = translate(p, dx, dy)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                piece = translate(q, i * period_mm, j * period_mm).intersection(cell)
                if not piece.is_empty and piece.area > 1e-9:
                    out.append(piece)
    merged = unary_union(out)
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    return [p for p in parts if p.area > 1e-9]
