"""The component: a CAD mesh in millimetres, split into named surface regions.

By default a procedural bracket (plate, flange, ribs, boss, pocket, holes) stands in for an aerospace part. It is
deliberately asymmetric, has large plain surfaces and repeated features, which is what makes localisation hard.
Any STL/OBJ/PLY file can be used instead (config: part.mesh).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon
from sklearn.cluster import KMeans


@dataclass
class Part:
    mesh: trimesh.Trimesh
    face_region: np.ndarray          # region id per face
    region_centers: np.ndarray       # (R, 3)

    @property
    def n_regions(self) -> int:
        return len(self.region_centers)

    @property
    def size_mm(self) -> float:
        return float(np.linalg.norm(self.mesh.extents))


def _box(x0, y0, z0, x1, y1, z1) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


def _cyl(x, y, z0, z1, r, sections=48) -> trimesh.Trimesh:
    c = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=sections)
    c.apply_translation([x, y, (z0 + z1) / 2])
    return c


def bracket() -> trimesh.Trimesh:
    """An asymmetric machined bracket, about 200 x 120 x 70 mm."""
    outline = Polygon([(0, 0), (200, 0), (200, 85), (170, 120), (0, 120), (0, 60), (-20, 45), (-20, 15), (0, 15)])
    base = trimesh.creation.extrude_polygon(outline, 10)
    flange = _box(0, 0, 10, 200, 12, 70)
    rib1 = _rib(40)
    rib2 = _rib(128, height=40, depth=35)
    boss = _cyl(150, 80, 10, 38, 16)
    solid = trimesh.boolean.union([base, flange, rib1, rib2, boss], engine="manifold")
    cuts = [_cyl(150, 80, 0, 40, 6),                                   # boss bore
            _box(45, 55, 5, 100, 90, 11),                              # pocket
            _cyl(20, 100, -1, 11, 5), _cyl(185, 25, -1, 11, 5), _cyl(110, 105, -1, 11, 4), _cyl(-8, 30, -1, 11, 4),
            _box(70, -1, 30, 100, 13, 55),                             # flange window
            _flange_hole(150, 45, 9), _flange_hole(175, 30, 5), _flange_hole(20, 50, 6)]
    for c in cuts:
        solid = trimesh.boolean.difference([solid, c], engine="manifold")
    solid.merge_vertices()
    return solid


def _rib(x: float, height: float = 55, depth: float = 50, t: float = 8) -> trimesh.Trimesh:
    """Triangular rib between plate (z=10) and flange (y=12), centred at x."""
    tri = Polygon([(12, 10), (12 + depth, 10), (12, 10 + height)])          # (y, z)
    m = trimesh.creation.extrude_polygon(tri, t)                           # extruded along +z (= x here)
    # map local (a, b, c) -> world (x = c + x - t/2, y = a, z = b)
    T = np.array([[0, 0, 1, x - t / 2], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    m.apply_transform(T)
    if m.volume < 0:
        m.invert()
    return m


def _flange_hole(x: float, z: float, r: float) -> trimesh.Trimesh:
    c = trimesh.creation.cylinder(radius=r, height=20, sections=40)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    c.apply_translation([x, 6, z])
    return c


def load_part(mesh_path: str | None, n_regions: int, seed: int = 0) -> Part:
    if mesh_path:
        mesh = trimesh.load(Path(mesh_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"{mesh_path} is not a single triangle mesh")
    else:
        mesh = bracket()
    mesh.apply_translation(-mesh.bounds.mean(axis=0))      # centre the part at the origin
    # Regions: k-means on surface samples (position + scaled normal), so a region is one connected face area.
    pts, fid = trimesh.sample.sample_surface_even(mesh, 6000, seed=seed)
    feat = np.hstack([pts, mesh.face_normals[fid] * 0.15 * float(np.linalg.norm(mesh.extents))])
    km = KMeans(n_clusters=n_regions, n_init=10, random_state=seed).fit(feat)  # pyright: ignore[reportArgumentType]
    face_feat = np.hstack([mesh.triangles_center, mesh.face_normals * 0.15 * float(np.linalg.norm(mesh.extents))])
    face_region = np.asarray(km.predict(face_feat), dtype=int)
    centers = np.array([mesh.triangles_center[face_region == r].mean(axis=0) if np.any(face_region == r)
                        else km.cluster_centers_[r, :3] for r in range(n_regions)])
    return Part(mesh=mesh, face_region=face_region, region_centers=centers)
