"""Synthetic captures of the part: grey-scale image plus, for every pixel, the true 3D point on the CAD surface.

Camera convention (OpenCV): x right, y down, z forward. A pose is (R, t) with x_cam = R @ x_part + t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .part import Part


@dataclass
class Camera:
    width: int = 192
    height: int = 192
    fov_deg: float = 50.0

    @property
    def K(self) -> np.ndarray:
        f = 0.5 * self.width / np.tan(np.radians(self.fov_deg) / 2)
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1]], dtype=float)


@dataclass
class Capture:
    image: np.ndarray            # (H, W) uint8
    xyz: np.ndarray              # (H, W, 3) float32, part frame, NaN on background
    region: np.ndarray           # (H, W) int, -1 on background
    R: np.ndarray                # (3, 3) part -> camera
    t: np.ndarray                # (3,)
    target: np.ndarray           # true 3D point at the image centre (the "data point")
    target_region: int
    meta: dict = field(default_factory=dict)

    @property
    def mask(self) -> np.ndarray:
        return self.region >= 0


class Renderer:
    def __init__(self, part: Part, camera: Camera):
        self.part, self.cam = part, camera
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor.from_numpy(np.ascontiguousarray(part.mesh.vertices, np.float32)),
                                 o3d.core.Tensor.from_numpy(np.ascontiguousarray(part.mesh.faces, np.uint32)))
        K = camera.K
        u, v = np.meshgrid(np.arange(camera.width) + 0.5, np.arange(camera.height) + 0.5)
        self._dirs_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)

    def raycast(self, R: np.ndarray, t: np.ndarray):
        c = -R.T @ t
        d = self._dirs_cam.reshape(-1, 3) @ R          # = (R.T @ d_cam.T).T
        rays = np.hstack([np.broadcast_to(c, d.shape), d]).astype(np.float32)
        ans = self.scene.cast_rays(o3d.core.Tensor.from_numpy(np.ascontiguousarray(rays)))
        thit = ans["t_hit"].numpy()
        prim = ans["primitive_ids"].numpy().astype(np.int64)
        hit = np.isfinite(thit)
        pts = c + d * np.where(hit, thit, np.nan)[:, None]
        return pts, prim, hit, d

    def render(self, R, t, light_dir_cam, rng: np.random.Generator, noise: float = 0.02,
               ambient: float = 0.18, specular: float = 0.35, background: float = 0.06,
               texture: float = 0.0) -> Capture:
        cam = self.cam
        pts, prim, hit, d = self.raycast(R, t)
        n = self.part.mesh.face_normals[np.where(hit, prim, 0)]
        view = -d / np.linalg.norm(d, axis=1, keepdims=True)
        n = np.where((np.sum(n * view, 1) < 0)[:, None], -n, n)     # face the camera
        light = R.T @ (light_dir_cam / np.linalg.norm(light_dir_cam))
        diff = np.clip(n @ light, 0, None)
        h = light + view
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        spec = np.clip(np.sum(n * h, 1), 0, None) ** 30
        # Real parts have surface texture (machining marks, AM roughness) that the CAD model does not have.
        # It is random per capture, so it cannot be matched against the CAD renders: a nuisance, as in reality.
        tex = 0.0
        if texture > 0:
            freq = rng.normal(0, 0.35, (16, 3))
            phase = rng.uniform(0, 2 * np.pi, 16)
            tex = texture * np.sin(np.nan_to_num(pts) @ freq.T + phase).sum(1) / 4
        val = np.where(hit, ambient + 0.65 * diff + specular * spec + tex, background)
        img = val.reshape(cam.height, cam.width) + rng.normal(0, noise, (cam.height, cam.width))
        img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        region = np.where(hit, self.part.face_region[np.where(hit, prim, 0)], -1).reshape(cam.height, cam.width).astype(np.int16)
        xyz = np.where(hit[:, None], pts, np.nan).reshape(cam.height, cam.width, 3).astype(np.float32)
        cy, cx = cam.height // 2, cam.width // 2
        return Capture(image=img8, xyz=xyz, region=region, R=R, t=t, target=xyz[cy, cx].astype(float),
                       target_region=int(region[cy, cx]))


UP = np.array([0.0, 0.0, 1.0])       # part frame "up" (the part sits in a known fixture orientation)


def look_at(eye: np.ndarray, target: np.ndarray, rng: np.random.Generator,
            max_roll_deg: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Camera at eye looking at target. Roll is free (None) or within max_roll_deg of upright, as when the camera's
    gravity sensor tells which way is up."""
    z = target - eye
    z /= np.linalg.norm(z)
    if max_roll_deg is None:
        a = rng.normal(size=3)
    else:
        a = -UP if abs(z.dot(UP)) < 0.95 else np.array([0.0, -1.0, 0.0])   # image y axis points down
        a = a - a.dot(z) * z
        a /= np.linalg.norm(a)
        roll = np.radians(rng.uniform(-max_roll_deg, max_roll_deg))
        a = np.cos(roll) * a + np.sin(roll) * np.cross(z, a)
    a -= a.dot(z) * z
    x = np.cross(a / np.linalg.norm(a), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])          # rows: camera axes in the part frame
    return R, -R @ eye


def sample_capture(renderer: Renderer, rng: np.random.Generator, dist_mm: tuple[float, float],
                   max_tilt_deg: float = 60, light: str = "camera", max_roll_deg: float | None = None,
                   tries: int = 50, **render_kw) -> Capture:
    """A capture of a random spot on the surface, as an inspector or a scanner would take it."""
    mesh = renderer.part.mesh
    area_p = mesh.area_faces / mesh.area_faces.sum()
    for _ in range(tries):
        fid = int(rng.choice(len(mesh.faces), p=area_p))          # area-weighted, from the seeded generator
        a_, b_ = rng.random(2)
        if a_ + b_ > 1:
            a_, b_ = 1 - a_, 1 - b_
        tri = mesh.triangles[fid]
        p = tri[0] + a_ * (tri[1] - tri[0]) + b_ * (tri[2] - tri[0])
        n = mesh.face_normals[fid]
        tilt = np.radians(rng.uniform(0, max_tilt_deg))
        a = rng.normal(size=3)
        a -= a.dot(n) * n
        a /= np.linalg.norm(a)
        direction = np.cos(tilt) * n + np.sin(tilt) * a
        eye = p + direction * rng.uniform(*dist_mm)
        R, t = look_at(eye, p, rng, max_roll_deg)
        if light == "camera":            # light mounted near the camera (ring light / flash)
            ldir = np.array([rng.normal(0, 0.15), rng.normal(0, 0.15), -1.0])
        else:                             # any direction from the camera side
            ldir = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), -rng.uniform(0.2, 1)])
        cap = renderer.render(R, t, ldir, rng, **render_kw)
        # keep captures where the centre ray hits the chosen spot (not hidden) and enough of the part is visible
        if cap.target_region >= 0 and np.linalg.norm(cap.target - p) < 1.0 and cap.mask.mean() > 0.3:
            cap.meta = {"distance_mm": float(np.linalg.norm(eye - p)), "tilt_deg": float(np.degrees(tilt)),
                        "light": light}
            return cap
    raise RuntimeError("could not find a visible capture; check the mesh and distances")
