"""ML: a small CNN predicts, for every image cell, which 3D point of the CAD model it shows (scene coordinates),
plus whether the cell shows the part at all. The camera pose then follows from PnP + RANSAC.

Coordinates are predicted in two steps (hierarchical scene coordinates, after Li et al., CVPR 2020): first which of
K surface patches the cell shows (classification), then the offset inside that patch (regression). This learns much
faster than regressing x, y, z directly, because one image cell can look like several places on the part.

It is trained only on renders of the CAD model. To bridge the gap to real captures, the training renders get random
surface texture, lighting, noise, blur, brightness and occluders (domain randomisation).
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import trimesh
from sklearn.cluster import KMeans
from torch import nn

from .methods import Estimate, _pnp
from .part import Part
from .render import Camera, Capture, Renderer, sample_capture

IN = 128          # network input size
OUT = 32          # output grid (one 3D point per 4 x 4 input pixels)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SCRNet(nn.Module):
    """Encoder down to 1/16 (each cell sees the whole image), decoder back to 1/4 with skip connections."""

    def __init__(self, w: int = 24, n_out: int = 4):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4, self.e5 = (_block(1, w), _block(w, 2 * w), _block(2 * w, 4 * w),
                                                       _block(4 * w, 8 * w), _block(8 * w, 8 * w))
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d4, self.d3 = _block(16 * w, 4 * w), _block(8 * w, 4 * w)
        self.head = nn.Conv2d(4 * w, n_out, 1)      # patch logits, offset x, y, z, part/background logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)                      # 128
        e2 = self.e2(self.pool(e1))          # 64
        e3 = self.e3(self.pool(e2))          # 32
        e4 = self.e4(self.pool(e3))          # 16
        e5 = self.e5(self.pool(e4))          # 8
        d4 = self.d4(torch.cat([self.up(e5), e4], 1))     # 16
        d3 = self.d3(torch.cat([self.up(d4), e3], 1))     # 32
        return self.head(d3)


def _prep(img: np.ndarray) -> np.ndarray:
    x = cv2.resize(img, (IN, IN), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return (x - x.mean()) / (x.std() + 1e-3)


def _augment(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random exposure, blur, noise and clutter. Returns the image and where clutter now hides the part."""
    x = img.astype(np.float32) / 255
    hidden = np.zeros(x.shape, dtype=bool)
    x = np.clip(x * rng.uniform(0.6, 1.4) + rng.uniform(-0.08, 0.08), 0, 1) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.5:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.8))
    x = x + rng.normal(0, rng.uniform(0, 0.06), x.shape)
    if rng.random() < 0.6:                                   # clutter in front of the part
        h, w = x.shape
        for _ in range(rng.integers(1, 4)):
            bw, bh = int(rng.uniform(0.2, 0.5) * w), int(rng.uniform(0.2, 0.5) * h)
            x0, y0 = rng.integers(0, w - bw), rng.integers(0, h - bh)
            if abs(x0 + bw / 2 - w / 2) < bw / 2 and abs(y0 + bh / 2 - h / 2) < bh / 2:
                continue                                     # keep the image centre visible
            ramp = np.linspace(0, 1, bw)[None, :] * rng.uniform(-0.3, 0.3) + rng.uniform(0.1, 0.8)
            x[y0:y0 + bh, x0:x0 + bw] = ramp + rng.normal(0, rng.uniform(0.02, 0.1), (bh, bw))
            hidden[y0:y0 + bh, x0:x0 + bw] = True
    return (np.clip(x, 0, 1) * 255).astype(np.uint8), hidden


def training_renders(renderer: Renderer, n: int, seed: int, dist=(25, 200), max_roll_deg: float | None = 20) -> list[Capture]:
    """CAD renders with randomised nuisances (texture, light, specular, ambient) for training the network."""
    rng = np.random.default_rng(seed)
    caps = []
    for i in range(n):
        c = sample_capture(renderer, rng, dist, light="camera" if i % 3 == 0 else "any",
                           max_roll_deg=max_roll_deg, texture=rng.uniform(0, 0.08),
                           ambient=rng.uniform(0.1, 0.25), specular=rng.uniform(0.1, 0.7))
        step = c.xyz.shape[0] // OUT                       # keep only the output grid: saves memory
        c.xyz = np.ascontiguousarray(c.xyz[step // 2::step, step // 2::step][:OUT, :OUT])
        c.region = np.ascontiguousarray(c.region[step // 2::step, step // 2::step][:OUT, :OUT])
        caps.append(c)
    return caps


def _cells(hidden: np.ndarray) -> np.ndarray:
    step = hidden.shape[0] // OUT
    return hidden[step // 2::step, step // 2::step][:OUT, :OUT].astype(np.float32)


def surface_patches(part: Part, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K patch centres evenly over the surface, and each patch's radius (for normalising the offsets)."""
    pts, _ = trimesh.sample.sample_surface_even(part.mesh, 20 * k, seed=seed)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(pts)  # pyright: ignore[reportArgumentType]
    d = np.linalg.norm(pts - km.cluster_centers_[km.labels_], axis=1)
    radius = np.array([d[km.labels_ == i].max() if (km.labels_ == i).any() else 1.0 for i in range(k)])
    return km.cluster_centers_, np.maximum(radius, 1.0)


def _targets(cap: Capture, centers: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = cap.xyz.shape[0]
    step = h // OUT
    xyz = cap.xyz[step // 2::step, step // 2::step][:OUT, :OUT].reshape(-1, 3)
    mask = np.isfinite(xyz).all(-1)
    cls = np.zeros(len(xyz), dtype=np.int64)
    off = np.zeros((len(xyz), 3), dtype=np.float32)
    if mask.any():
        d = np.linalg.norm(xyz[mask, None] - centers[None], axis=2)
        cls[mask] = d.argmin(1)
        off[mask] = (xyz[mask] - centers[cls[mask]]) / radius[cls[mask], None]
    return (cls.reshape(OUT, OUT), off.reshape(OUT, OUT, 3).transpose(2, 0, 1),
            mask.reshape(OUT, OUT).astype(np.float32))


class CNN:
    name = "cnn"

    def __init__(self, camera: Camera, part: Part, width: int = 24, patches: int = 64, device: str | None = None):
        self.cam, self.width = camera, width
        self.centers, self.radius = surface_patches(part, patches)
        self.k = patches
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SCRNet(width, patches + 4).to(self.device)
        step = camera.width / OUT
        g = (np.arange(OUT) + 0.5) * step
        self.grid_uv = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)      # pixel centre of each output cell
        self.history: list[dict] = []

    def fit(self, caps: list[Capture], epochs: int = 20, batch: int = 32, lr: float = 2e-3, seed: int = 0,
            log=print) -> "CNN":
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        tgt = [_targets(c, self.centers, self.radius) for c in caps]
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        steps = epochs * int(np.ceil(len(caps) / batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        self.net.train()
        for ep in range(epochs):
            t0, tot, n = time.time(), 0.0, 0
            for b in np.array_split(rng.permutation(len(caps)), int(np.ceil(len(caps) / batch))):
                aug = [_augment(caps[i].image, rng) for i in b]
                x = torch.from_numpy(np.stack([_prep(a[0]) for a in aug])[:, None]).to(self.device)
                cls = torch.from_numpy(np.stack([tgt[i][0] for i in b])).to(self.device)
                off = torch.from_numpy(np.stack([tgt[i][1] for i in b])).to(self.device)
                visible = [tgt[i][2] * (1 - _cells(a[1])) for i, a in zip(b, aug)]   # hidden cells are not part
                m = torch.from_numpy(np.stack(visible).astype(np.float32)).to(self.device)
                out = self.net(x)
                k, msum = self.k, m.sum().clamp(min=1)
                l_cls = (nn.functional.cross_entropy(out[:, :k], cls, reduction="none") * m).sum() / msum
                l_off = ((out[:, k:k + 3] - off).abs().sum(1) * m).sum() / msum
                l_mask = nn.functional.binary_cross_entropy_with_logits(out[:, k + 3], m)
                loss = l_cls + l_off + 0.5 * l_mask
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item() * len(b)
                n += len(b)
            self.history.append({"epoch": ep + 1, "loss": tot / n, "seconds": time.time() - t0})
            log(f"  epoch {ep + 1:2d}/{epochs}  loss {tot / n:.3f}  ({time.time() - t0:.0f} s)")
        self.net.eval()
        return self

    @torch.no_grad()
    def correspondences(self, img: np.ndarray, top: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """2D-3D pairs for every part cell; the `top` most likely patches each give a candidate for RANSAC."""
        x = torch.from_numpy(_prep(img)[None, None]).to(self.device)
        out = self.net(x)[0].cpu().numpy()
        k = self.k
        fg = out[k + 3].reshape(-1) > 0
        logits = out[:k].reshape(k, -1).T[fg]
        off = out[k:k + 3].reshape(3, -1).T[fg]
        uv = self.grid_uv[fg]
        best = np.argsort(-logits, axis=1)[:, :top]
        pts2, pts3 = [], []
        for j in range(best.shape[1]):
            c = best[:, j]
            pts2.append(uv)
            pts3.append(self.centers[c] + off * self.radius[c, None])
        return np.vstack(pts2), np.vstack(pts3)

    def locate(self, img: np.ndarray) -> Estimate:
        return _pnp(*self.correspondences(img), self.cam.K, reproj_px=6.0, iters=1000)

    def save(self, path) -> None:
        torch.save({"state": self.net.state_dict(), "centers": self.centers, "radius": self.radius,
                    "width": self.width, "history": self.history}, path)

    def load(self, path) -> "CNN":
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.centers, self.radius, self.k = d["centers"], d["radius"], len(d["centers"])
        self.net = SCRNet(d["width"], self.k + 4).to(self.device)
        self.net.load_state_dict(d["state"])
        self.history = d.get("history", [])
        self.net.eval()
        return self
