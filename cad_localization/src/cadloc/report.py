"""One self-contained HTML dashboard: how well each method places captured data on the CAD model."""
from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from .part import Part
from .render import Capture, Renderer

CSS = """
body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f5f6f8;color:#1c2230}
header{background:#14365d;color:#fff;padding:18px 28px}header h1{margin:0;font-size:22px}
header p{margin:4px 0 0;opacity:.85;font-size:13px}main{max-width:1200px;margin:0 auto;padding:18px}
section{background:#fff;border:1px solid #dde2ea;border-radius:8px;padding:16px 18px;margin-bottom:16px}
h2{font-size:17px;margin:0 0 8px}p.note{font-size:13px;color:#556;margin:4px 0 10px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.kpi{background:#eef2f8;border-radius:6px;padding:10px 12px}.kpi b{display:block;font-size:22px}.kpi span{font-size:12px;color:#556}
table{border-collapse:collapse;font-size:13px;width:100%}th,td{border-bottom:1px solid #e3e7ee;padding:5px 8px;text-align:left}
th{background:#eef2f8}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.gallery figure{margin:0;font-size:12px}.gallery img{width:100%;border-radius:4px;image-rendering:pixelated}
"""
METHOD_NAMES = {"retrieval": "image retrieval (HOG)", "features": "SIFT features + PnP",
                "forest": "pixel forest + PnP", "cnn": "CNN scene coordinates + PnP"}
COLORS = {"retrieval": "#8c96a8", "features": "#e69f00", "forest": "#56b4e9", "cnn": "#c0392b"}


def _fig(fig: go.Figure, first: bool, height: int = 400) -> str:
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), template="plotly_white", height=height)
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if first else False)  # pyright: ignore[reportArgumentType]


def _png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def coverage_curve(df: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Accept only the most confident captures: how accurate are the accepted ones?"""
    rows = []
    for m, g in df.groupby("method"):
        g = g.sort_values("confidence", ascending=False)
        ok = (g["centre_error_mm"] <= tol).to_numpy()
        for k in range(1, len(g) + 1, max(1, len(g) // 40)):
            rows.append({"method": m, "accepted": k / len(g), "accurate": ok[:k].mean()})
    return pd.DataFrame(rows)


def build(part: Part, renderer: Renderer, df: pd.DataFrame, summ: pd.DataFrame, tests: dict[str, list[Capture]],
          info: dict, out: Path) -> Path:
    tol = info["tolerance_mm"]
    col = f"located_within_{tol:g}mm"
    methods = [m for m in METHOD_NAMES if m in set(df["method"])]
    nominal = cast(pd.DataFrame, summ[summ["condition"] == "nominal"])
    best = nominal.sort_values(by=col, ascending=False).iloc[0]
    overall = cast(pd.Series, df.assign(ok=df["centre_error_mm"] <= tol).groupby("method")["ok"].mean())
    kpis = [(f"{best[col]:.0%}", f"located within {tol:g} mm, nominal ({METHOD_NAMES[best['method']]})"),
            (f"{best['median_error_mm']:.1f} mm", "median error of the image centre, nominal"),
            (f"{float(overall.max()):.0%}", "located, all six conditions together (best method)"),
            (str(info["references"]), "CAD reference renders"),
            (str(info["captures_per_condition"] * len(tests)), "test captures")]
    parts = ["<section><div class='kpis'>" + "".join(f"<div class='kpi'><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>"
                                                     for v, k in kpis) + "</div></section>"]

    s = summ.assign(method=[METHOD_NAMES[str(m)] for m in summ["method"]])
    bar = px.bar(s, x="condition", y=col, color="method", barmode="group",
                 color_discrete_map={METHOD_NAMES[k]: v for k, v in COLORS.items()},
                 category_orders={"condition": list(tests)}, title=f"Share of captures located within {tol:g} mm")
    bar.update_yaxes(tickformat=".0%", range=[0, 1])
    parts.append(f"<section><h2>Accuracy and robustness</h2>{_fig(bar, True)}"
                 "<p class='note'>nominal: normal inspection photos. free_roll: camera held at any angle. close_up: "
                 "25-50 mm away, little context. lighting: unseen light directions, strong reflections. occlusion: "
                 "30-50 % of the image covered. noise_blur: sensor noise, blur and exposure changes. Every capture has "
                 "surface texture that the CAD model does not have.</p>"
                 + s.rename(columns={col: "located", "region_correct": "region correct"}).to_html(
                     index=False, border=0, float_format=lambda v: f"{v:.2f}") + "</section>")

    # 3D view: where the best method puts the captures, against the truth
    m = part.mesh
    mesh3d = go.Mesh3d(x=m.vertices[:, 0], y=m.vertices[:, 1], z=m.vertices[:, 2], i=m.faces[:, 0], j=m.faces[:, 1],
                       k=m.faces[:, 2], intensity=part.face_region, intensitymode="cell", colorscale="Turbo",
                       opacity=0.55, showscale=False, name="CAD regions", hoverinfo="skip")
    d = df[(df["method"] == best["method"]) & df["found"]]
    fig3 = go.Figure([mesh3d])
    fig3.add_trace(go.Scatter3d(x=d["true_x"], y=d["true_y"], z=d["true_z"], mode="markers", name="true location",
                                marker=dict(size=3, color="#14365d")))
    ok = d["centre_error_mm"] <= tol
    for flag, name, color in ((True, "located", "#1e9e5a"), (False, f"off by more than {tol:g} mm", "#c0392b")):
        e = d[ok == flag]
        fig3.add_trace(go.Scatter3d(x=e["est_x"], y=e["est_y"], z=e["est_z"], mode="markers", name=name,
                                    marker=dict(size=3, color=color),
                                    text=e["condition"] + ": " + e["centre_error_mm"].round(1).astype(str) + " mm",
                                    hoverinfo="text"))
    fig3.update_layout(scene=dict(aspectmode="data"), legend=dict(orientation="h"))
    parts.append(f"<section><h2>Captured data on the CAD model ({html.escape(METHOD_NAMES[best['method']])})</h2>"
                 f"{_fig(fig3, False, 560)}<p class='note'>Colours are the {part.n_regions} surface regions used for "
                 f"traceability. Drag to rotate.</p></section>")

    cov = coverage_curve(df, tol).assign(method=lambda x: x["method"].map(METHOD_NAMES))
    cf = px.line(cov, x="accepted", y="accurate", color="method",
                 color_discrete_map={METHOD_NAMES[k]: v for k, v in COLORS.items()},
                 title="Accept only confident results: accuracy of the accepted captures")
    cf.update_xaxes(tickformat=".0%", title="share of captures accepted (most confident first)")
    cf.update_yaxes(tickformat=".0%", range=[0, 1.02], title=f"accepted ones within {tol:g} mm")
    parts.append(f"<section><h2>Can the system tell when it is unsure?</h2>{_fig(cf, False)}<p class='note'>Confidence "
                 "is the number of RANSAC inliers. A flat curve near 100 % on the left means confident results can "
                 "be stored automatically and the rest sent to a person.</p></section>")

    figs = []
    for cond, caps in tests.items():
        cap = caps[0]
        rows = df[(df["condition"] == cond) & (df["capture"] == 0)].set_index("method")
        lines = "".join(f"<br>{html.escape(METHOD_NAMES[mm])}: " + (f"{rows.loc[mm, 'centre_error_mm']:.1f} mm"
                        if pd.notna(rows.loc[mm, "centre_error_mm"]) else "not found") for mm in methods)
        figs.append(f"<figure><img src='{_png(cap.image)}'><figcaption><b>{html.escape(cond)}</b>, "
                    f"{cap.meta['distance_mm']:.0f} mm away{lines}</figcaption></figure>")
    parts.append("<section><h2>Example captures</h2><div class='gallery'>" + "".join(figs) + "</div></section>")

    trace = d.loc[:, ["condition", "capture", "true_region", "est_x", "est_y", "est_z", "confidence",
                      "centre_error_mm"]].head(12)
    parts.append("<section><h2>Traceability table (first rows of results/captures.csv)</h2>"
                 + trace.to_html(index=False, border=0, float_format=lambda v: f"{v:.1f}") + "</section>")

    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            "content='width=device-width,initial-scale=1'><title>CAD Localisation</title>"
            f"<style>{CSS}</style></head><body><header><h1>Where on the part was this captured?</h1>"
            f"<p>Localising captured images on a CAD model: {info['part_triangles']} triangles, "
            f"{info['part_size_mm']:.0f} mm part, {info['references']} reference renders, "
            f"{info['minutes']:.0f} min run</p></header><main>{''.join(parts)}</main></body></html>")
    out.write_text(page, encoding="utf8")
    return out
