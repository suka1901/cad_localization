"""Command line.

    cadloc run                 build part -> reference renders -> train/load models -> evaluate -> tables + dashboard
    cadloc run --quick         small sizes, a few minutes on a laptop CPU (for trying it out)
    cadloc run --retrain       train the CNN again even if models/cnn.pt exists
    cadloc run --config config_colab.yaml --retrain --out results_colab   bigger training on a GPU (Colab)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from . import report
from .data import CONDITIONS, reference_set, test_set
from .evaluate import run as evaluate, summary
from .methods import Features, Forest, Retrieval
from .part import load_part
from .render import Camera, Renderer

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path.cwd() if (Path.cwd() / "config.yaml").exists() else PROJECT

QUICK = {"references": {"n": 400}, "cnn": {"train_images": 600, "epochs": 4}, "test": {"per_condition": 15},
         "forest": {"trees": 12}}


def _merge(cfg: dict, over: dict) -> dict:
    for k, v in over.items():
        cfg[k] = _merge(cfg.get(k, {}), v) if isinstance(v, dict) else v
    return cfg


def run(cfg: dict, out: Path, retrain: bool = False, models_dir: Path | None = None) -> dict:
    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    part = load_part(cfg["part"]["mesh"], cfg["part"]["regions"])
    cam = Camera(cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fov_deg"])
    renderer = Renderer(part, cam)
    roll = cfg["references"]["max_roll_deg"]
    print(f"part: {len(part.mesh.faces)} triangles, {part.size_mm:.0f} mm across, {part.n_regions} regions")

    refs = reference_set(renderer, cfg["references"]["n"], cfg["references"]["seed"], max_roll_deg=roll)
    print(f"{len(refs)} reference renders")
    methods = []
    names = cfg["methods"]
    retrieval = Retrieval(refs)
    if "retrieval" in names:
        methods.append(retrieval)
    if "features" in names:
        methods.append(Features(refs, cam))
    if "forest" in names:
        methods.append(Forest(refs, cam, part.size_mm, cfg["forest"]["pixels_per_image"], cfg["forest"]["trees"],
                              oriented=roll is not None))
    if "cnn" in names:
        from .cnn import CNN, training_renders          # torch is only needed for this method
        c = cfg["cnn"]
        model = CNN(cam, part, c["width"], c.get("patches", 64))
        path = (models_dir or ROOT / "models") / c.get("model_file", "cnn.pt")
        if path.exists() and not retrain:
            print(f"loading {path}")
            model.load(path)
        else:
            print(f"training the CNN on {c['train_images']} randomised CAD renders ({model.device})")
            model.fit(training_renders(renderer, c["train_images"], c["seed"], max_roll_deg=roll), epochs=c["epochs"])
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(path)
        methods.append(model)

    t = cfg["test"]
    tests = {name: test_set(renderer, name, t["per_condition"], t["seed"] + i, max_roll_deg=roll, texture=t["texture"])
             for i, name in enumerate(CONDITIONS)}
    print(f"evaluating {len(methods)} methods on {sum(map(len, tests.values()))} captures")
    df = evaluate(methods, tests, renderer)
    summ = summary(df, t["tolerance_mm"], t.get("strict_mm", 5.0))
    df.to_csv(out / "captures.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    col = f"located_within_{t['tolerance_mm']:g}mm"
    print(summ.pivot(index="condition", columns="method", values=col).round(2).to_string())
    info = {"part_triangles": len(part.mesh.faces), "part_size_mm": part.size_mm, "regions": part.n_regions,
            "references": len(refs), "captures_per_condition": t["per_condition"], "tolerance_mm": t["tolerance_mm"],
            "max_roll_deg": roll, "minutes": (time.time() - t_start) / 60,
            "cnn_history": getattr(methods[-1], "history", []) if "cnn" in names else []}
    (out / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    page = report.build(part, renderer, df, summ, tests, info, out / "dashboard.html")
    print(f"dashboard: {page}\ntables:    {out / 'captures.csv'}, {out / 'summary.csv'}")
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cadloc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--quick", action="store_true", help="small sizes for a first try")
    ap.add_argument("--retrain", action="store_true", help="train the CNN again")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    models = ROOT / "models"
    if args.quick:
        cfg = _merge(cfg, QUICK)
        models = ROOT / "models" / "quick"
    np.seterr(invalid="ignore")
    run(cfg, args.out, retrain=args.retrain, models_dir=models)


if __name__ == "__main__":
    main()
