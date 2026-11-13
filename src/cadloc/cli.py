"""Command line.

    cadloc run                 build part -> reference renders -> train/load models -> evaluate -> analyses
                               (error budget, calibration, localisability, held-out) -> tables + dashboard
    cadloc run --quick         small sizes, a few minutes on a laptop CPU (for trying it out)
    cadloc run --retrain       train the CNN again even if models/cnn.pt exists
    cadloc run --config config_colab.yaml --retrain --out results_colab   bigger training on a GPU (Colab)
    cadloc run --resume        continue after a restart: finished stages in --out are reused
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
         "forest": {"trees": 12},
         "analysis": {"calibration_per_condition": 15, "heldout_per_set": 15, "map_spots": 60}}


def _merge(cfg: dict, over: dict) -> dict:
    for k, v in over.items():
        cfg[k] = _merge(cfg.get(k, {}), v) if isinstance(v, dict) else v
    return cfg


def run(cfg: dict, out: Path, retrain: bool = False, models_dir: Path | None = None, resume: bool = False) -> dict:
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
        if "cnn_fit" in names:
            from .fit import CNNFit
            f = cfg.get("fit", {})
            methods.append(CNNFit(model, renderer, f.get("anchor_weight", 0.3), tuple(f.get("stages", (20, 8, 4))),
                                  f.get("hypotheses", 3)))

    t = cfg["test"]
    tests = {name: test_set(renderer, name, t["per_condition"], t["seed"] + i, max_roll_deg=roll, texture=t["texture"])
             for i, name in enumerate(CONDITIONS)}
    print(f"evaluating {len(methods)} methods on {sum(map(len, tests.values()))} captures")
    main_csv = out / "captures.csv"
    if resume and main_csv.exists():
        print(f"  reusing {main_csv.name}")
        df = pd.read_csv(main_csv)
    else:
        df = evaluate(methods, tests, renderer)
        df.to_csv(main_csv, index=False)                 # saved now, so a restart can resume from here
    summ = summary(df, t["tolerance_mm"], t.get("strict_mm", 5.0))
    summ.to_csv(out / "summary.csv", index=False)
    col = f"located_within_{t['tolerance_mm']:g}mm"
    print(summ.pivot(index="condition", columns="method", values=col).round(2).to_string())
    by_name = {m.name: m for m in methods}
    extra = None
    if cfg.get("analysis", {}).get("enabled", True) and {"cnn", "cnn_fit"} <= set(by_name):
        from .analysis import analyse
        extra = analyse(cfg, renderer, by_name["cnn"], by_name["cnn_fit"], retrieval, tests, df, out, resume=resume)
    df.to_csv(out / "captures.csv", index=False)
    minutes = (time.time() - t_start) / 60
    if resume and (out / "run_info.json").exists():      # count the time of the run that is being resumed too
        minutes += json.loads((out / "run_info.json").read_text(encoding="utf8")).get("minutes", 0.0)
    info = {"part_triangles": len(part.mesh.faces), "part_size_mm": part.size_mm, "regions": part.n_regions,
            "references": len(refs), "captures_per_condition": t["per_condition"], "tolerance_mm": t["tolerance_mm"],
            "max_roll_deg": roll, "minutes": minutes,
            "cnn_history": getattr(by_name.get("cnn"), "history", [])}
    (out / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    page = report.build(part, renderer, df, summ, tests, info, out / "dashboard.html", extra,
                        by_name.get("cnn"), by_name.get("cnn_fit"))
    print(f"dashboard: {page}\ntables:    {out / 'captures.csv'}, {out / 'summary.csv'}")
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cadloc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--quick", action="store_true", help="small sizes for a first try")
    ap.add_argument("--retrain", action="store_true", help="train the CNN again")
    ap.add_argument("--resume", action="store_true", help="reuse the finished stages in --out (after a restart)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    models = ROOT / "models"
    if args.quick:
        cfg = _merge(cfg, QUICK)
        models = ROOT / "models" / "quick"
    np.seterr(invalid="ignore")
    run(cfg, args.out, retrain=args.retrain, models_dir=models, resume=args.resume)


if __name__ == "__main__":
    main()
