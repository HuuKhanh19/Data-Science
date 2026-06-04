"""Runner Step 11 (ĐV5): features.parquet + hparams.json + feature_sets.json
→ predictions_model.parquet + learning curves + deploy_choice.json + deploy/ (freeze).

Chạy từ gốc repo (env `ds`):
    python scripts/train_infer_phase1.py                          # FULL: 3 model × 3 fset × 4 k + freeze
    python scripts/train_infer_phase1.py --models elastic_net lightgbm   # bỏ LSTM (không cần GPU)
    python scripts/train_infer_phase1.py --horizons 1 --models lightgbm --feature-sets full --skip-freeze
                                                                 #  ↑ smoke test plumbing (nhanh)
    python scripts/train_infer_phase1.py --gpu 1                  # LSTM trên card index 1

FULL = đủ 4 horizon × 3 model × 3 feature_set → ghi artifact canonical + validate + freeze.
Smoke (thiếu horizon/model/feature_set) → ghi _smoke_predictions_model.parquet, KHÔNG
đụng file canonical, KHÔNG validate chặt, KHÔNG freeze (bảo vệ artifact thật).

Lưu ý Windows: nạp `import torch` TRƯỚC numpy/pandas/lightgbm/sklearn (xung đột DLL MKL).
"""
from __future__ import annotations

# torch ĐẦU TIÊN (Windows DLL order). try/except để EN/LGB vẫn chạy khi torch vắng.
try:
    import torch  # noqa: F401
except Exception:
    pass

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.train_infer import (  # noqa: E402
    CANONICAL, COL_OF, FEATURE_SET_ORDER, HORIZONS, MODELS,
    freeze_deploy, run, select_deploy, validate,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features.parquet"
HPARAMS = ROOT / "config" / "hparams.json"
FSETS = ROOT / "config" / "feature_sets.json"
BASELINE = ROOT / "data" / "processed" / "predictions_baseline.parquet"
OUT = ROOT / "data" / "processed" / "predictions_model.parquet"
OUT_SMOKE = ROOT / "data" / "processed" / "_smoke_predictions_model.parquet"
CHOICE = ROOT / "config" / "deploy_choice.json"
LOG = ROOT / "data" / "processed" / "_train_infer_log.json"
FIGDIR = ROOT / "reports" / "figures"
DEPLOY = ROOT / "deploy"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _read_json(path: Path) -> dict:
    """Đọc JSON bền encoding (hparams.json có thể ghi cp1252 trên Windows)."""
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Không decode được {path}")


def _acc(proba: pd.Series, y_true: pd.Series) -> float | None:
    m = (~proba.isna()) & (~y_true.isna())
    if not m.any():
        return None
    pred = (proba[m].to_numpy(float) >= 0.5).astype(int)
    y = (y_true[m].to_numpy(float) > 0).astype(int)
    return round(float((pred == y).mean()), 4)


# ──────────────────────────── learning curves → figures ────────────────────────────
def _plot_curves(curves: dict, figdir: Path) -> list[str]:
    figdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fs, per_k in curves.items():
        for ks, ck in per_k.items():
            if "lgb" in ck:
                c = ck["lgb"]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(c["iters"], c["train_logloss"], label="train")
                ax.plot(c["iters"], c["val_logloss"], label="val")
                ax.set(title=f"LGB logloss vs cây — {fs} k={ks}",
                       xlabel="n_estimators (cây)", ylabel="binary_logloss")
                ax.legend()
                p = figdir / f"learning_lgb_{fs}_k{ks}.png"
                fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
                saved.append(p.name)
            if "lstm" in ck:
                c = ck["lstm"]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(c["epochs"], c["train_bce"], label="inner-train")
                ax.plot(c["epochs"], c["val_bce"], label="inner-val")
                if c.get("best_epoch", -1) > 0:
                    ax.axvline(c["best_epoch"], ls="--", c="grey", lw=1,
                               label=f"best @ {c['best_epoch']}")
                ax.set(title=f"LSTM BCE vs epoch — {fs} k={ks}",
                       xlabel="epoch", ylabel="BCE")
                ax.legend()
                p = figdir / f"learning_lstm_{fs}_k{ks}.png"
                fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
                saved.append(p.name)
            if "en" in ck:
                c = ck["en"]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(c["lambdas"], c["val_logloss"])
                ax.axvline(c["chosen_lambda"], ls="--", c="red", lw=1,
                           label=f"λ* = {c['chosen_lambda']:.4g}")
                ax.set(xscale="log", title=f"EN val-logloss vs λ — {fs} k={ks}",
                       xlabel="λ (log)", ylabel="val binary_logloss")
                ax.legend()
                p = figdir / f"learning_en_{fs}_k{ks}.png"
                fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
                saved.append(p.name)
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS), choices=list(HORIZONS))
    ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SET_ORDER),
                    choices=list(FEATURE_SET_ORDER))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-freeze", action="store_true")
    ap.add_argument("--no-curves", action="store_true")
    args = ap.parse_args()

    is_full = (set(args.models) == set(MODELS)
               and set(args.horizons) == set(HORIZONS)
               and set(args.feature_sets) == set(FEATURE_SET_ORDER))
    out_path = OUT if is_full else OUT_SMOKE

    try:
        df = pd.read_parquet(FEATURES)
        hparams = _read_json(HPARAMS)
        fsets = _read_json(FSETS)
        feature_sets = {k: fsets[k] for k in FEATURE_SET_ORDER}  # bỏ _meta

        pred, curves = run(df, hparams, feature_sets,
                           models=args.models, horizons=args.horizons,
                           feature_set_names=args.feature_sets, gpu=args.gpu,
                           want_curves=not args.no_curves, verbose=True)

        figs = [] if args.no_curves else _plot_curves(curves, FIGDIR)

        if is_full:
            baseline = pd.read_parquet(BASELINE) if BASELINE.exists() else None
            validate(pred, baseline, feature_set_names=args.feature_sets)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(out_path, index=False)

        # selection per-k (chấm trên val)
        choice = select_deploy(pred)
        if is_full:
            CHOICE.parent.mkdir(parents=True, exist_ok=True)
            CHOICE.write_text(json.dumps(choice, ensure_ascii=False, indent=2), encoding="utf-8")

        # freeze refit-on-all (chỉ FULL, không skip)
        manifest = None
        if is_full and not args.skip_freeze:
            manifest = freeze_deploy(df, hparams, feature_sets, choice, DEPLOY, gpu=args.gpu)

        # ── log accuracy per (fs,model,k,segment) ──
        acc = {}
        for fs in args.feature_sets:
            acc[fs] = {}
            for k in args.horizons:
                acc[fs][str(k)] = {}
                for seg in ("val", "test"):
                    sub = pred[(pred.feature_set == fs) & (pred.k == k) & (pred.segment == seg)]
                    acc[fs][str(k)][seg] = {
                        m: _acc(sub[COL_OF[m]], sub["y_true"]) for m in args.models}

        log = {
            "status": "ok", "mode": "full" if is_full else "smoke",
            "output": str(out_path), "rows": int(len(pred)),
            "models": args.models, "horizons": args.horizons, "feature_sets": args.feature_sets,
            "acc_at_0.5": acc, "deploy_choice": choice,
            "figures": figs, "freeze": manifest,
            "generated_at": datetime.now(TZ).isoformat(),
        }
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── bảng ra màn hình ──
        print(f"\n[{'FULL' if is_full else 'SMOKE'}] {out_path.name}: {len(pred)} dòng "
              f"| {len(figs)} figure | freeze={'có' if manifest else 'không'}")
        print("deploy per-k (best single trên val):")
        for k in sorted(args.horizons):
            w = choice.get(str(k))
            if w:
                tag = " [ALL-DEGEN]" if w.get("all_degenerate") else ""
                print(f"  k={k:<2d} → {w['model']:<11s} fs={w['feature_set']:<4s} "
                      f"val-MCC={w['val_mcc']:+.3f} balacc={w['val_balacc']:.3f} "
                      f"pos={w['val_pred_pos_rate']:.2f}{tag}")
        return 0
    except Exception as e:
        err = {"status": "error", "error": repr(e),
               "generated_at": datetime.now(TZ).isoformat()}
        try:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            LOG.write_text(json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        print(f"[ERROR] {e!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())