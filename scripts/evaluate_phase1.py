"""Step 12 runner — đánh giá thống kê.

  python scripts/evaluate_phase1.py

Đọc data/processed/predictions_{model,baseline}.parquet
  → reports/results.json + reports/figures/*.png + _evaluate_log.json
exit 0 nếu mọi check pass, 1 nếu lỗi.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import evaluate as ev   # noqa: E402

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODEL_PQ = PROC / "predictions_model.parquet"
BASE_PQ = PROC / "predictions_baseline.parquet"
RESULTS = REPORTS / "results.json"
LOG = PROC / "_evaluate_log.json"


def _clean(o):
    """numpy/NaN -> JSON-safe (NaN/Inf -> null) cho Step 13 (JS)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if hasattr(o, "item"):                       # numpy scalar
        return _clean(o.item())
    return o


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=2),
                    encoding="utf-8")


def make_figures(res: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []

    # 1) predictability theo k: acc best-model vs rào always_pos (= pct_pos)
    curve = res["predictability_curve"]
    ks = [c["k"] for c in curve]
    acc = [c["acc_best_model"] for c in curve]
    rao = [res["horizons"][str(k)]["pct_pos_test"] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, acc, "o-", label="best model acc")
    ax.plot(ks, rao, "s--", c="darkorange", label="rào always_pos")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xlabel("horizon k"); ax.set_ylabel("accuracy")
    ax.set_title("Predictability theo k"); ax.set_xticks(ks); ax.legend()
    p = FIGURES / "predictability_vs_k.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    written.append(p.name)

    for k in ev.HORIZONS:
        h = res["horizons"][str(k)]
        # 2) DM* heatmap 3 model x 3 baseline
        mat = np.array([[h["models"][m]["dm"][b]["dm_star"]
                         if h["models"][m]["dm"][b]["dm_star"] is not None else np.nan
                         for b in ev.BASELINES] for m in ev.MODELS])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(ev.BASELINES, rotation=20)
        ax.set_yticks(range(3)); ax.set_yticklabels(list(ev.MODELS))
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}",
                        ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="DM*")
        ax.set_title(f"DM* (k={k})")
        pth = FIGURES / f"dm_heatmap_k{k}.png"
        fig.tight_layout(); fig.savefig(pth, dpi=120); plt.close(fig)
        written.append(pth.name)

        for m in ev.MODELS:
            md = h["models"][m]
            # 3) confusion
            c = md["confusion"]
            cm = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
            fig, ax = plt.subplots(figsize=(3.4, 3.2))
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["pred +1", "pred −1"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["true +1", "true −1"])
            ax.set_title(f"{m} k={k}")
            pth = FIGURES / f"confusion_{m}_k{k}.png"
            fig.tight_layout(); fig.savefig(pth, dpi=120); plt.close(fig)
            written.append(pth.name)
            # 4) calibration
            cal = md["calibration"]
            if cal:
                fig, ax = plt.subplots(figsize=(3.6, 3.6))
                ax.plot([0, 1], [0, 1], ls=":", c="gray")
                ax.plot([b["p_mid"] for b in cal], [b["emp_freq"] for b in cal],
                        "o-")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xlabel("mean proba"); ax.set_ylabel("emp. freq (+1)")
                ax.set_title(f"calibration {m} k={k}")
                pth = FIGURES / f"calibration_{m}_k{k}.png"
                fig.tight_layout(); fig.savefig(pth, dpi=120); plt.close(fig)
                written.append(pth.name)
    return written


def main() -> int:
    log = {"step": 12, "ts": datetime.now(timezone.utc).isoformat()}
    try:
        model_df = pd.read_parquet(MODEL_PQ)
        base_df = pd.read_parquet(BASE_PQ)
        df = ev.align(model_df, base_df)
        res = ev.build_results(df)
        ev.validate(res)
        _write_json(RESULTS, res)            # ghi contract TRƯỚC figures

        figs = []
        try:
            figs = make_figures(res)
        except Exception as e:               # figures best-effort
            log["figures_error"] = f"{type(e).__name__}: {e}"

        log.update({
            "ok": True,
            "n_rows_aligned": int(len(df)),
            "per_horizon": {str(k): {
                "n_test": res["horizons"][str(k)]["n_test"],
                "predictable": res["horizons"][str(k)]["verdict"]["predictable"],
            } for k in ev.HORIZONS},
            "overall": res["overall"],
            "dyn_eq_always_all": res["meta"]["dyn_eq_always_all"],
            "n_figures": len(figs),
            "results_path": str(RESULTS),
        })
        _write_json(LOG, log)
        print(f"OK — {RESULTS}  overall={res['overall']['label']} "
              f"({res['overall']['n_positive_horizons']}/4)")
        return 0
    except Exception as e:
        log.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc()})
        _write_json(LOG, log)
        print(f"FAIL — {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())