"""Runner Step 12 (ĐV6): predictions_{model,baseline}.parquet (+deploy_choice.json)
→ reports/results.json + reports/figures/*.png + _evaluate_log.json.

    python scripts/evaluate_phase1.py

Chấm TEST 1 lần toàn lưới (feature_set × model × k) + baselines + config deploy.
exit 0 nếu mọi check pass, 1 nếu lỗi.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import evaluate as ev  # noqa: E402

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODEL_PQ = PROC / "predictions_model.parquet"
BASE_PQ = PROC / "predictions_baseline.parquet"
CHOICE = ROOT / "config" / "deploy_choice.json"
RESULTS = REPORTS / "results.json"
LOG = PROC / "_evaluate_log.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _clean(o):
    """numpy/NaN → JSON-safe (NaN/Inf → null) cho web ĐV7 (JS)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path):
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Không decode được {path}")


# ──────────────────────────── figures ────────────────────────────
def make_figures(res: dict) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []
    hz = res["horizons"]
    ks = [k for k in ev.HORIZONS if str(k) in hz]

    # 1) predictability theo k: deploy balacc/acc vs always_pos (base_pos_rate) + 0.5
    dep = {int(r["k"]): r for r in res["summary"]["deploy_per_k"]}
    if dep:
        kk = [k for k in ks if k in dep]
        bal = [dep[k]["test_balacc"] for k in kk]
        acc = [dep[k]["test_acc"] for k in kk]
        base = [dep[k]["base_pos_rate"] for k in kk]
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.plot(kk, bal, "o-", label="deploy balanced-acc")
        ax.plot(kk, acc, "^-", c="tab:green", alpha=.7, label="deploy acc (thô)")
        ax.plot(kk, base, "s--", c="darkorange", label="always_pos acc (=base_pos_rate)")
        ax.axhline(0.5, ls=":", c="gray", lw=1, label="0.5 (coin-flip balacc)")
        for k in kk:
            if dep[k]["degenerate"]:
                ax.annotate("degen", (k, dep[k]["test_balacc"]), fontsize=7,
                            color="red", xytext=(0, 6), textcoords="offset points", ha="center")
        ax.set(xlabel="horizon k", ylabel="score", title="Predictability test theo k (config deploy)")
        ax.set_xticks(kk); ax.legend(fontsize=8)
        p = FIGURES / "predictability_vs_k.png"
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        written.append(p.name)

    short_order = [s for s in ev.MODELS]                       # en, lgb, lstm
    for k in ks:
        h = hz[str(k)]
        fsets = res["meta"]["feature_sets"]

        # 2) heatmap MCC test (feature_set × model)
        mat = np.full((len(fsets), len(short_order)), np.nan)
        for i, fs in enumerate(fsets):
            for j, s in enumerate(short_order):
                cell = h["grid"].get(fs, {}).get(s)
                if cell is not None:
                    mat[i, j] = cell["mcc"]
        fig, ax = plt.subplots(figsize=(5, 3.6))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
        ax.set_xticks(range(len(short_order)))
        ax.set_xticklabels([ev.MODEL_FULL[s] for s in short_order], rotation=15, fontsize=8)
        ax.set_yticks(range(len(fsets))); ax.set_yticklabels(fsets)
        for i in range(len(fsets)):
            for j in range(len(short_order)):
                v = mat[i, j]
                cell = h["grid"].get(fsets[i], {}).get(short_order[j])
                tag = "*" if (cell and cell.get("degenerate")) else ""
                ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}{tag}",
                        ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="test MCC")
        ax.set_title(f"MCC test (k={k})  * = degenerate")
        p = FIGURES / f"grid_mcc_k{k}.png"
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        written.append(p.name)

        # 3) confusion config deploy
        if "deploy" in h:
            c = h["deploy"]["test"]["confusion"]
            cm = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
            fig, ax = plt.subplots(figsize=(3.4, 3.2))
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["pred +1", "pred −1"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["true +1", "true −1"])
            d = h["deploy"]
            ax.set_title(f"deploy {d['model']}/{d['feature_set']} k={k}")
            p = FIGURES / f"confusion_deploy_k{k}.png"
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            written.append(p.name)
    return written


def main() -> int:
    log = {"step": 12, "ts": datetime.now(TZ).isoformat()}
    try:
        model_df = pd.read_parquet(MODEL_PQ)
        base_df = pd.read_parquet(BASE_PQ)
        choice = _read_json(CHOICE) if CHOICE.exists() else None

        res = ev.build_results(model_df, base_df, deploy_choice=choice)
        ev.validate(res)
        _write_json(RESULTS, res)               # contract trước figures

        figs = []
        try:
            figs = make_figures(res)
        except Exception as e:                  # figures best-effort
            log["figures_error"] = f"{type(e).__name__}: {e}"

        log.update({
            "ok": True, "results_path": str(RESULTS),
            "n_figures": len(figs), "figures": figs,
            "summary": res["summary"],
        })
        _write_json(LOG, log)

        # ── bảng ra màn hình ──
        print(f"OK — {RESULTS.name} | {len(figs)} figure")
        print(f"     {res['summary']['label']}")
        print("     deploy test per-k:")
        for r in res["summary"]["deploy_per_k"]:
            flag = " [DEGEN]" if r["degenerate"] else (" ✓signal" if r["has_signal"] else "")
            print(f"       k={r['k']:<2d} {r['model']:<11s}/{r['feature_set']:<4s} "
                  f"balacc={r['test_balacc']:.3f} mcc={r['test_mcc']:+.3f} "
                  f"acc={r['test_acc']:.3f} (base {r['base_pos_rate']:.3f}) "
                  f"pos={r['pred_pos_rate']:.2f}{flag}")
        return 0
    except Exception as e:
        log.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc()})
        _write_json(LOG, log)
        print(f"FAIL — {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())