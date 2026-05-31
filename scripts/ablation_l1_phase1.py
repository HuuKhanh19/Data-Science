"""Thí nghiệm ablation L1-only — NẰM NGOÀI pre-registration (exploratory).

Train lại đúng 3 model × 4 horizon nhưng CHỈ dùng lớp L1 (4 log-return thuần
r1,r5,r10,r20), bỏ L2/L3/L4 — để đo L2+L3+L4 có thêm predictability so với chỉ giá.

Cơ chế: `walk_forward.feature_columns(df)` suy cột feature từ `df.columns` (mọi cột trừ
date + 4 nhãn). Nên chỉ cần truyền df rút gọn còn `date + L1 + nhãn` là `tune_all` và `run`
tự chạy L1-only — KHÔNG sửa module nào đã done. Mọi artifact gắn hậu tố `_L1`; file canonical
(hparams.json, predictions_model.parquet, results.json) GIỮ NGUYÊN.

Re-tune riêng cho L1 (hparams_L1.json): hparams gốc tune cho 20 feature, áp lên 4 feature sẽ
lẫn tác động-bỏ-feature với lệch-hyperparameter. Đây là ablation sạch, không đụng pre-reg.

    python scripts/ablation_l1_phase1.py                    # FULL: 3 model × 4 horizon + so sánh
    python scripts/ablation_l1_phase1.py --models elastic_net lightgbm   # bỏ LSTM
    python scripts/ablation_l1_phase1.py --gpu 1            # LSTM trên card 1
"""
import argparse
import json
import math
import sys
import traceback
from pathlib import Path

try:                                   # preload torch (thứ tự DLL Windows — như Step 11)
    import torch  # noqa: F401
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.model.tuning import tune_all
from src.model.tuning import validate as tune_validate
from src.model.train_infer import HORIZONS, MODELS, run
from src.model.train_infer import validate as ti_validate
from src.model import evaluate as ev
from src.model.walk_forward import LABEL_COLS

L1 = ["r1", "r5", "r10", "r20"]
COL_OF = {"elastic_net": "en_proba", "lightgbm": "lgb_proba", "lstm": "lstm_proba"}

FEATURES = Path("data/processed/features.parquet")
BASELINE = Path("data/processed/predictions_baseline.parquet")
FULL_RESULTS = Path("reports/results.json")              # để so sánh (chỉ đọc)
HPARAMS_L1 = Path("config/hparams_L1.json")
PREDS_L1 = Path("data/processed/predictions_model_L1.parquet")
PREDS_L1_SMOKE = Path("data/processed/_smoke_predictions_model_L1.parquet")
RESULTS_L1 = Path("reports/results_L1.json")
FIG = Path("reports/figures_L1/ablation_predictability.png")
LOG = Path("data/processed/_ablation_l1_log.json")


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def _curve_row(res: dict, k: int) -> dict:
    for c in res["predictability_curve"]:
        if c["k"] == k:
            return c
    return {}


def compare_figure(res_l1: dict, res_full: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = list(HORIZONS)
    acc_l1 = [_curve_row(res_l1, k).get("acc_best_model") for k in ks]
    rao = [res_l1["horizons"][str(k)]["pct_pos_test"] for k in ks]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(ks, acc_l1, "o-", label="L1-only best acc")
    if res_full is not None:
        acc_full = [_curve_row(res_full, k).get("acc_best_model") for k in ks]
        ax.plot(ks, acc_full, "^-", c="green", label="full (20 feat) best acc")
    ax.plot(ks, rao, "s--", c="darkorange", label="rào always_pos")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xlabel("horizon k"); ax.set_ylabel("accuracy")
    ax.set_title("Ablation: L1-only vs full"); ax.set_xticks(ks); ax.legend()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG, dpi=120); plt.close(fig)


def print_compare(res_l1: dict, res_full: dict | None) -> dict:
    """In bảng L1 vs full + trả dict so sánh để ghi log."""
    cmp = {"overall_L1": res_l1["overall"]}
    print("\n=== ABLATION L1-only vs FULL (20 feat) ===")
    print(f"  overall L1   : {res_l1['overall']['label']} "
          f"({res_l1['overall']['n_positive_horizons']}/4)")
    if res_full is not None:
        cmp["overall_full"] = res_full["overall"]
        print(f"  overall full : {res_full['overall']['label']} "
              f"({res_full['overall']['n_positive_horizons']}/4)")
    head = f"  {'k':>3} | {'acc_L1':>7} {'acc_full':>8} {'Δ(full-L1)':>10} | " \
           f"{'pAdj_L1':>8} {'pAdj_full':>9} | {'verdict_L1':>10} {'verdict_full':>12}"
    print(head); print("  " + "-" * (len(head) - 2))
    cmp["per_k"] = {}
    for k in HORIZONS:
        cl = _curve_row(res_l1, k)
        vL = res_l1["horizons"][str(k)]["verdict"]["predictable"]
        aL, pL = cl.get("acc_best_model"), cl.get("p_adj_min_vs_persistence")
        if res_full is not None:
            cf = _curve_row(res_full, k)
            vF = res_full["horizons"][str(k)]["verdict"]["predictable"]
            aF, pF = cf.get("acc_best_model"), cf.get("p_adj_min_vs_persistence")
            d = (aF - aL) if (aF is not None and aL is not None) else float("nan")
            print(f"  {k:>3} | {aL:7.4f} {aF:8.4f} {d:10.4f} | "
                  f"{pL:8.4f} {pF:9.4f} | {str(vL):>10} {str(vF):>12}")
            cmp["per_k"][str(k)] = {"acc_L1": aL, "acc_full": aF, "delta_full_minus_L1": d,
                                    "p_adj_L1": pL, "p_adj_full": pF,
                                    "verdict_L1": vL, "verdict_full": vF}
        else:
            print(f"  {k:>3} | {aL:7.4f} {'—':>8} {'—':>10} | {pL:8.4f} {'—':>9} | {str(vL):>10}")
            cmp["per_k"][str(k)] = {"acc_L1": aL, "p_adj_L1": pL, "verdict_L1": vL}
    return cmp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS), choices=HORIZONS)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--gpu", type=int, default=0, help="index card GPU cho LSTM")
    args = ap.parse_args()
    horizons = sorted(set(args.horizons))
    models = [m for m in MODELS if m in set(args.models)]
    full = set(horizons) == set(HORIZONS) and set(models) == set(MODELS)
    log = {"experiment": "ablation_L1_only", "pre_registered": False,
           "note": "exploratory ablation — KHÔNG vào confirmatory results (features locked, §12 #1)",
           "feature_set": L1, "mode": "full" if full else "smoke",
           "horizons": horizons, "models": models}

    try:
        df = pd.read_parquet(FEATURES)
        df_l1 = df[["date", *L1, *LABEL_COLS]].copy()      # → feature_columns trả đúng L1

        # 1) re-tune RIÊNG cho L1 (hparams gốc giữ nguyên)
        hp = tune_all(df_l1)
        tune_validate(hp, df_l1)
        _write_json(HPARAMS_L1, hp)
        log["hparams_L1"] = str(HPARAMS_L1)

        # 2) walk-forward train + infer trên L1
        if "lstm" in models:
            print(f"[gpu] LSTM dùng cuda:{args.gpu} = {torch.cuda.get_device_name(args.gpu)}")
        preds = run(df_l1, hp, horizons=horizons, models=models, gpu=args.gpu)

        baseline = pd.read_parquet(BASELINE)
        dst = PREDS_L1 if full else PREDS_L1_SMOKE
        if full:
            ti_validate(preds, baseline)                   # khớp (date,k) với baseline
        dst.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(dst, index=False)
        log["predictions_L1"] = str(dst)

        if not full:                                       # smoke: dừng, không evaluate
            print(f"[smoke] preds L1 -> {dst} (bỏ evaluate vì thiếu model/horizon)")
            _write_json(LOG, log)
            return 0

        # 3) evaluate L1 (tái dùng Step 12) + so sánh full
        aligned = ev.align(preds, baseline)
        res_l1 = ev.build_results(aligned)
        ev.validate(res_l1)
        _write_json(RESULTS_L1, res_l1)
        log["results_L1"] = str(RESULTS_L1)

        res_full = None
        if FULL_RESULTS.exists():
            res_full = json.loads(FULL_RESULTS.read_text(encoding="utf-8"))
        try:
            compare_figure(res_l1, res_full)
            log["figure"] = str(FIG)
        except Exception as e:
            log["figure_error"] = f"{type(e).__name__}: {e}"

        log["comparison"] = print_compare(res_l1, res_full)
        log["status"] = "ok"
        _write_json(LOG, log)
        print(f"\n[ok] results L1 -> {RESULTS_L1}  | log -> {LOG}")
        return 0
    except Exception as e:
        log["status"] = "error"; log["error"] = f"{type(e).__name__}: {e}"
        log["trace"] = traceback.format_exc()
        _write_json(LOG, log)
        print(f"[error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())