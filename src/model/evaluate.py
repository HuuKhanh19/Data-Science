"""Step 12 — Đánh giá thống kê (logic thuần, không I/O).

Đọc dự đoán đã căn (model + baseline), tính:
  - metrics: acc / balacc / mcc / confusion / precision / recall
  - 95% CI: stationary block bootstrap (mean block 2k), paired cho Δacc
  - Diebold-Mariano: 0-1 loss, HAC Newey-West lag q=k-1, hiệu chỉnh Harvey
  - Holm-Bonferroni: trong từng horizon, M_k = 9 (3 model x 3 baseline)
  - test phụ analytical-50%: z-test acc vs 0.5, HAC (ngoài family Holm)
  - verdict §8: predictable iff (p_adj vs persistence <=.05) AND (CI Δacc > 0)

Pre-reg: research_design.md §6, §7, §8. Trục quyết định = persistence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = (1, 5, 10, 20)
MODELS = {"en": "en_proba", "lgb": "lgb_proba", "lstm": "lstm_proba"}
BASELINES = ("persistence", "dyn_majority", "always_pos")
DECISION_BASELINE = "persistence"          # §1.3 / §8.1 — trục phán quyết

ALPHA = 0.05
SUGGESTIVE_HI = 0.10                        # §8.3 vùng mập mờ
B_BOOT = 2000                              # §7.2
BOOT_SEED = 42
HOLM_M = 9                                 # §7.3 — khóa pre-reg

PRED_COLS = list(BASELINES)
PROBA_COLS = list(MODELS.values())


def block_len(k: int) -> int:
    return 2 * k                            # mean block length §7.2


# ───────────────────────── căn dữ liệu ─────────────────────────

def _sign_from_proba(p: np.ndarray) -> np.ndarray:
    """proba >= 0.5 -> +1, else -1 (tie 0.5 -> +1, khớp baseline)."""
    return np.where(p >= 0.5, 1, -1).astype(np.int64)


def align(model_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """Merge (date,k), bỏ đuôi y_true NaN, suy sign model. Pure, raise nếu lệch."""
    m = model_df[["date", "k", "y_true", *PROBA_COLS]].copy()
    b = base_df[["date", "k", "y_true", *PRED_COLS]].copy()
    out = m.merge(b, on=["date", "k"], how="inner", suffixes=("_m", "_b"))
    if len(out) != len(m) or len(out) != len(b):
        raise ValueError(
            f"align: (date,k) không khớp — model={len(m)} base={len(b)} merged={len(out)}")
    yd = (out["y_true_m"].fillna(-99) - out["y_true_b"].fillna(-99)).abs().max()
    if yd > 0:
        raise ValueError("align: y_true model vs baseline lệch nhau")
    out = out.rename(columns={"y_true_m": "y_true"}).drop(columns=["y_true_b"])
    out = out[out["y_true"].notna()].reset_index(drop=True)   # bỏ đuôi inference
    for name, col in MODELS.items():
        out[f"{name}_sign"] = _sign_from_proba(out[col].to_numpy())
    out["y_true"] = out["y_true"].astype(np.int64)
    return out


# ───────────────────────── metrics ─────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """+1 = positive class."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == -1) & (y_true == -1)))
    fp = int(np.sum((y_pred == 1) & (y_true == -1)))
    fn = int(np.sum((y_pred == -1) & (y_true == 1)))
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else np.nan
    tpr = tp / (tp + fn) if (tp + fn) else np.nan      # recall pos
    tnr = tn / (tn + fp) if (tn + fp) else np.nan      # recall neg
    balacc = np.nanmean([tpr, tnr])
    prec_pos = tp / (tp + fp) if (tp + fp) else np.nan
    prec_neg = tn / (tn + fn) if (tn + fn) else np.nan
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {
        "acc": acc, "balacc": balacc, "mcc": mcc,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": {"pos": prec_pos, "neg": prec_neg},
        "recall": {"pos": tpr, "neg": tnr},
    }


# ───────────────────────── HAC / DM ─────────────────────────

def hac_var_mean(d: np.ndarray, q: int) -> float:
    """Newey-West HAC variance của TRUNG BÌNH d, Bartlett lag q (>=0)."""
    n = d.size
    dc = d - d.mean()
    g0 = float(np.dot(dc, dc) / n)
    s = g0
    for j in range(1, q + 1):
        gj = float(np.dot(dc[j:], dc[:-j]) / n)
        s += 2.0 * (1.0 - j / (q + 1)) * gj
    return s / n                                       # Bartlett -> luôn >= 0


def dm_test(loss_b: np.ndarray, loss_m: np.ndarray, q: int) -> dict:
    """d = L_baseline - L_model (>0 nghĩa model tốt hơn). Một phía, t_{N-1}.

    Degenerate (var=0, model≡baseline) -> dm=NaN, p_raw=1.0, flag.
    """
    d = loss_b - loss_m
    n = d.size
    dbar = float(d.mean())
    v = hac_var_mean(d, q)
    if v <= 1e-15:
        return {"dm": np.nan, "dm_star": np.nan, "p_raw": 1.0, "degenerate": True}
    dm = dbar / np.sqrt(v)
    harvey = np.sqrt((n + 1 - 2 * q + q * (q - 1) / n) / n)
    dm_star = dm * harvey
    p_raw = float(stats.t.sf(dm_star, df=n - 1))       # H1: model tốt hơn -> đuôi phải
    return {"dm": float(dm), "dm_star": float(dm_star),
            "p_raw": p_raw, "degenerate": False}


def analytical_50(correct: np.ndarray, q: int) -> dict:
    """Test phụ: acc vs 0.5, HAC lag q. Một phía (acc>0.5). Ngoài Holm."""
    d = correct - 0.5
    v = hac_var_mean(d, q)
    if v <= 1e-15:
        return {"z": np.nan, "p": np.nan}
    z = float(d.mean() / np.sqrt(v))
    return {"z": z, "p": float(stats.norm.sf(z))}


# ───────────────────────── bootstrap ─────────────────────────

def _sb_indices(n: int, L: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary bootstrap (Politis-Romano 1994): ma trận (B,n) chỉ số resample."""
    p = 1.0 / L
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=B)
    cont = rng.random((B, n)) >= p                     # True = nối block
    starts = rng.integers(0, n, size=(B, n))           # khởi đầu block mới
    for t in range(1, n):
        nxt = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(cont[:, t], nxt, starts[:, t])
    return idx


def boot_ci(correct_m: np.ndarray, correct_p: np.ndarray, L: int,
            rng: np.random.Generator) -> tuple[list, list, float]:
    """CI acc model + CI Δacc(model-persistence), PAIRED (cùng index resample)."""
    n = correct_m.size
    idx = _sb_indices(n, L, B_BOOT, rng)
    acc_b = correct_m[idx].mean(axis=1)
    dlt_b = correct_m[idx].mean(axis=1) - correct_p[idx].mean(axis=1)
    acc_ci = [float(np.percentile(acc_b, 2.5)), float(np.percentile(acc_b, 97.5))]
    dlt_ci = [float(np.percentile(dlt_b, 2.5)), float(np.percentile(dlt_b, 97.5))]
    return acc_ci, dlt_ci, float(dlt_b.mean())


# ───────────────────────── calibration ─────────────────────────

def calibration_bins(proba: np.ndarray, y_true: np.ndarray, nbins: int = 10) -> list:
    pos = (y_true == 1).astype(float)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    out = []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (proba >= lo) & (proba < hi if i < nbins - 1 else proba <= hi)
        if m.sum() == 0:
            continue
        out.append({"p_mid": float(proba[m].mean()),
                    "emp_freq": float(pos[m].mean()),
                    "n": int(m.sum())})
    return out


# ───────────────────────── Holm ─────────────────────────

def holm(pmap: dict, M: int = HOLM_M) -> dict:
    """Holm-Bonferroni trong 1 horizon. NaN/degenerate coi như p=1.0 (bảo thủ)."""
    items = sorted(pmap.items(),
                   key=lambda kv: 1.0 if kv[1] is None or np.isnan(kv[1]) else kv[1])
    adj, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        pe = 1.0 if (p is None or np.isnan(p)) else p
        running = max(running, (M - i) * pe)           # 1-indexed: M-(i+1)+1 = M-i
        adj[key] = min(1.0, running)
    return adj


# ───────────────────────── build ─────────────────────────

def _eval_horizon(g: pd.DataFrame, k: int, rng: np.random.Generator) -> dict:
    y = g["y_true"].to_numpy()
    n = len(g)
    q = k - 1
    L = block_len(k)

    base_pred = {b: g[b].to_numpy().astype(np.int64) for b in BASELINES}
    base_loss = {b: (base_pred[b] != y).astype(float) for b in BASELINES}
    dyn_eq_always = bool(np.array_equal(base_pred["dyn_majority"],
                                        base_pred["always_pos"]))

    models_out, p_for_holm = {}, {}
    for name in MODELS:
        pred = g[f"{name}_sign"].to_numpy()
        proba = g[MODELS[name]].to_numpy()
        loss_m = (pred != y).astype(float)
        correct_m = (pred == y).astype(float)
        correct_p = (base_pred[DECISION_BASELINE] == y).astype(float)

        mt = metrics(y, pred)
        acc_ci, dlt_ci, dlt = boot_ci(correct_m, correct_p, L, rng)

        dm_block = {}
        for b in BASELINES:
            r = dm_test(base_loss[b], loss_m, q)
            coincident = bool(dyn_eq_always and b in ("dyn_majority", "always_pos"))
            dm_block[b] = {**r, "coincident": coincident}
            p_for_holm[(name, b)] = r["p_raw"]

        models_out[name] = {
            **mt,
            "acc_ci": acc_ci,
            "delta_acc_vs_persistence": dlt,
            "delta_acc_ci": dlt_ci,
            "analytical_50": analytical_50(correct_m, q),
            "calibration": calibration_bins(proba, y),
            "dm": dm_block,
        }

    # Holm trong horizon (9 test) -> gắn p_adj + reject
    p_adj = holm(p_for_holm, HOLM_M)
    for (name, b), pa in p_adj.items():
        models_out[name]["dm"][b]["p_adj"] = pa
        models_out[name]["dm"][b]["reject"] = bool(pa <= ALPHA)

    # verdict §8.1: tồn tại model với p_adj(persistence)<=.05 AND CI Δacc>0
    predictable, suggestive = False, False
    for name in MODELS:
        d = models_out[name]["dm"][DECISION_BASELINE]
        ci_lo = models_out[name]["delta_acc_ci"][0]
        if d["p_adj"] <= ALPHA and ci_lo > 0:
            predictable = True
        if ALPHA < d["p_adj"] <= SUGGESTIVE_HI:
            suggestive = True
    if predictable:
        reason = "≥1 model: p_adj(persistence)≤.05 AND CI Δacc>0"
    elif suggestive:
        reason = "p_adj(persistence)∈(.05,.10] — suggestive, inconclusive"
    else:
        reason = "không model nào vượt persistence có ý nghĩa"

    iso = g["date"].dt.isocalendar()
    n_weeks = int((iso["year"].astype(int) * 100 + iso["week"].astype(int)).nunique())
    return {
        "n_test": n,
        "n_weeks": n_weeks,
        "pct_pos_test": float((y == 1).mean()),
        "dyn_eq_always": dyn_eq_always,
        "models": models_out,
        "verdict": {"predictable": predictable,
                    "suggestive": bool(suggestive and not predictable),
                    "reason": reason},
    }


def build_results(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rng = np.random.default_rng(BOOT_SEED)

    horizons, curve = {}, []
    for k in HORIZONS:
        g = df[df["k"] == k].sort_values("date").reset_index(drop=True)
        h = _eval_horizon(g, k, rng)
        horizons[str(k)] = h

        best_name = min(MODELS, key=lambda nm: (
            h["models"][nm]["dm"][DECISION_BASELINE]["p_adj"]))
        bm = h["models"][best_name]
        curve.append({
            "k": k,
            "acc_best_model": max(h["models"][nm]["acc"] for nm in MODELS),
            "p_adj_min_vs_persistence": bm["dm"][DECISION_BASELINE]["p_adj"],
            "delta_acc": bm["delta_acc_vs_persistence"],
        })

    n_pos = sum(horizons[str(k)]["verdict"]["predictable"] for k in HORIZONS)
    label = ("strong" if n_pos >= 3 else "multi" if n_pos == 2
             else "limited" if n_pos == 1 else "no_evidence")

    tie_rate = {"1": 0.078, "5": 0.023, "10": 0.011, "20": 0.006}  # EDA Phase 0
    return {
        "meta": {
            "B": B_BOOT, "bootstrap_seed": BOOT_SEED,
            "block_lengths": {str(k): block_len(k) for k in HORIZONS},
            "holm_M_k": HOLM_M, "alpha": ALPHA,
            "decision_baseline": DECISION_BASELINE,
            "dyn_eq_always_all": all(horizons[str(k)]["dyn_eq_always"]
                                     for k in HORIZONS),
            "tie_rate": tie_rate,
            "sensitivity_note": ("cond-3 §8.1 (λ×{.5,1,2}, window{750,1000,1250}) "
                                 "cần chạy lại Step 11 — ngoài scope Step 12"),
        },
        "horizons": horizons,
        "overall": {"n_positive_horizons": int(n_pos),
                    "label": label, "predictable": bool(n_pos >= 2)},
        "predictability_curve": curve,
    }


# ───────────────────────── validate (N check) ─────────────────────────

def validate(res: dict) -> None:
    for k in HORIZONS:
        h = res["horizons"][str(k)]
        assert len(h["models"]) == 3, f"k={k}: thiếu model"
        pmap = {}
        for name in MODELS:
            m = h["models"][name]
            # dm_finite_or_flagged
            for b in BASELINES:
                d = m["dm"][b]
                ok = np.isfinite(d["dm"]) if d["dm"] is not None and \
                    not (isinstance(d["dm"], float) and np.isnan(d["dm"])) else False
                assert ok or d["degenerate"] or d["coincident"], \
                    f"k={k} {name}/{b}: DM không finite mà không flag"
                pmap[(name, b)] = d["p_raw"]
            # ci_order
            for lo, hi in (m["acc_ci"], m["delta_acc_ci"]):
                assert lo <= hi, f"k={k} {name}: CI lo>hi"
            dlt = m["delta_acc_vs_persistence"]
            lo, hi = m["delta_acc_ci"]
            assert lo - 1e-9 <= dlt <= hi + 1e-9, \
                f"k={k} {name}: Δacc ngoài CI của nó"
        # holm_monotone + M_k=9
        assert len(pmap) == HOLM_M, f"k={k}: số test Holm != {HOLM_M}"
        order = sorted(pmap, key=lambda key: 1.0 if pmap[key] is None or
                       np.isnan(pmap[key]) else pmap[key])
        adj_seq = [h["models"][nm]["dm"][b]["p_adj"] for nm, b in order]
        for a, c in zip(adj_seq, adj_seq[1:]):
            assert c >= a - 1e-12, f"k={k}: p_adj không đơn điệu"
        # verdict_consistent
        pred = any(h["models"][nm]["dm"][DECISION_BASELINE]["p_adj"] <= ALPHA and
                   h["models"][nm]["delta_acc_ci"][0] > 0 for nm in MODELS)
        assert pred == h["verdict"]["predictable"], f"k={k}: verdict lệch"

    n_pos = sum(res["horizons"][str(k)]["verdict"]["predictable"] for k in HORIZONS)
    assert res["overall"]["n_positive_horizons"] == n_pos, "overall count lệch"
    assert res["overall"]["predictable"] == (n_pos >= 2), "overall threshold lệch"  