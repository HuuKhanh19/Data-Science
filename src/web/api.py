"""FastAPI app (Step 13) — API mỏng phục vụ web local, tái dùng cho Phase 2/3.

Endpoint:
    GET /api/health            → liveness
    GET /api/inference/latest  → dự đoán phiên mới nhất (4 horizon × 3 model) + caveat
    GET /api/results           → results.json (study chính)
    GET /api/results/ablation  → results_L1.json (ablation L1; 404 nếu chưa chạy)
    GET /                      → dashboard tĩnh (src/web/static/index.html)

Chạy: python scripts/serve_phase1.py  (hoặc: uvicorn src.web.api:app)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from src.web import data

app = FastAPI(title="TCB Predictability — Phase 1", version="1.0")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/inference/latest")
def inference_latest() -> dict:
    try:
        return data.latest_inference()
    except FileNotFoundError:
        raise HTTPException(404, "predictions_model.parquet chưa có — chạy Step 11 trước")


@app.get("/api/inference/history")
def inference_history(k: int = Query(..., description="horizon ∈ {1,5,10,20}"),
                      n: int = Query(20, ge=1, le=200)) -> dict:
    if k not in data.HORIZONS:
        raise HTTPException(400, f"k phải thuộc {data.HORIZONS}")
    try:
        return data.inference_history(k, n)
    except FileNotFoundError:
        raise HTTPException(404, "predictions_model.parquet chưa có — chạy Step 11 trước")


@app.get("/api/results")
def results() -> dict:
    r = data.load_results()
    if r is None:
        raise HTTPException(404, "results.json chưa có — chạy Step 12 trước")
    return r


@app.get("/api/results/ablation")
def results_ablation() -> dict:
    r = data.load_results_l1()
    if r is None:
        raise HTTPException(404, "results_L1.json chưa có — chạy ablation_l1_phase1.py")
    return r


# Mount static SAU các route /api (để không che) — html=True phục vụ index.html ở "/".
_STATIC = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")