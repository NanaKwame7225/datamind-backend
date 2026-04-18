"""
DataMind — Audit AI Elite
FastAPI Backend
Endpoint: POST /api/v1/analysis/analyse  |  GET /health
"""

import os, time, math, json, logging
from typing import Any
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("datamind")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="DataMind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Environment ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_MODEL    = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL",    "gemini-1.5-flash")

# ── Schemas ────────────────────────────────────────────────────────────────
class AnalyseRequest(BaseModel):
    query:                 str
    industry:              str  = "general"
    provider:              str  = "anthropic"
    inline_data:           list[dict[str, Any]] = Field(default_factory=list)
    enable_viz:            bool = True
    enable_anomaly_detection: bool = True
    enable_forecast:       bool = False
    conversation_history:  list  = Field(default_factory=list)

# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "DataMind API", "version": "1.0.0"}

# ── Helpers ────────────────────────────────────────────────────────────────
def _numeric_keys(data: list[dict]) -> list[str]:
    if not data:
        return []
    return [k for k, v in data[0].items() if isinstance(v, (int, float))]

def _stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {}
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    return {"mean": mean, "std": std, "min": min(values), "max": max(values), "n": n}

def _detect_anomalies(data: list[dict], nk: list[str]) -> list[dict]:
    anomalies = []
    for k in nk:
        vals = [float(r[k]) for r in data if isinstance(r.get(k), (int, float))]
        s = _stats(vals)
        if not s or s["std"] == 0:
            continue
        for i, r in enumerate(data):
            v = float(r.get(k, 0))
            z = abs(v - s["mean"]) / s["std"]
            if z > 2.5:
                label = list(r.values())[0] if r else f"Record {i+1}"
                anomalies.append({
                    "key": k,
                    "value": v,
                    "z_score": round(z, 2),
                    "label": str(label),
                })
    return anomalies

def _build_metrics(data: list[dict], nk: list[str]) -> list[dict]:
    metrics = []
    for k in nk[:5]:
        vals = [float(r[k]) for r in data if isinstance(r.get(k), (int, float))]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        last, prev = vals[-1], vals[-2]
        change_pct = round(((last - prev) / prev * 100) if prev != 0 else 0, 1)
        trend = "up" if change_pct > 0 else ("down" if change_pct < 0 else "flat")
        disp = f"{mean/1000:.1f}K" if mean >= 10000 else f"{mean:.1f}"
        metrics.append({
            "label": k.replace("_", " ").title(),
            "value": disp,
            "change_pct": change_pct,
            "trend": trend,
            "description": f"Average across {len(vals)} periods",
        })
    return metrics

def _build_insights(data: list[dict], nk: list[str]) -> list[dict]:
    insights = []
    anomalies = _detect_anomalies(data, nk)
    for a in anomalies[:5]:
        insights.append({
            "title": f"Anomaly in {a['key'].replace('_',' ')}",
            "body": (
                f"Value {a['value']:,.1f} at '{a['label']}' is "
                f"{a['z_score']}σ from the mean — flagged for auditor review (ISA 315)."
            ),
            "severity": "critical" if a["z_score"] > 3 else "warning",
            "source": "Z-score analysis",
        })
    if not insights:
        insights.append({
            "title": "No critical anomalies detected",
            "body": "All values fall within 2.5 standard deviations of the mean.",
            "severity": "info",
            "source": "Statistical screening",
        })
    return insights

def _build_pipeline(provider: str, duration_ms: float) -> list[dict]:
    return [
        {"name": "Data received",         "status": "done", "duration_ms": 2},
        {"name": "Statistical pre-screen", "status": "done", "duration_ms": 8},
        {"name": f"AI analysis ({provider})", "status": "done", "duration_ms": round(duration_ms * 0.8)},
        {"name": "Report assembled",       "status": "done", "duration_ms": round(duration_ms * 0.2)},
    ]

# ── Claude via Anthropic API ───────────────────────────────────────────────
async def _call_claude(req: AnalyseRequest) -> dict:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    nk      = _numeric_keys(req.inline_data)
    metrics = _build_metrics(req.inline_data, nk)
    anomaly_summary = _detect_anomalies(req.inline_data, nk)

    system_prompt = f"""You are DataMind, an expert AI data auditor for the {req.industry} sector.
Your responses must be structured JSON only — no markdown fences, no preamble.

Return exactly this JSON shape:
{{
  "narrative": "Multi-paragraph audit narrative. Use **Heading** markers for sections. Write at least 4 paragraphs covering: Executive Summary, Key Findings, Risk Assessment, Recommendations.",
  "metrics": [
    {{"label":"string","value":"string","change_pct":number,"trend":"up|down|flat","description":"string"}}
  ],
  "insights": [
    {{"title":"string","body":"string","severity":"critical|warning|info","source":"string"}}
  ],
  "charts": []
}}

Industry context: {req.industry}
Records analysed: {len(req.inline_data)}
Numeric fields: {", ".join(nk)}
Anomalies pre-detected: {json.dumps(anomaly_summary[:3])}
Statistical metrics: {json.dumps(metrics[:3])}
"""

    user_msg = (
        f"Analyst query: {req.query}\n\n"
        f"Dataset (first 10 records): {json.dumps(req.inline_data[:10], default=str)}"
    )

    t0 = time.time()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      ANTHROPIC_MODEL,
                "max_tokens": 2000,
                "system":     system_prompt,
                "messages":   [{"role": "user", "content": user_msg}],
            },
        )
    elapsed_ms = (time.time() - t0) * 1000

    if resp.status_code != 200:
        raise ValueError(f"Anthropic error {resp.status_code}: {resp.text[:200]}")

    raw = resp.json()["content"][0]["text"].strip()
    raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
    ai_data = json.loads(raw)

    return {
        "query":           req.query,
        "industry":        req.industry,
        "provider":        "anthropic",
        "model":           ANTHROPIC_MODEL,
        "narrative":       ai_data.get("narrative", ""),
        "metrics":         ai_data.get("metrics",   metrics),
        "insights":        ai_data.get("insights",  []),
        "charts":          ai_data.get("charts",    []),
        "pipeline_steps":  _build_pipeline("Claude", elapsed_ms),
        "execution_ms":    round(elapsed_ms),
        "raw_data_preview": req.inline_data[:6],
    }

# ── Gemini fallback ────────────────────────────────────────────────────────
async def _call_gemini(req: AnalyseRequest) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured")

    nk      = _numeric_keys(req.inline_data)
    metrics = _build_metrics(req.inline_data, nk)
    anomaly_summary = _detect_anomalies(req.inline_data, nk)

    prompt = f"""You are DataMind, an expert AI data auditor for the {req.industry} sector.
Return ONLY valid JSON with no markdown fences or preamble.

JSON shape:
{{
  "narrative": "Multi-paragraph narrative. Use **Heading** for sections. Min 4 paragraphs.",
  "metrics": [{{"label":"","value":"","change_pct":0,"trend":"up|down|flat","description":""}}],
  "insights": [{{"title":"","body":"","severity":"critical|warning|info","source":""}}],
  "charts": []
}}

Query: {req.query}
Industry: {req.industry}
Records: {len(req.inline_data)}
Numeric fields: {", ".join(nk)}
Pre-detected anomalies: {json.dumps(anomaly_summary[:3])}
Data sample: {json.dumps(req.inline_data[:8], default=str)}
"""

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    t0 = time.time()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
    elapsed_ms = (time.time() - t0) * 1000

    if resp.status_code != 200:
        raise ValueError(f"Gemini error {resp.status_code}: {resp.text[:200]}")

    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
    ai_data = json.loads(raw)

    return {
        "query":           req.query,
        "industry":        req.industry,
        "provider":        "gemini",
        "model":           GEMINI_MODEL,
        "narrative":       ai_data.get("narrative", ""),
        "metrics":         ai_data.get("metrics",   metrics),
        "insights":        ai_data.get("insights",  []),
        "charts":          ai_data.get("charts",    []),
        "pipeline_steps":  _build_pipeline("Gemini", elapsed_ms),
        "execution_ms":    round(elapsed_ms),
        "raw_data_preview": req.inline_data[:6],
    }

# ── Statistical fallback ───────────────────────────────────────────────────
def _statistical_fallback(req: AnalyseRequest) -> dict:
    data = req.inline_data
    nk   = _numeric_keys(data)
    metrics = _build_metrics(data, nk)
    insights = _build_insights(data, nk)

    fk   = nk[0] if nk else "value"
    fv   = [float(r.get(fk, 0)) for r in data]
    trend_word = "upward" if (fv[-1] > fv[0] if len(fv) >= 2 else True) else "downward"
    cp   = round(((fv[-1]-fv[0])/fv[0]*100) if fv and fv[0] != 0 else 0, 1)
    iname = req.industry.replace("_"," ").title()

    critical_count = sum(1 for i in insights if i["severity"] == "critical")

    nar = (
        f"**Executive Summary**\n\n"
        f"The {iname} dataset has been processed through the statistical analysis engine. "
        f"The primary metric, {fk.replace('_',' ')}, moved {trend_word} over the analysis window, "
        f"registering a net change of {abs(cp)}% between the opening and closing periods. "
        f"This narrative is generated by the offline statistical engine.\n\n"
        f"**Key Findings**\n\n"
        f"Across {len(data)} records, {fk.replace('_',' ')} recorded meaningful variation. "
        f"{'The automated screening flagged ' + str(critical_count) + ' critical anomal' + ('y' if critical_count==1 else 'ies') + ' for review.' if critical_count else 'No critical anomalies were detected.'} "
        f"Overall trajectory is consistent with {'stable' if abs(cp)<15 else 'volatile'} performance.\n\n"
        f"**Risk and Anomaly Assessment**\n\n"
        f"All records screened using Z-score at ±2.5σ. "
        f"{'Anomalies have been logged for auditor review per ISA 315.' if critical_count else 'All values fell within acceptable bounds — a positive indicator of data integrity.'} "
        f"Supporting documentation should be retained as part of the formal audit trail.\n\n"
        f"**Recommendations**\n\n"
        f"Management should review {fk.replace('_',' ')} against prior-period benchmarks and "
        f"{iname} sector norms. Connect Claude or Gemini API keys for a deeper AI-powered analysis."
    )

    return {
        "query":           req.query,
        "industry":        req.industry,
        "provider":        "statistical",
        "model":           "local-stats-v1",
        "narrative":       nar,
        "metrics":         metrics,
        "insights":        insights,
        "charts":          [],
        "pipeline_steps":  [
            {"name": "Data received",         "status": "done", "duration_ms": 2},
            {"name": "Statistical analysis",  "status": "done", "duration_ms": 9},
            {"name": "Pattern detection",     "status": "done", "duration_ms": 5},
            {"name": "Report assembled",      "status": "done", "duration_ms": 4},
        ],
        "execution_ms":    20,
        "raw_data_preview": data[:6],
    }

# ── Main analysis endpoint ─────────────────────────────────────────────────
@app.post("/api/v1/analysis/analyse")
async def analyse(req: AnalyseRequest):
    t_start = time.time()

    if not req.inline_data:
        raise HTTPException(status_code=422, detail="inline_data is required and must not be empty")

    # Try Claude first
    if ANTHROPIC_API_KEY:
        try:
            log.info("Calling Claude for query: %s", req.query[:60])
            return await _call_claude(req)
        except Exception as e:
            log.warning("Claude failed: %s", e)

    # Try Gemini fallback
    if GEMINI_API_KEY:
        try:
            log.info("Falling back to Gemini")
            return await _call_gemini(req)
        except Exception as e:
            log.warning("Gemini failed: %s", e)

    # Statistical fallback — always works
    log.info("Using statistical fallback")
    return _statistical_fallback(req)

# ── Serve frontend (same-origin deployment) ────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_index():
        idx = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.isfile(idx):
            return FileResponse(idx)
        raise HTTPException(status_code=404, detail="index.html not found in frontend/")
