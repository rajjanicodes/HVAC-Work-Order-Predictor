"""API routes, request validation, and logging."""

import logging
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException

from src.config import MIN_READINGS, MODEL_VERSION, RISK_CRITICAL, RISK_HIGH, RISK_MEDIUM, TEMP_MIN, TEMP_MAX
from src.inference import predict
from src.schemas import BatchPredictRequest, BatchPredictResponse, PredictRequest, PredictResponse

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("monaire")

IST = timezone(timedelta(hours=5, minutes=30))  # Indian Standard Time offset

# App
app = FastAPI(
    title="MonAir HVAC Inference API",
    description="Predicts work-order need from IoT temperature time-series.",
    version="1.0.0",
)


# Helpers

def unix_to_ist(ts: float) -> str:
    """Format a unix timestamp as IST string."""
    return datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M:%S IST")


def risk_level(prob_wo: float) -> str:
    """prob(WO) -> risk tier. Uses prob_wo, not confidence - see README for why."""
    if prob_wo >= RISK_CRITICAL:
        return "CRITICAL"
    if prob_wo >= RISK_HIGH:
        return "HIGH"
    if prob_wo >= RISK_MEDIUM:
        return "MEDIUM"
    return "LOW"


def validate(req: PredictRequest) -> None:
    """Raises 422 if readings are too few or temperatures are out of physical bounds."""

    # Minimum reading count - too few points produce unreliable features
    if len(req.timeseries_data) < MIN_READINGS:
        raise HTTPException(
            status_code=422,
            detail=f"{req.hvac_id}: need >= {MIN_READINGS} readings, got {len(req.timeseries_data)}",
        )

    # Physical temperature bounds - values outside this range indicate a sensor fault
    for pt in req.timeseries_data:
        if not (TEMP_MIN <= pt.temperature <= TEMP_MAX):
            raise HTTPException(
                status_code=422,
                detail=f"{req.hvac_id}: temperature {pt.temperature} F at ts={pt.timestamp} is outside [{TEMP_MIN}, {TEMP_MAX}]",
            )


def build_response(req: PredictRequest, result: dict) -> PredictResponse:
    """Build the response object from a predict() result."""
    return PredictResponse(
        hvac_id=req.hvac_id,
        request_id=req.request_id,
        label=result["label"],
        confidence=result["confidence"],
        risk_level=risk_level(result["prob_wo"]),
        predicted_at=result["predicted_at"],
        model_version=MODEL_VERSION,
    )


# Routes

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_single(req: PredictRequest) -> PredictResponse:
    validate(req)

    # Log the incoming request - IST window shows the real-world time span of the sensor data
    timestamps = [pt.timestamp for pt in req.timeseries_data]
    logger.info(
        "[REQUEST]  hvac=%s  rid=%s  readings=%d  window=[%s -> %s]",
        req.hvac_id, req.request_id, len(timestamps), unix_to_ist(min(timestamps)), unix_to_ist(max(timestamps)),
    )

    # Run inference
    record = {"timeseries_data": [pt.model_dump() for pt in req.timeseries_data]}
    result = predict([record])[0]
    resp = build_response(req, result)

    # Log the prediction
    logger.info(
        "[RESPONSE] hvac=%s  rid=%s  label=%s  confidence=%.3f  risk=%s",
        req.hvac_id, req.request_id, resp.label, resp.confidence, resp.risk_level,
    )
    return resp


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:

    # Validate all records upfront - fail fast before any inference runs
    for r in req.records:
        validate(r)

    # Log the batch summary with IST window per unit
    for r in req.records:
        timestamps = [pt.timestamp for pt in r.timeseries_data]
        logger.info(
            "[BATCH REQUEST]  hvac=%s  rid=%s  readings=%d  window=[%s -> %s]",
            r.hvac_id, r.request_id, len(timestamps), unix_to_ist(min(timestamps)), unix_to_ist(max(timestamps)),
        )

    # Single model call across all records (vectorised - no Python loop over model)
    records = [{"timeseries_data": [pt.model_dump() for pt in r.timeseries_data]} for r in req.records]
    results = predict(records)
    responses = [build_response(r, res) for r, res in zip(req.records, results)]

    # Log each outcome
    for resp in responses:
        logger.info(
            "[BATCH RESPONSE] hvac=%s  label=%s  confidence=%.3f  risk=%s",
            resp.hvac_id, resp.label, resp.confidence, resp.risk_level,
        )

    return BatchPredictResponse(results=responses)
