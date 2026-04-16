"""Request and response schemas."""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class TimeseriesPoint(BaseModel):
    timestamp: float
    temperature: float


class PredictRequest(BaseModel):
    hvac_id: str
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timeseries_data: list[TimeseriesPoint]


class PredictResponse(BaseModel):
    hvac_id: str
    request_id: str
    label: str
    confidence: float
    risk_level: str
    predicted_at: datetime
    model_version: str


class BatchPredictRequest(BaseModel):
    records: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
