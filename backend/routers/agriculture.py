from fastapi import APIRouter, Depends, Query
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel

from backend.database import get_db
from backend.services.agriculture_service import agriculture_service


router = APIRouter()


class FarmDataRecord(BaseModel):
    farm_id: str
    field_id: Optional[str] = None
    crop_type: str
    location: dict
    soil_data: Optional[dict] = None
    climate_data: Optional[dict] = None
    crop_health_score: Optional[float] = None
    growth_stage: Optional[str] = None
    estimated_yield: Optional[float] = None
    irrigation_level: Optional[float] = None
    fertilizer_data: Optional[dict] = None
    pest_detection: Optional[dict] = None
    metadata: Optional[dict] = None


class CropRecommendationRequest(BaseModel):
    farm_id: str
    soil_data: dict
    climate_data: dict
    field_id: Optional[str] = None


@router.post("/data")
async def record_farm_data(
    data: FarmDataRecord,
    db: Session = Depends(get_db)
):
    result = agriculture_service.record_farm_data(db, data.farm_id, data.dict())
    return result


@router.get("/data/{farm_id}")
async def get_farm_data(
    farm_id: str,
    field_id: Optional[str] = Query(None),
    hours: int = Query(168, le=720),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db)
):
    data = agriculture_service.get_farm_data(db, farm_id, field_id, hours, limit)
    return {"farm_id": farm_id, "data": data, "count": len(data)}


@router.post("/recommendations")
async def generate_crop_recommendation(
    request: CropRecommendationRequest,
    db: Session = Depends(get_db)
):
    recommendation = agriculture_service.generate_crop_recommendation(
        db,
        request.farm_id,
        request.soil_data,
        request.climate_data,
        request.field_id
    )
    return recommendation


@router.get("/analyze/{farm_id}")
async def stream_crop_analysis(
    farm_id: str,
    db: Session = Depends(get_db)
):
    return EventSourceResponse(
        agriculture_service.stream_crop_analysis(db, farm_id)
    )


@router.get("/crops")
async def get_crop_database():
    return {
        "crops": agriculture_service.crop_database,
        "count": len(agriculture_service.crop_database)
    }
