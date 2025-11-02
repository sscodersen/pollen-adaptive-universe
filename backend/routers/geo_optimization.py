from fastapi import APIRouter, Depends, Query
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel

from backend.database import get_db
from backend.services.geo_optimizer import geo_optimizer


router = APIRouter()


class OptimizationRequest(BaseModel):
    industry: str
    location: dict
    parameters: dict


@router.post("/optimize")
async def optimize_location(
    request: OptimizationRequest,
    db: Session = Depends(get_db)
):
    result = geo_optimizer.optimize_location(
        db,
        request.industry,
        request.location,
        request.parameters
    )
    return result


@router.get("/optimizations")
async def get_optimizations(
    industry: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db)
):
    optimizations = geo_optimizer.get_optimizations(db, industry, limit)
    return {"optimizations": optimizations, "count": len(optimizations)}


@router.post("/analyze")
async def stream_optimization_analysis(
    request: OptimizationRequest,
    db: Session = Depends(get_db)
):
    return EventSourceResponse(
        geo_optimizer.stream_optimization_analysis(
            db,
            request.industry,
            request.location,
            request.parameters
        )
    )


@router.get("/industries")
async def get_supported_industries():
    return {
        "industries": geo_optimizer.supported_industries,
        "count": len(geo_optimizer.supported_industries)
    }
