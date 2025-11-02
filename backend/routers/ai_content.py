from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel

from backend.database import get_db
from backend.services.ai_content_detector import ai_content_detector


router = APIRouter()


class ContentAnalysisRequest(BaseModel):
    content: str
    content_type: str = "text"
    metadata: Optional[dict] = None


class ContentVerificationRequest(BaseModel):
    is_ai_generated: bool
    verifier_notes: Optional[str] = None


@router.post("/analyze")
async def analyze_content(
    request: ContentAnalysisRequest,
    db: Session = Depends(get_db)
):
    result = ai_content_detector.analyze_content(
        db,
        request.content,
        request.content_type,
        request.metadata
    )
    return result


@router.post("/{content_id}/verify")
async def verify_content(
    content_id: str,
    verification: ContentVerificationRequest,
    db: Session = Depends(get_db)
):
    result = ai_content_detector.verify_content(
        db,
        content_id,
        verification.is_ai_generated,
        verification.verifier_notes
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Content not found")
    
    return result


@router.get("/detections")
async def get_detections(
    min_ai_probability: Optional[float] = Query(None, ge=0, le=100),
    verification_status: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db)
):
    detections = ai_content_detector.get_detections(
        db,
        min_ai_probability,
        verification_status,
        limit
    )
    return {"detections": detections, "count": len(detections)}


@router.get("/stats")
async def get_detection_stats(db: Session = Depends(get_db)):
    all_detections = ai_content_detector.get_detections(db, limit=1000)
    
    total = len(all_detections)
    ai_generated = sum(1 for d in all_detections if d["ai_generated_probability"] >= 75)
    human_generated = sum(1 for d in all_detections if d["ai_generated_probability"] < 25)
    uncertain = total - ai_generated - human_generated
    verified = sum(1 for d in all_detections if d["verified_by_human"])
    
    return {
        "total_analyses": total,
        "ai_generated_count": ai_generated,
        "human_generated_count": human_generated,
        "uncertain_count": uncertain,
        "verified_count": verified,
        "verification_rate": (verified / total * 100) if total > 0 else 0
    }
