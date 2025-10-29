from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from typing import Optional
from ..services.enhanced_scraper import enhanced_scraper

router = APIRouter()

@router.get("/upcoming")
async def get_upcoming_events(
    category: Optional[str] = Query(None, description="Filter by category (technology, business, etc.)"),
    max_results: int = Query(20, ge=1, le=50, description="Maximum number of events")
):
    """
    Stream upcoming events via SSE
    """
    async def generate():
        async for chunk in enhanced_scraper.stream_events(
            category=category,
            max_results=max_results
        ):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())

@router.get("/categories")
async def get_event_categories():
    return {
        "categories": [
            "Technology",
            "Business",
            "Science",
            "Arts & Culture",
            "Health & Wellness",
            "Sports",
            "Education"
        ]
    }

@router.get("/health")
async def events_health():
    return {"status": "operational", "service": "events"}
