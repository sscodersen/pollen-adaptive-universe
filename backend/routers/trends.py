from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from ..services.enhanced_scraper import enhanced_scraper

router = APIRouter()

@router.get("/market")
async def get_market_trends(
    max_results: int = Query(15, ge=1, le=30, description="Maximum number of trends")
):
    """
    Stream market trends and predictions via SSE
    Aggregates from Exploding Topics, Hacker News, and other sources
    """
    async def generate():
        async for chunk in enhanced_scraper.stream_market_trends(
            max_results=max_results
        ):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())

@router.get("/health")
async def trends_health():
    return {"status": "operational", "service": "trends"}
