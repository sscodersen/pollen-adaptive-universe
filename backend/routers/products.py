from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from typing import Optional
from ..services.enhanced_scraper import enhanced_scraper

router = APIRouter()

@router.get("/discover")
async def discover_products(
    category: Optional[str] = Query(None, description="Filter by category (software, productivity, etc.)"),
    min_score: float = Query(50.0, ge=0, le=100, description="Minimum quality score"),
    max_results: int = Query(20, ge=1, le=50, description="Maximum number of products")
):
    """
    Stream quality product recommendations via SSE
    """
    async def generate():
        async for chunk in enhanced_scraper.stream_products(
            category=category,
            min_score=min_score,
            max_results=max_results
        ):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())

@router.get("/categories")
async def get_product_categories():
    return {
        "categories": [
            "Software",
            "Productivity",
            "Security",
            "Hardware",
            "Mobile Apps",
            "Web Services",
            "Developer Tools"
        ]
    }

@router.get("/health")
async def products_health():
    return {"status": "operational", "service": "products"}
