from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
from backend.services.enhanced_scraper import enhanced_scraper
from typing import Optional
import json

router = APIRouter()

@router.get("/fetch")
async def fetch_news(
    category: Optional[str] = Query(None, description="Filter by category (technology, business, etc.)"),
    min_score: float = Query(50.0, ge=0, le=100, description="Minimum quality score"),
    max_results: int = Query(20, ge=1, le=50, description="Maximum number of articles")
):
    """
    Stream news articles from multiple sources via SSE
    Sources: BBC News, TechCrunch, Hacker News, Reuters
    """
    async def generate():
        async for chunk in enhanced_scraper.stream_curated_content(
            category=category,
            min_score=min_score,
            max_results=max_results
        ):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())

@router.get("/categories")
async def get_news_categories():
    return {
        "categories": [
            "Technology",
            "Business",
            "Science",
            "Health",
            "Politics",
            "Entertainment",
            "Sports"
        ]
    }
