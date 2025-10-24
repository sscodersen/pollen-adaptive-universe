from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from backend.models.schemas import ScraperRequest
from backend.services.scraper_service import scraper_service

router = APIRouter()

@router.post("/search")
async def scrape_search(request: ScraperRequest):
    """
    Stream web scraping results using SSE
    """
    async def generate():
        async for chunk in scraper_service.search_and_scrape(request.query, request.max_results or 5):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())

@router.get("/health")
async def scraper_health():
    return {"status": "operational", "service": "scraper"}
