from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from backend.models.schemas import NewsRequest
from backend.services.ai_service import ai_service
from backend.services.scraper_service import scraper_service

router = APIRouter()

@router.post("/fetch")
async def fetch_news(request: NewsRequest):
    """
    Stream news articles using SSE (with scraper fallback)
    """
    async def generate():
        prompt = f"Find and summarize recent news about: {request.query}"
        if request.categories:
            prompt += f" in categories: {', '.join(request.categories)}"
        
        prompt += ". Provide unbiased news summaries from diverse sources."
        
        try:
            async for chunk in ai_service.stream_response(prompt, request.context or {}):
                yield {"event": "ai_response", "data": chunk}
        except Exception:
            async for chunk in scraper_service.search_and_scrape(request.query, 5):
                yield {"event": "scraped_content", "data": chunk}
    
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
