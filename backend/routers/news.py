from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.get("/fetch")
async def fetch_news(
    query: str = Query(...),
    categories: str = Query(None)
):
    """
    Stream news articles using SSE
    """
    async def generate():
        prompt = f"Find and summarize recent news about: {query}"
        if categories:
            prompt += f" in categories: {categories}"
        
        prompt += ". Provide unbiased news summaries from diverse sources."
        
        try:
            yield {"event": "message", "data": json.dumps({"text": "Fetching latest news...\n\n"})}
            async for chunk in ai_service.stream_response(prompt, {}):
                yield {"event": "message", "data": chunk}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            yield {"event": "done", "data": ""}
    
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
