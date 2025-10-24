from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.get("/search")
async def shopping_search(
    query: str = Query(...),
    budget: float = Query(None),
    category: str = Query(None),
    user_id: str = Query(None)
):
    """
    Stream shopping recommendations using SSE (GET request for EventSource compatibility)
    """
    async def generate():
        prompt = f"Help me find {query}"
        if budget:
            prompt += f" within a budget of ${budget}"
        if category:
            prompt += f" in the {category} category"
        
        prompt += ". Provide detailed product recommendations with pros and cons."
        
        try:
            yield {"event": "message", "data": json.dumps({"text": "Searching for the best products...\n\n"})}
            
            if not ai_service.api_key and not ai_service.model_url:
                yield {"event": "message", "data": json.dumps({"text": "AI model not configured. Using mock data.\n\n"})}
                yield {"event": "message", "data": json.dumps({"text": f"Based on your query '{query}':\n\n"})}
                yield {"event": "message", "data": json.dumps({"text": "1. Product A - Highly rated with excellent reviews\n"})}
                yield {"event": "message", "data": json.dumps({"text": "2. Product B - Best value for money\n"})}
                yield {"event": "message", "data": json.dumps({"text": "3. Product C - Premium quality option\n"})}
            else:
                async for chunk in ai_service.stream_response(prompt, {}):
                    yield {"event": "message", "data": chunk}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            yield {"event": "done", "data": ""}
    
    return EventSourceResponse(generate())

@router.get("/categories")
async def get_categories():
    return {
        "categories": [
            "Electronics",
            "Fashion",
            "Home & Garden",
            "Sports & Outdoors",
            "Books & Media",
            "Health & Beauty"
        ]
    }
