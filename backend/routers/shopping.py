from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from backend.models.schemas import ShoppingRequest
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.post("/search")
async def shopping_search(request: ShoppingRequest):
    """
    Stream shopping recommendations using SSE
    """
    async def generate():
        prompt = f"Help me find {request.query}"
        if request.budget:
            prompt += f" within a budget of ${request.budget}"
        if request.category:
            prompt += f" in the {request.category} category"
        
        prompt += ". Provide detailed product recommendations with pros and cons."
        
        async for chunk in ai_service.stream_response(prompt, request.context or {}):
            yield {"event": "message", "data": chunk}
    
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
