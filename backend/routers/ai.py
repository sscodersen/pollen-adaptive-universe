from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from backend.models.schemas import QueryRequest
from backend.services.ai_service import ai_service

router = APIRouter()

@router.post("/chat")
async def ai_chat(request: QueryRequest):
    """
    General AI chat endpoint with SSE streaming
    """
    async def generate():
        async for chunk in ai_service.stream_response(request.query, request.context or {}):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())

@router.get("/models")
async def get_available_models():
    return {
        "models": [
            {"id": "pollen-ai", "name": "Pollen AI", "description": "Custom AI model"},
            {"id": "gpt-4", "name": "GPT-4", "description": "OpenAI GPT-4"},
            {"id": "scraper-fallback", "name": "Web Scraper", "description": "Fallback to web scraping"}
        ]
    }
