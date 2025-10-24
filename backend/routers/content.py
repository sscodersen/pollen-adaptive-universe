from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from backend.models.schemas import ContentRequest
from backend.services.ai_service import ai_service

router = APIRouter()

@router.post("/generate")
async def generate_content(request: ContentRequest):
    """
    Stream AI-generated content using SSE
    """
    async def generate():
        prompt = f"Generate {request.content_type} content about: {request.query}"
        prompt += f" with a {request.tone} tone and {request.length} length."
        
        async for chunk in ai_service.stream_response(prompt, request.context or {}):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())

@router.get("/types")
async def get_content_types():
    return {
        "types": [
            {"id": "article", "name": "Article", "description": "Long-form written content"},
            {"id": "blog", "name": "Blog Post", "description": "Casual blog-style writing"},
            {"id": "social", "name": "Social Media", "description": "Short posts for social platforms"},
            {"id": "email", "name": "Email", "description": "Professional email content"},
            {"id": "marketing", "name": "Marketing Copy", "description": "Persuasive marketing content"}
        ]
    }
