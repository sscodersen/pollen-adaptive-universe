from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.get("/generate")
async def generate_content(
    query: str = Query(...),
    content_type: str = Query("article"),
    tone: str = Query("professional"),
    length: str = Query("medium")
):
    """
    Stream AI-generated content using SSE
    """
    async def generate():
        prompt = f"Generate {content_type} content about: {query}"
        prompt += f" with a {tone} tone and {length} length."
        
        try:
            yield {"event": "message", "data": json.dumps({"text": f"Creating {content_type} content...\n\n"})}
            async for chunk in ai_service.stream_response(prompt, {}):
                yield {"event": "message", "data": chunk}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            yield {"event": "done", "data": ""}
    
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
