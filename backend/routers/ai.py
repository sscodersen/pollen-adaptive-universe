from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.models.schemas import QueryRequest
from backend.services.ai_service import ai_service
import json
import asyncio

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

@router.get("/generate-comments")
async def generate_ai_comments(
    context: str = Query(..., description="Post context for generating comments"),
    count: int = Query(3, ge=1, le=10, description="Number of comments to generate")
):
    """
    Generate AI-powered comments for a post with SSE streaming
    """
    async def generate():
        try:
            tones = ['thoughtful', 'enthusiastic', 'critical', 'supportive', 'curious']
            authors = [
                'Alex Chen', 'Sarah Johnson', 'Michael Park', 'Emily Rodriguez', 
                'David Kim', 'Jessica Liu', 'Ryan Martinez', 'Sophia Anderson'
            ]
            
            for i in range(count):
                prompt = f"""Generate a {tones[i % len(tones)]} comment about this content: {context}

Make it:
- Natural and authentic
- {tones[i % len(tones)]} in tone
- 1-2 sentences
- Engaging and meaningful

Just return the comment text, nothing else."""
                
                comment_text = ""
                async for chunk in ai_service.stream_response(prompt, {"mode": "comment"}):
                    data = json.loads(chunk)
                    if "text" in data:
                        comment_text += data["text"]
                
                yield {"event": "message", "data": json.dumps({
                    "type": "comment",
                    "author": authors[i % len(authors)],
                    "text": comment_text.strip(),
                    "tone": tones[i % len(tones)]
                }) + "\n"}
                
                await asyncio.sleep(0.5)
            
            yield {"event": "message", "data": json.dumps({
                "type": "complete"
            }) + "\n"}
            
        except Exception as e:
            yield {"event": "message", "data": json.dumps({
                "type": "error",
                "error": str(e)
            }) + "\n"}
    
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
