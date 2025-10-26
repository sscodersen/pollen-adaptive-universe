from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.get("/help")
async def get_code_help(
    query: str = Query(...),
):
    """
    Stream coding help using SSE
    """
    async def generate():
        prompt = f"Coding help request: {query}. Provide code review, debugging assistance, optimization suggestions, and development tips."
        
        try:
            yield {"event": "message", "data": json.dumps({"text": "Analyzing your code...\n\n"})}
            async for chunk in ai_service.stream_response(prompt, {}):
                yield {"event": "message", "data": chunk}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            yield {"event": "done", "data": ""}
    
    return EventSourceResponse(generate())
