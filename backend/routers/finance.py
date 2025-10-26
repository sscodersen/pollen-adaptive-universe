from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.get("/advice")
async def get_finance_advice(
    query: str = Query(...),
):
    """
    Stream financial advice using SSE
    """
    async def generate():
        prompt = f"Financial advice request: {query}. Provide budget planning, investment insights, savings tips, and financial guidance."
        
        try:
            yield {"event": "message", "data": json.dumps({"text": "Analyzing your financial query...\n\n"})}
            async for chunk in ai_service.stream_response(prompt, {}):
                yield {"event": "message", "data": chunk}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            yield {"event": "done", "data": ""}
    
    return EventSourceResponse(generate())
