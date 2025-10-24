from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.get("/advice")
async def health_advice(
    query: str = Query(...),
    focus_area: str = Query(None),
    goals: str = Query(None)
):
    """
    Stream health and wellness advice using SSE
    """
    async def generate():
        prompt = f"Health and wellness advice: {query}"
        if focus_area:
            prompt += f" focusing on {focus_area}"
        if goals:
            prompt += f" with goals: {goals}"
        
        prompt += ". Provide evidence-based health advice, wellness tips, and lifestyle recommendations. Disclaimer: This is general information, not medical advice."
        
        try:
            yield {"event": "message", "data": json.dumps({"text": "Gathering health information...\n\n"})}
            async for chunk in ai_service.stream_response(prompt, {}):
                yield {"event": "message", "data": chunk}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            yield {"event": "done", "data": ""}
    
    return EventSourceResponse(generate())

@router.get("/categories")
async def get_health_categories():
    return {
        "categories": [
            "Fitness",
            "Nutrition",
            "Mental Health",
            "Sleep",
            "Stress Management",
            "Preventive Care"
        ]
    }
