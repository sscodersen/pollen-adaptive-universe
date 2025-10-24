from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.get("/plan")
async def plan_trip(
    query: str = Query(...),
    destination: str = Query(None),
    budget: float = Query(None),
    travelers: int = Query(1)
):
    """
    Stream travel recommendations and itinerary using SSE
    """
    async def generate():
        prompt = f"Help me plan a trip: {query}"
        if destination:
            prompt += f" to {destination}"
        if budget:
            prompt += f" with a budget of ${budget}"
        if travelers:
            prompt += f" for {travelers} travelers"
        
        prompt += ". Provide a detailed itinerary with recommendations for accommodations, activities, and dining."
        
        try:
            yield {"event": "message", "data": json.dumps({"text": "Planning your perfect trip...\n\n"})}
            async for chunk in ai_service.stream_response(prompt, {}):
                yield {"event": "message", "data": chunk}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            yield {"event": "done", "data": ""}
    
    return EventSourceResponse(generate())

@router.get("/destinations")
async def get_popular_destinations():
    return {
        "destinations": [
            {"name": "Paris, France", "type": "City"},
            {"name": "Tokyo, Japan", "type": "City"},
            {"name": "Bali, Indonesia", "type": "Beach"},
            {"name": "Swiss Alps", "type": "Mountain"},
            {"name": "Iceland", "type": "Adventure"}
        ]
    }
