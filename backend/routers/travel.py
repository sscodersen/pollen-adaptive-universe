from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from backend.models.schemas import TravelRequest
from backend.services.ai_service import ai_service

router = APIRouter()

@router.post("/plan")
async def plan_trip(request: TravelRequest):
    """
    Stream travel recommendations and itinerary using SSE
    """
    async def generate():
        prompt = f"Help me plan a trip: {request.query}"
        if request.destination:
            prompt += f" to {request.destination}"
        if request.budget:
            prompt += f" with a budget of ${request.budget}"
        if request.travelers:
            prompt += f" for {request.travelers} travelers"
        
        prompt += ". Provide a detailed itinerary with recommendations for accommodations, activities, and dining."
        
        async for chunk in ai_service.stream_response(prompt, request.context or {}):
            yield {"event": "message", "data": chunk}
    
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
