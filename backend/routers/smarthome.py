from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.get("/control")
async def control_devices(
    query: str = Query(...),
    devices: str = Query(None),
    room: str = Query(None)
):
    """
    Stream smart home control suggestions using SSE
    """
    async def generate():
        prompt = f"Smart home assistance: {query}"
        if devices:
            prompt += f" for devices: {devices}"
        if room:
            prompt += f" in the {room}"
        
        prompt += ". Provide smart home automation suggestions, device control instructions, and energy-saving tips."
        
        try:
            yield {"event": "message", "data": json.dumps({"text": "Analyzing your smart home setup...\n\n"})}
            async for chunk in ai_service.stream_response(prompt, {}):
                yield {"event": "message", "data": chunk}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            yield {"event": "done", "data": ""}
    
    return EventSourceResponse(generate())

@router.get("/devices")
async def get_device_types():
    return {
        "devices": [
            {"type": "lights", "name": "Smart Lights"},
            {"type": "thermostat", "name": "Thermostat"},
            {"type": "locks", "name": "Smart Locks"},
            {"type": "cameras", "name": "Security Cameras"},
            {"type": "speakers", "name": "Smart Speakers"},
            {"type": "appliances", "name": "Smart Appliances"}
        ]
    }
