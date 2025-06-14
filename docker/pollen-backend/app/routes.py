
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Optional, Any
from datetime importdatetime
import time
import asyncio

from ..pollen_model import PollenLLMX
from .utils import get_user_session, process_request, format_response

router = APIRouter()
pollen_ai: Optional[PollenLLMX] = None

@router.on_event("startup")
async def startup_event():
    """Initializes the PollenLLMX model on application startup."""
    global pollen_ai
    print("🌸 Initializing Pollen LLMX with Absolute Zero Reasoner...")
    pollen_ai = PollenLLMX()
    # The reasoning loop starts automatically within PollenLLMX
    print("✅ Pollen LLMX Platform ready!")

class GenerateRequest(BaseModel):
    prompt: str
    mode: str
    context: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    content: str
    confidence: float
    reasoning: Optional[str] = None
    metadata: Dict[str, Any]

@router.get("/health")
async def health_check():
    """Checks the health of the AI service."""
    return {
        "status": "healthy",
        "model_loaded": pollen_ai is not None,
        "model_version": pollen_ai.version if pollen_ai else "N/A",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/generate", response_model=GenerateResponse)
async def generate_content(
    request: GenerateRequest,
    user_session: str = Depends(get_user_session)
):
    """Generates content using the PollenLLMX model."""
    if not pollen_ai:
        raise HTTPException(status_code=503, detail="Pollen AI not initialized")

    try:
        start_time = time.time()
        
        response = await pollen_ai.generate(
            prompt=request.prompt,
            mode=request.mode,
            context=request.context,
            user_session=user_session
        )
        
        generation_time = time.time() - start_time
        
        return format_response(response, generation_time, user_session)
        
    except Exception as e:
        print(f"Generation error: {e}")
        # Provide a more informative error response
        error_response = {
            "content": f"An error occurred during generation: {e}",
            "confidence": 0.0,
            "reasoning": "Error in generation pipeline",
            "metadata": {"error": True}
        }
        return format_response(error_response, time.time() - start_time, user_session)

@router.get("/reasoning/stats")
async def get_reasoning_stats():
    """Gets statistics from the Absolute Zero Reasoner."""
    if not pollen_ai:
        raise HTTPException(status_code=503, detail="Pollen AI not initialized")
    
    try:
        return pollen_ai.get_reasoning_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
