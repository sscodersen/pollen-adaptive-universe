
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import time

from ..pollen_model import PollenLLMX
from .utils import get_user_session, process_request, format_response

router = APIRouter()
security = HTTPBearer(auto_error=False)

# Global instances
pollen_ai: Optional[PollenLLMX] = None

@router.on_event("startup")
async def startup_event():
    global pollen_ai
    print("🌸 Initializing Pollen AI Platform...")
    
    pollen_ai = PollenLLMX()
    
    print("✅ Pollen AI Platform ready!")

# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str
    mode: str
    context: Optional[Dict[str, Any]] = None
    stream: bool = False

class GenerateResponse(BaseModel):
    content: str
    confidence: float
    reasoning: Optional[str] = None
    metadata: Dict[str, Any]

# NOTE: The memory stats below are part of the old model architecture.
# They are being temporarily commented out until new endpoints for the 
# PollenLLMX model's learning and adaptation stats are created.
# class MemoryStats(BaseModel):
#     total_interactions: int
#     learning_tasks: int
#     success_rate: float
#     recent_performance: float

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pollen_ai is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

@router.post("/generate", response_model=GenerateResponse)
async def generate_content(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    user_session: str = Depends(get_user_session)
):
    try:
        if not pollen_ai:
            raise HTTPException(status_code=503, detail="Pollen AI not loaded")
        
        # Process request
        processed_request = process_request(request, user_session)
        
        # Generate response
        start_time = time.time()
        response = await pollen_ai.generate(
            prompt=processed_request["prompt"],
            mode=processed_request["mode"],
            context=processed_request["context"],
            user_session=user_session
        )
        
        generation_time = time.time() - start_time
        
        # NOTE: Background memory updates are temporarily disabled as they
        # were tied to the old model's memory system. The new PollenLLMX
        # handles its own learning and memory internally.
        # background_tasks.add_task(
        #     update_memory_async,
        #     user_session,
        #     request.prompt,
        #     response["content"],
        #     request.mode,
        #     generation_time
        # )
        
        return format_response(response, generation_time, user_session)
        
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @router.get("/memory/stats", response_model=MemoryStats)
# async def get_memory_stats(user_session: str = Depends(get_user_session)):
#     try:
#         if not memory_manager:
#             raise HTTPException(status_code=503, detail="Memory manager not available")
        
#         stats = memory_manager.get_user_stats(user_session)
#         return MemoryStats(**stats)
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/memory/clear")
# async def clear_memory(user_session: str = Depends(get_user_session)):
#     try:
#         if not memory_manager:
#             raise HTTPException(status_code=503, detail="Memory manager not available")
        
#         memory_manager.clear_user_memory(user_session)
#         return {"message": "Memory cleared successfully"}
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.get("/reasoning/stats")
async def get_reasoning_stats(user_session: str = Depends(get_user_session)):
    try:
        if not pollen_ai:
            raise HTTPException(status_code=503, detail="Pollen AI not available")
        
        stats = pollen_ai.reasoner.get_reasoning_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# async def update_memory_async(user_session: str, prompt: str, response: str, mode: str, generation_time: float):
#     try:
#         if memory_manager:
#             memory_manager.add_interaction(
#                 user_session=user_session,
#                 input_text=prompt,
#                 output_text=response,
#                 mode=mode,
#                 metadata={"generation_time": generation_time}
#             )
#     except Exception as e:
#         print(f"Memory update error: {e}")
