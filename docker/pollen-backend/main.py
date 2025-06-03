
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import torch
import json
import time
import uuid
import os
from datetime import datetime
import asyncio
import numpy as np

# Pollen Model Components
from pollen_model import PollenLLMX
from memory_engine import MemoryEngine
from learning_engine import LearningEngine

app = FastAPI(title="Pollen LLMX API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global model instance
pollen_model = None
memory_engine = None
learning_engine = None

@app.on_event("startup")
async def startup_event():
    global pollen_model, memory_engine, learning_engine
    
    print("🌸 Initializing Pollen LLMX...")
    
    # Initialize memory engine
    memory_engine = MemoryEngine()
    
    # Initialize learning engine
    learning_engine = LearningEngine()
    
    # Initialize or load Pollen model
    model_path = os.getenv("POLLEN_MODEL_PATH", "./models/pollen_llmx.pth")
    
    if os.path.exists(model_path):
        print(f"📦 Loading existing Pollen model from {model_path}")
        pollen_model = PollenLLMX.load(model_path)
    else:
        print("🆕 Creating new Pollen model - starting from zero knowledge")
        pollen_model = PollenLLMX()
        
        # Save initial model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pollen_model.save(model_path)
    
    print("✅ Pollen LLMX initialized and ready!")

# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str
    mode: str
    context: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    content: str
    confidence: float
    learning: bool
    memoryUpdated: bool
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MemoryStats(BaseModel):
    shortTermSize: int
    longTermPatterns: int
    topPatterns: List[Dict[str, Any]]
    isLearning: bool

# Helper function to get user session
def get_user_session(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if credentials:
        return credentials.credentials
    return f"anon-{uuid.uuid4().hex[:8]}"

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pollen_model is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_content(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    user_session: str = Depends(get_user_session)
):
    try:
        if not pollen_model:
            raise HTTPException(status_code=503, detail="Pollen model not loaded")
        
        # Extract memory context
        memory_context = request.memory or {}
        
        # Generate response using Pollen
        start_time = time.time()
        
        response = await pollen_model.generate(
            prompt=request.prompt,
            mode=request.mode,
            context=request.context,
            memory_context=memory_context,
            user_session=user_session
        )
        
        generation_time = time.time() - start_time
        
        # Update memory in background
        background_tasks.add_task(
            update_memory_async,
            user_session,
            request.prompt,
            response["content"],
            request.mode,
            generation_time
        )
        
        # Update learning patterns in background
        background_tasks.add_task(
            update_learning_async,
            user_session,
            request.prompt,
            response,
            request.mode
        )
        
        return GenerateResponse(
            content=response["content"],
            confidence=response.get("confidence", 0.8),
            learning=True,
            memoryUpdated=True,
            reasoning=response.get("reasoning"),
            metadata={
                "generation_time": generation_time,
                "model_version": pollen_model.version,
                "user_session": user_session[:8] + "..."
            }
        )
        
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/stats", response_model=MemoryStats)
async def get_memory_stats(user_session: str = Depends(get_user_session)):
    try:
        if not memory_engine:
            raise HTTPException(status_code=503, detail="Memory engine not available")
        
        stats = memory_engine.get_user_stats(user_session)
        
        return MemoryStats(
            shortTermSize=stats["short_term_size"],
            longTermPatterns=stats["long_term_patterns"],
            topPatterns=stats["top_patterns"],
            isLearning=stats["is_learning"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/clear")
async def clear_memory(user_session: str = Depends(get_user_session)):
    try:
        if not memory_engine:
            raise HTTPException(status_code=503, detail="Memory engine not available")
        
        memory_engine.clear_user_memory(user_session)
        
        return {"message": "Memory cleared successfully", "user_session": user_session[:8] + "..."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning/toggle")
async def toggle_learning(user_session: str = Depends(get_user_session)):
    try:
        if not learning_engine:
            raise HTTPException(status_code=503, detail="Learning engine not available")
        
        new_state = learning_engine.toggle_learning(user_session)
        
        return {
            "learning_enabled": new_state,
            "message": f"Learning {'enabled' if new_state else 'disabled'}",
            "user_session": user_session[:8] + "..."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def update_memory_async(user_session: str, prompt: str, response: str, mode: str, generation_time: float):
    try:
        if memory_engine:
            memory_engine.add_interaction(
                user_session=user_session,
                input_text=prompt,
                output_text=response,
                mode=mode,
                metadata={"generation_time": generation_time}
            )
    except Exception as e:
        print(f"Memory update error: {e}")

async def update_learning_async(user_session: str, prompt: str, response: dict, mode: str):
    try:
        if learning_engine and pollen_model:
            await learning_engine.process_interaction(
                model=pollen_model,
                user_session=user_session,
                prompt=prompt,
                response=response,
                mode=mode
            )
    except Exception as e:
        print(f"Learning update error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
