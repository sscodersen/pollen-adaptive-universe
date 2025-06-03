
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model.pollen_model import PollenModel
    model = PollenModel()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = FastAPI(title="Pollen AI API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    input_text: str

class GenerateRequest(BaseModel):
    prompt: str
    mode: str = "social"

class FeedbackRequest(BaseModel):
    input_text: str
    expected_output: str

class SearchRequest(BaseModel):
    query_text: str

@app.get("/")
def root():
    return {"message": "Pollen AI API is running", "model_loaded": model is not None}

@app.post("/generate")
def generate_content(req: GenerateRequest):
    if not model:
        # Fallback content generation
        import random
        import time
        time.sleep(random.uniform(1, 2))
        
        fallback_content = {
            "social": [
                "🚀 Exploring the future of AI and human collaboration...",
                "💡 Just discovered an interesting pattern in data visualization",
                "🌟 Building something amazing with machine learning today"
            ],
            "news": [
                "Breaking: Revolutionary advancement in quantum computing announced",
                "Scientists develop new sustainable energy solution",
                "Tech industry leaders discuss ethical AI development"
            ],
            "entertainment": [
                "🎬 Interactive Story: The Digital Frontier Adventure",
                "🎮 Generated Game: Explore a world where AI and reality merge",
                "🎵 AI-Composed Symphony: 'Echoes of Tomorrow'"
            ]
        }
        
        content = random.choice(fallback_content.get(req.mode, fallback_content["social"]))
        
        return {
            "content": content,
            "confidence": random.uniform(0.7, 0.9),
            "learning": True,
            "reasoning": "Generated using fallback templates"
        }
    
    try:
        result = model.generate_content(req.prompt, req.mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(req: PredictRequest):
    if not model:
        return {"prediction": 0, "confidence": 0.5}
    
    try:
        logits, _ = model(req.input_text)
        prediction = int(logits.argmax(dim=1).item())
        confidence = float(torch.softmax(logits, dim=1).max().item())
        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn")
def learn(req: FeedbackRequest):
    if not model:
        return {"message": "Model not loaded"}
    
    try:
        model.learn_from_feedback(req.input_text, req.expected_output)
        return {"message": "Feedback learned successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reflect")
def reflect():
    if not model:
        return {"message": "Model not loaded"}
    
    try:
        model.reflect_and_update()
        return {"message": "Reflection completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search(req: SearchRequest):
    if not model:
        return {"matches": []}
    
    try:
        matches = model.semantic_search(req.query_text)
        return {"matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/stats")
def get_memory_stats():
    if not model:
        return {
            "episodic_count": 0,
            "long_term_keys": 0,
            "contextual_embeddings": 0,
            "recent_interactions": []
        }
    
    try:
        stats = model.get_memory_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
