
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .routes import router
import uvicorn

app = FastAPI(
    title="Pollen AI Platform",
    description="Self-evolving AI platform with Absolute Zero Reasoner",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Pollen AI Platform - Ready"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
