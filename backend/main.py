from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import shopping, travel, news, content, scraper, ai
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="New Frontier AI Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(shopping.router, prefix="/api/shopping", tags=["shopping"])
app.include_router(travel.router, prefix="/api/travel", tags=["travel"])
app.include_router(news.router, prefix="/api/news", tags=["news"])
app.include_router(content.router, prefix="/api/content", tags=["content"])
app.include_router(scraper.router, prefix="/api/scraper", tags=["scraper"])
app.include_router(ai.router, prefix="/api/ai", tags=["ai"])

@app.get("/")
async def root():
    return {"message": "New Frontier AI Platform API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
