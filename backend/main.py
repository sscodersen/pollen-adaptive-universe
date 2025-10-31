from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import shopping, travel, news, content, scraper, ai, smarthome, health, education, finance, code, feed, adaptive_intelligence, events, products, trends, playground
from backend.database import init_db
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="New Frontier AI Platform", version="1.0.0", description="Privacy-first anonymous AI assistant platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(feed.router, prefix="/api/feed", tags=["feed"])
app.include_router(adaptive_intelligence.router, prefix="/api/adaptive-intelligence", tags=["adaptive-intelligence"])
app.include_router(playground.router, prefix="/api/playground", tags=["playground"])
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(products.router, prefix="/api/products", tags=["products"])
app.include_router(trends.router, prefix="/api/trends", tags=["trends"])
app.include_router(shopping.router, prefix="/api/shopping", tags=["shopping"])
app.include_router(travel.router, prefix="/api/travel", tags=["travel"])
app.include_router(news.router, prefix="/api/news", tags=["news"])
app.include_router(content.router, prefix="/api/content", tags=["content"])
app.include_router(scraper.router, prefix="/api/scraper", tags=["scraper"])
app.include_router(ai.router, prefix="/api/ai", tags=["ai"])
app.include_router(smarthome.router, prefix="/api/smarthome", tags=["smarthome"])
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(education.router, prefix="/api/education", tags=["education"])
app.include_router(finance.router, prefix="/api/finance", tags=["finance"])
app.include_router(code.router, prefix="/api/code", tags=["code"])

@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/")
async def root():
    return {"message": "New Frontier AI Platform API - Privacy First, No Sign-in Required", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
