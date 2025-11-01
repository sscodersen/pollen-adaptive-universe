from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import shopping, travel, news, content, scraper, ai, smarthome, health, education, finance, code, feed, feed_v2, adaptive_intelligence, events, products, trends, playground, ask_ai
from backend.database import init_db
from backend.services.background_scraper import background_scraper
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

app = FastAPI(
    title="Pollen AI Platform - Bento Buzz Powered", 
    version="2.0.0", 
    description="Privacy-first AI platform with Bento Buzz scoring and Pollen AI training"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_ai.router, prefix="/api/ask-ai", tags=["ask-ai"])
app.include_router(feed_v2.router, prefix="/api/feed-v2", tags=["feed-v2-database"])
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
    print("✅ Database initialized")
    print("🚀 Pollen AI Platform ready - Bento Buzz scoring active")
    
    asyncio.create_task(background_scraper.start_periodic_scraping(interval_hours=24))
    print("⏰ Background scraper scheduled - runs every 24 hours")

@app.on_event("shutdown")
async def shutdown_event():
    background_scraper.stop()
    print("⏸️  Background scraper stopped")

@app.get("/")
async def root():
    last_run = background_scraper.last_run.isoformat() if background_scraper.last_run else "Never"
    return {
        "message": "Pollen AI Platform - Bento Buzz Powered", 
        "version": "2.0.0",
        "features": [
            "Bento Buzz 7-Factor Content Scoring",
            "Real-time Pollen AI Training",
            "Privacy-First Architecture",
            "High-Quality Content Curation",
            "Automated Daily Scraping"
        ],
        "endpoints": {
            "database_feed": "/api/feed-v2/posts",
            "stats": "/api/feed-v2/stats",
            "scrape": "/api/feed-v2/trigger-scrape"
        },
        "scraper": {
            "status": "running" if background_scraper.is_running else "stopped",
            "last_run": last_run
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "platform": "bento-buzz-powered",
        "scraper_active": background_scraper.is_running
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
