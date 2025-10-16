#!/usr/bin/env python3
"""
SSE Worker Bot for Exploding Topics Trend Scraping
Continuously scrapes trends and streams them to Pollen AI for processing
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Pollen Trend Scraper SSE", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
POLLEN_AI_URL = "http://localhost:8000"
SCRAPE_INTERVAL = 60  # Scrape every 60 seconds
TREND_CATEGORIES = [
    "technology", "business", "health", "finance", "entertainment",
    "education", "marketing", "ecommerce", "ai-ml", "crypto"
]

class TrendData(BaseModel):
    id: str
    topic: str
    category: str
    score: float
    growth_rate: float
    search_volume: int
    timestamp: str
    source: str = "Exploding Topics"
    keywords: List[str] = []
    description: Optional[str] = None


class TrendScraperSSE:
    """SSE Worker Bot for real-time trend scraping"""
    
    def __init__(self):
        self.active_connections = 0
        self.trends_cache: List[TrendData] = []
        self.last_scrape = 0
        self.pollen_ai_available = False
    
    async def check_pollen_ai(self):
        """Check if Pollen AI backend is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{POLLEN_AI_URL}/health", timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status == 200:
                        self.pollen_ai_available = True
                        print("✅ Trend Scraper: Connected to Pollen AI")
                        return True
        except Exception as e:
            print(f"⚠️ Trend Scraper: Pollen AI not available - {e}")
            self.pollen_ai_available = False
        return False
    
    async def scrape_exploding_topics(self) -> List[Dict[str, Any]]:
        """
        Scrape trends from Exploding Topics
        Note: This is a simulated scraper. In production, you would use actual API or web scraping.
        """
        trends = []
        timestamp = datetime.utcnow().isoformat()
        
        # Simulated trending topics (in production, replace with actual API call)
        simulated_trends = [
            {
                "topic": "AI Code Assistants",
                "category": "ai-ml",
                "growth_rate": 285.5,
                "search_volume": 125000,
                "keywords": ["AI", "coding", "automation", "development"],
                "description": "AI-powered code generation and assistance tools seeing massive adoption"
            },
            {
                "topic": "Sustainable Packaging",
                "category": "business",
                "growth_rate": 178.3,
                "search_volume": 89000,
                "keywords": ["sustainability", "eco-friendly", "packaging", "green"],
                "description": "Eco-friendly packaging solutions trending across industries"
            },
            {
                "topic": "Mental Health Apps",
                "category": "health",
                "growth_rate": 245.7,
                "search_volume": 156000,
                "keywords": ["mental health", "wellness", "therapy", "meditation"],
                "description": "Digital mental health platforms experiencing rapid growth"
            },
            {
                "topic": "DeFi Insurance",
                "category": "crypto",
                "growth_rate": 312.4,
                "search_volume": 67000,
                "keywords": ["DeFi", "insurance", "blockchain", "crypto"],
                "description": "Decentralized finance insurance protocols gaining traction"
            },
            {
                "topic": "No-Code Platforms",
                "category": "technology",
                "growth_rate": 198.6,
                "search_volume": 203000,
                "keywords": ["no-code", "low-code", "automation", "development"],
                "description": "No-code development platforms democratizing software creation"
            }
        ]
        
        for idx, trend_data in enumerate(simulated_trends):
            trend = TrendData(
                id=f"et-{int(time.time())}-{idx}",
                topic=trend_data["topic"],
                category=trend_data["category"],
                score=min(100, 50 + (trend_data["growth_rate"] / 5)),
                growth_rate=trend_data["growth_rate"],
                search_volume=trend_data["search_volume"],
                timestamp=timestamp,
                keywords=trend_data["keywords"],
                description=trend_data.get("description")
            )
            trends.append(trend.dict())
        
        return trends
    
    async def process_with_pollen_ai(self, trend: Dict[str, Any]) -> Dict[str, Any]:
        """Send trend to Pollen AI for enhanced processing"""
        if not self.pollen_ai_available:
            return trend
        
        try:
            async with aiohttp.ClientSession() as session:
                prompt = f"Analyze this trending topic and provide market insights: {trend['topic']}. Current growth: {trend['growth_rate']}%. Keywords: {', '.join(trend['keywords'])}."
                
                payload = {
                    "prompt": prompt,
                    "mode": "analysis",
                    "type": "trends",
                    "context": {
                        "category": trend["category"],
                        "growth_rate": trend["growth_rate"]
                    },
                    "use_cache": True,
                    "compression_level": "medium"
                }
                
                async with session.post(
                    f"{POLLEN_AI_URL}/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        ai_response = await response.json()
                        trend["ai_insights"] = ai_response.get("content", "")
                        trend["ai_confidence"] = ai_response.get("confidence", 0.0)
                        print(f"✅ Enhanced trend '{trend['topic']}' with Pollen AI")
        except Exception as e:
            print(f"⚠️ Failed to enhance trend with Pollen AI: {e}")
        
        return trend
    
    async def trend_stream(self):
        """Generate SSE stream of trending topics"""
        self.active_connections += 1
        print(f"📡 SSE Client connected (total: {self.active_connections})")
        
        try:
            # Check Pollen AI availability
            await self.check_pollen_ai()
            
            # Send initial connection message
            yield {
                "event": "connected",
                "data": json.dumps({
                    "message": "Trend Scraper SSE Connected",
                    "pollen_ai_available": self.pollen_ai_available,
                    "timestamp": datetime.utcnow().isoformat()
                })
            }
            
            while True:
                current_time = time.time()
                
                # Scrape trends if interval has passed
                if current_time - self.last_scrape >= SCRAPE_INTERVAL:
                    yield {
                        "event": "scraping",
                        "data": json.dumps({
                            "message": "Scraping Exploding Topics...",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    }
                    
                    # Scrape new trends
                    raw_trends = await self.scrape_exploding_topics()
                    
                    # Process each trend with Pollen AI
                    processed_trends = []
                    for trend in raw_trends:
                        enhanced_trend = await self.process_with_pollen_ai(trend)
                        processed_trends.append(enhanced_trend)
                        
                        # Stream individual trend updates
                        yield {
                            "event": "trend_update",
                            "data": json.dumps(enhanced_trend)
                        }
                        
                        await asyncio.sleep(0.5)  # Small delay between trends
                    
                    # Update cache
                    self.trends_cache = processed_trends
                    self.last_scrape = current_time
                    
                    # Send batch complete message
                    yield {
                        "event": "batch_complete",
                        "data": json.dumps({
                            "message": f"Scraped {len(processed_trends)} trends",
                            "trends_count": len(processed_trends),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    }
                
                # Keep-alive ping every 15 seconds
                yield {
                    "event": "heartbeat",
                    "data": json.dumps({
                        "timestamp": datetime.utcnow().isoformat(),
                        "cache_size": len(self.trends_cache)
                    })
                }
                
                await asyncio.sleep(15)
                
        finally:
            self.active_connections -= 1
            print(f"📡 SSE Client disconnected (remaining: {self.active_connections})")


# Global scraper instance
scraper = TrendScraperSSE()


@app.get("/trends/stream")
async def stream_trends():
    """SSE endpoint for real-time trend streaming"""
    return StreamingResponse(
        scraper.trend_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/trends/latest")
async def get_latest_trends():
    """Get latest scraped trends (non-SSE)"""
    if not scraper.trends_cache:
        # Scrape immediately if cache is empty
        trends = await scraper.scrape_exploding_topics()
        # Process with Pollen AI
        processed = []
        for trend in trends:
            enhanced = await scraper.process_with_pollen_ai(trend)
            processed.append(enhanced)
        scraper.trends_cache = processed
    
    return {
        "trends": scraper.trends_cache,
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(scraper.trends_cache),
        "pollen_ai_available": scraper.pollen_ai_available
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    await scraper.check_pollen_ai()
    
    return {
        "status": "healthy",
        "service": "Trend Scraper SSE",
        "active_connections": scraper.active_connections,
        "cached_trends": len(scraper.trends_cache),
        "pollen_ai_available": scraper.pollen_ai_available,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Pollen Trend Scraper SSE Service")
    print("📡 Real-time trend streaming from Exploding Topics")
    print("🧠 Integrated with Pollen AI for enhanced insights")
    uvicorn.run(app, host="0.0.0.0", port=8099)
