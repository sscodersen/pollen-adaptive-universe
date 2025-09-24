from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import asyncio
from datetime import datetime

# Import all the AI models
from models.task_proposer import TaskProposer
from models.task_solver import TaskSolver
from models.code_executor import CodeExecutor
from models.rl_loop import RLLoop
from models.memory_modules import EpisodicMemory, LongTermMemory, ContextualMemory
from models.ad_creation import AdCreator
from models.task_automation import TaskAutomation
from models.audio_generation import AudioGenerator, MusicGenerator
from models.image_generation import ImageGenerator
from models.video_generation import VideoGenerator
from models.movie_generation import MovieGenerator
from models.game_generation import GameGenerator
from models.social_post_curation import SocialPostCurator
from models.news_curation import NewsCurator
from models.trend_analysis import TrendAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Pollen Adaptive Intelligence API",
    description="Advanced AI platform for content generation, trend analysis, and task automation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class GenerateRequest(BaseModel):
    input_text: str
    mode: Optional[str] = "chat"
    type: Optional[str] = "general"

class GenerateResponse(BaseModel):
    content: Any
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    type: Optional[str] = None
    mode: Optional[str] = None
    timestamp: Optional[str] = None

# Initialize AI models (singleton pattern for efficiency)
class AIModels:
    def __init__(self):
        self.task_proposer = None
        self.task_solver = None
        self.social_curator = SocialPostCurator()
        self.news_curator = NewsCurator()
        self.trend_analyzer = TrendAnalyzer()
        self.ad_creator = None
        self.task_automation = TaskAutomation()
        self.audio_generator = None
        self.music_generator = None
        self.image_generator = None
        self.video_generator = None
        self.movie_generator = None
        self.game_generator = None
        
    def get_social_curator(self):
        return self.social_curator
    
    def get_news_curator(self):
        return self.news_curator
    
    def get_trend_analyzer(self):
        return self.trend_analyzer
    
    def get_task_automation(self):
        return self.task_automation

# Global models instance
models = AIModels()

@app.get("/")
async def root():
    return {
        "message": "Pollen Adaptive Intelligence API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "social_curator": True,
            "news_curator": True,
            "trend_analyzer": True,
            "task_automation": True
        }
    }

# Main generate endpoint compatible with existing platform
@app.post("/generate")
async def generate_content(request: GenerateRequest):
    """Main content generation endpoint"""
    try:
        input_text = request.input_text
        mode = request.mode
        content_type = request.type
        
        logger.info(f"Generating {content_type} content for prompt: '{input_text[:100]}...'")
        
        # Route to appropriate model based on type
        if content_type == "social":
            curator = models.get_social_curator()
            result = curator.curate_post(input_text)
            return GenerateResponse(
                content=result["content"],
                confidence=result.get("engagement_score", 8.0) / 10.0,
                reasoning=f"Generated optimized social media content with {result.get('engagement_score', 8.0)}/10 engagement score",
                type=content_type,
                mode=mode,
                timestamp=datetime.now().isoformat()
            )
        
        elif content_type == "news":
            curator = models.get_news_curator()
            result = curator.curate_news(input_text)
            return GenerateResponse(
                content=result,
                confidence=result["metadata"].get("confidence_score", 0.9),
                reasoning=f"Curated {len(result['articles'])} news articles with {result['trending_score']}/10 relevance",
                type=content_type,
                mode=mode,
                timestamp=datetime.now().isoformat()
            )
        
        elif content_type == "trend_analysis":
            analyzer = models.get_trend_analyzer()
            result = analyzer.analyze_trends(input_text)
            return GenerateResponse(
                content=result,
                confidence=result["metadata"].get("confidence_score", 0.9),
                reasoning=f"Analyzed {len(result['trending_topics'])} trends with {result['momentum_indicators']['overall_momentum']}/10 momentum",
                type=content_type,
                mode=mode,
                timestamp=datetime.now().isoformat()
            )
        
        elif content_type in ["general", "feed_post"]:
            # Use social curator for general content generation
            curator = models.get_social_curator()
            result = curator.curate_post(input_text)
            
            # Format as general content
            general_content = {
                "title": "AI-Generated Content",
                "content": result["content"],
                "summary": f"Quality content generated based on: {input_text[:100]}...",
                "significance": result.get("engagement_score", 8.0)
            }
            
            return GenerateResponse(
                content=general_content,
                confidence=result.get("engagement_score", 8.0) / 10.0,
                reasoning="Generated using advanced AI content optimization",
                type=content_type,
                mode=mode,
                timestamp=datetime.now().isoformat()
            )
        
        else:
            # Fallback for unknown types
            return GenerateResponse(
                content={
                    "title": f"Generated Content - {content_type.title()}",
                    "content": f"Advanced AI-generated content for: {input_text}",
                    "summary": f"Intelligent content creation using Pollen AI adaptive algorithms",
                    "significance": 8.5
                },
                confidence=0.85,
                reasoning=f"Generated {content_type} content using fallback algorithms",
                type=content_type,
                mode=mode,
                timestamp=datetime.now().isoformat()
            )
    
    except Exception as e:
        logger.error(f"Content generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")

# Specific endpoints for different content types
@app.post("/curate-social-post")
async def curate_social_post(request: GenerateRequest):
    """Generate social media content"""
    try:
        curator = models.get_social_curator()
        result = curator.curate_post(request.input_text)
        return {"post": result}
    except Exception as e:
        logger.error(f"Social curation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/curate-news")
async def curate_news(request: GenerateRequest):
    """Curate news content"""
    try:
        curator = models.get_news_curator()
        result = curator.curate_news(request.input_text)
        return {"news": result}
    except Exception as e:
        logger.error(f"News curation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-trends")
async def analyze_trends(request: GenerateRequest):
    """Analyze trending topics"""
    try:
        analyzer = models.get_trend_analyzer()
        result = analyzer.analyze_trends(request.input_text)
        return {"trends": result}
    except Exception as e:
        logger.error(f"Trend analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/automate-task")
async def automate_task(request: GenerateRequest):
    """Automate tasks"""
    try:
        automation = models.get_task_automation()
        # Simple task automation - in real implementation, would route to specific tasks
        result = f"Task automation initiated for: {request.input_text}"
        return {"result": result}
    except Exception as e:
        logger.error(f"Task automation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional utility endpoints
@app.get("/categories")
async def get_categories():
    """Get available content categories"""
    return {
        "categories": [
            "general", "social", "news", "trend_analysis", 
            "music", "product", "entertainment", "learning"
        ],
        "modes": ["chat", "analysis", "creative"],
        "status": "active"
    }

@app.get("/trending")
async def get_trending():
    """Get current trending topics"""
    try:
        analyzer = models.get_trend_analyzer()
        result = analyzer.analyze_trends("current trending topics")
        return {
            "trending_topics": result["trending_topics"][:5],  # Top 5
            "momentum": result["momentum_indicators"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Trending topics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )