"""
Adaptive Intelligence Worker Bee Router
SSE streaming endpoints for curated content and AI training
"""

from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.enhanced_scraper import enhanced_scraper
from backend.services.content_processor import content_processor
from backend.services.pollen_ai_trainer import pollen_ai_trainer
from backend.models.schemas import QueryRequest
from typing import Optional

router = APIRouter()


@router.get("/curated-feed")
async def get_curated_feed(
    category: Optional[str] = Query(None, description="Filter by category (tech, business, etc.)"),
    min_score: float = Query(50.0, ge=0, le=100, description="Minimum Adaptive Intelligence quality score"),
    max_results: int = Query(20, ge=1, le=50, description="Maximum number of results")
):
    """
    Stream curated content with Adaptive Intelligence scoring via SSE
    
    Returns real-time curated news and content scored by:
    - Scope, Intensity, Originality, Immediacy
    - Practicability, Positivity, Credibility
    """
    async def generate():
        async for chunk in enhanced_scraper.stream_curated_content(
            category=category,
            min_score=min_score,
            max_results=max_results
        ):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())


@router.get("/trending-analysis")
async def analyze_trending(
    max_results: int = Query(10, ge=1, le=30)
):
    """
    Analyze and stream trending topics with quality scoring
    """
    async def generate():
        async for chunk in enhanced_scraper.stream_curated_content(
            category=None,
            min_score=70.0,  # High quality only for trending
            max_results=max_results
        ):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())


@router.get("/search")
async def search_curated(
    query: str = Query(..., description="Search query"),
    max_results: int = Query(5, ge=1, le=20)
):
    """
    Search the web and return curated, scored results via SSE
    """
    async def generate():
        async for chunk in enhanced_scraper.search_and_scrape_web(
            query=query,
            max_results=max_results
        ):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())


@router.post("/train")
async def train_pollen_ai(
    min_score: float = Query(70.0, description="Minimum quality score for training data"),
    max_items: int = Query(50, description="Maximum training items")
):
    """
    Train Pollen AI with curated content via SSE streaming
    """
    async def generate():
        import json
        
        # First, gather high-quality content
        yield {"event": "message", "data": json.dumps({
            "type": "status",
            "message": "📚 Gathering high-quality content for training..."
        }) + "\n"}
        
        content_items = []
        async for chunk in enhanced_scraper.stream_curated_content(
            category=None,
            min_score=min_score,
            max_results=max_items
        ):
            data = json.loads(chunk)
            if data.get("type") == "content":
                content_items.append(data["data"])
        
        # Prepare training data
        training_data = content_processor.prepare_training_data(
            content_items,
            min_score=min_score
        )
        
        # Stream training progress
        async for chunk in pollen_ai_trainer.stream_incremental_training(training_data):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())


@router.get("/knowledge-base")
async def get_knowledge_base():
    """
    Get current Pollen AI knowledge base summary
    """
    return pollen_ai_trainer.get_knowledge_summary()


@router.get("/score-content")
async def score_content(
    title: str = Query(..., description="Content title"),
    description: str = Query("", description="Content description"),
    source: str = Query("Unknown", description="Content source"),
    category: str = Query("general", description="Content category")
):
    """
    Score individual content using Adaptive Intelligence algorithm
    """
    from backend.services.adaptive_scorer import adaptive_scorer
    from datetime import datetime
    
    content = {
        "title": title,
        "description": description,
        "source": source,
        "category": category,
        "published_at": datetime.now().isoformat()
    }
    
    score = adaptive_scorer.score_content(content)
    
    return {
        "content": content,
        "score": score.to_dict(),
        "explanation": {
            "scope": "Number of individuals affected by the content",
            "intensity": "Magnitude of the content's impact",
            "originality": "Unexpected or distinctive nature",
            "immediacy": "Temporal proximity (how recent)",
            "practicability": "Actionable steps for readers",
            "positivity": "Positive aspects of the content",
            "credibility": "Source reliability assessment"
        }
    }


@router.get("/health")
async def adaptive_intelligence_health():
    """
    Health check for Adaptive Intelligence Worker Bee services
    """
    kb_summary = pollen_ai_trainer.get_knowledge_summary()
    
    return {
        "status": "operational",
        "service": "Adaptive Intelligence Worker Bee",
        "version": "1.0.0",
        "components": {
            "scraper": "active",
            "scorer": "active",
            "trainer": "active",
            "processor": "active"
        },
        "knowledge_base": {
            "categories": kb_summary["total_categories"],
            "training_sessions": kb_summary["training_sessions"]
        }
    }
