"""
Feed Router - Updated to use real database and Bento Buzz scoring
"""

from fastapi import APIRouter, Depends, Query
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import json
import asyncio

from backend.database import get_db
from backend.services.content_storage import content_storage
from backend.services.background_scraper import background_scraper

router = APIRouter()


@router.get("/posts")
async def get_feed_posts(
    content_type: Optional[str] = Query(None, description="Filter by content type (news, trend, event, product)"),
    min_score: float = Query(50.0, description="Minimum quality score"),
    limit: int = Query(50, description="Number of posts to return"),
    db: Session = Depends(get_db)
):
    """
    Get high-quality feed posts from database
    Applies Bento Buzz scoring and filtering
    """
    posts = content_storage.get_quality_content(
        db,
        content_type=content_type,
        min_score=min_score,
        limit=limit,
        hours_old=168  # Last 7 days
    )
    
    formatted_posts = []
    for i, post in enumerate(posts):
        score_data = post.get("adaptive_score", {})
        
        formatted_posts.append({
            "id": i + 1,
            "user": {
                "name": "Anonymous",
                "username": None,
                "avatar": None,
                "verified": False
            },
            "time": post.get("stored_at", "recently"),
            "content": post.get("description", "")[:200],
            "title": post.get("title", ""),
            "url": post.get("url"),
            "source": post.get("source", "Unknown"),
            "tags": _extract_tags(post),
            "image": post.get("image_url"),
            "views": post.get("engagement_metrics", {}).get("views", 0),
            "engagement": int(score_data.get("overall", 75)),
            "qualityScore": int(score_data.get("overall", 75)),
            "trending": post.get("trending", False),
            "type": "post",
            "adaptive_score": score_data
        })
    
    return formatted_posts


@router.get("/trending")
async def get_trending_topics(
    limit: int = Query(10, description="Number of trending topics"),
    db: Session = Depends(get_db)
):
    """
    Get trending topics from database
    """
    trends = content_storage.get_quality_content(
        db,
        content_type="trend",
        min_score=75.0,
        limit=limit,
        hours_old=48
    )
    
    trending_topics = []
    for i, trend in enumerate(trends):
        title = trend.get("title", "trending").replace("Trending: ", "").strip()
        words = title.split()
        tag = " ".join(words[:3]) if len(words) > 3 else title
        
        growth = trend.get("engagement_metrics", {}).get("growth", "")
        posts = trend.get("engagement_metrics", {}).get("search_volume", "N/A")
        
        trending_topics.append({
            "id": i + 1,
            "tag": f"#{tag}",
            "posts": posts,
            "trend": growth or f"+{int(trend.get('adaptive_score', {}).get('intensity', 50))}%"
        })
    
    return trending_topics[:limit]


@router.get("/events")
async def get_upcoming_events(
    limit: int = Query(10, description="Number of events"),
    db: Session = Depends(get_db)
):
    """
    Get upcoming events from database
    """
    events = content_storage.get_quality_content(
        db,
        content_type="event",
        min_score=70.0,
        limit=limit,
        hours_old=336  # Last 2 weeks
    )
    
    formatted_events = []
    for i, event in enumerate(events):
        formatted_events.append({
            "id": i + 1,
            "title": event.get("title", ""),
            "category": event.get("raw_data", {}).get("category", "Event"),
            "date": event.get("raw_data", {}).get("date", ""),
            "location": event.get("raw_data", {}).get("location", "Virtual"),
            "attendees": event.get("raw_data", {}).get("attendees", ""),
            "quality_score": int(event.get("adaptive_score", {}).get("overall", 70))
        })
    
    return formatted_events


@router.get("/suggested-users")
async def get_suggested_users(limit: int = Query(10)):
    """
    Get suggested anonymous users (static for privacy)
    """
    return [
        {"id": i+1, "name": "Anonymous User", "bio": "Privacy-first platform user", "mutual": 0}
        for i in range(limit)
    ]


@router.post("/trigger-scrape")
async def trigger_manual_scrape(db: Session = Depends(get_db)):
    """
    Manually trigger a scraping job
    Admin/testing endpoint
    """
    asyncio.create_task(background_scraper.run_daily_scrape_job())
    return {"message": "Scraping job started in background"}


@router.get("/scrape-status")
async def get_scrape_status(db: Session = Depends(get_db)):
    """
    Get status of recent scraping jobs
    """
    from backend.database import ScraperJob
    
    recent_jobs = db.query(ScraperJob).order_by(
        ScraperJob.created_at.desc()
    ).limit(10).all()
    
    return [{
        "job_id": job.job_id,
        "job_type": job.job_type,
        "status": job.status,
        "items_scraped": job.items_scraped,
        "items_passed": job.items_passed,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    } for job in recent_jobs]


@router.get("/stats")
async def get_feed_stats(db: Session = Depends(get_db)):
    """
    Get statistics about content in the database
    """
    from backend.database import Content, ContentScore, TrainingData
    
    total_content = db.query(Content).count()
    total_high_quality = db.query(ContentScore).filter(
        ContentScore.overall_score >= 50
    ).count()
    total_excellent_quality = db.query(ContentScore).filter(
        ContentScore.overall_score >= 70
    ).count()
    total_training_data = db.query(TrainingData).filter(
        TrainingData.used_for_training == False
    ).count()
    
    return {
        "total_content": total_content,
        "high_quality_content": total_high_quality,
        "excellent_quality_content": total_excellent_quality,
        "training_data_available": total_training_data,
        "quality_percentage": int((total_high_quality / total_content * 100)) if total_content > 0 else 0
    }


def _extract_tags(post: dict) -> List[str]:
    """Extract tags from post content"""
    title = post.get("title", "").lower()
    category = post.get("raw_data", {}).get("category", "")
    
    tags = []
    if category:
        tags.append(f"#{category}")
    
    common_tags = ["ai", "tech", "design", "startup", "product", "trending"]
    for tag in common_tags:
        if tag in title and f"#{tag}" not in tags:
            tags.append(f"#{tag}")
            if len(tags) >= 3:
                break
    
    return tags[:3]
