from fastapi import APIRouter, Query
from typing import List, Optional
import random
from datetime import datetime, timedelta
import httpx

router = APIRouter()

TOPICS = [
    "AI", "blockchain", "web3", "crypto", "design", "technology", 
    "sustainability", "health", "fitness", "travel", "food", 
    "photography", "art", "music", "gaming", "startup", "finance"
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Los Angeles, CA", 
    "Austin, TX", "Seattle, WA", "Miami, FL", "Chicago, IL",
    "Boston, MA", "Denver, CO", "Portland, OR", "Virtual"
]

NAMES = [
    "Alex Chen", "Sarah Johnson", "Marcus Williams", "Emily Zhang",
    "David Park", "Jessica Martinez", "Ryan Thompson", "Lisa Anderson",
    "Kevin Lee", "Amanda Davis", "Chris Brown", "Nicole Taylor"
]

def generate_trending_topics():
    trends = []
    selected_topics = random.sample(TOPICS, 5)
    
    for i, topic in enumerate(selected_topics):
        trend_percentage = random.randint(5, 30)
        post_count = random.randint(20, 200) * 1000
        
        trends.append({
            "id": i + 1,
            "tag": f"#{topic.lower()}",
            "posts": f"{post_count // 1000}K",
            "trend": f"+{trend_percentage}%"
        })
    
    return trends

def generate_events(limit: int = 10):
    events = []
    categories = ["Hangout", "Workshop", "Social", "Networking", "Conference", "Meetup"]
    
    event_templates = [
        "Monthly Talk: Defining DeFi in an Omnichain Future",
        "AI & Design Workshop: Creating the Future",
        "Coffee Meetup - Tech Professionals",
        "Startup Founders Networking Night",
        "Creative Coding Workshop",
        "Photography Walk & Share",
        "Fitness Bootcamp in the Park",
        "Sustainable Living Panel Discussion",
        "Music Jam Session",
        "Book Club: Tech & Society",
        "Cooking Class: International Cuisine",
        "Gaming Tournament: Esports Night"
    ]
    
    for i in range(min(limit, len(event_templates))):
        hours_from_now = random.randint(2, 168)
        attendees = random.randint(20, 500)
        
        if hours_from_now < 24:
            time_str = f"Starting in {hours_from_now} hours"
        elif hours_from_now < 48:
            time_str = "Tomorrow at " + random.choice(["3PM", "6PM", "7PM", "8PM"])
        else:
            time_str = "This Weekend"
        
        events.append({
            "id": i + 1,
            "title": event_templates[i],
            "time": time_str,
            "attendees": f"{attendees}+ interested",
            "category": random.choice(categories),
            "location": random.choice(LOCATIONS),
            "image": None
        })
    
    return events

def generate_suggestions(limit: int = 10):
    suggestions = []
    bios = [
        "Sweet, simple, repeat! 🔄✨",
        "Steering every flavor into perfection",
        "Style the way to impress",
        "Creating beautiful experiences",
        "Building the future, one line at a time",
        "Designer, dreamer, doer",
        "Coffee enthusiast ☕ | Tech lover",
        "Making the web more beautiful",
        "Exploring the intersection of art & tech",
        "Digital nomad 🌍"
    ]
    
    for i in range(limit):
        quality_score = random.randint(60, 98)
        suggestions.append({
            "id": i + 1,
            "name": "Anonymous User",
            "bio": random.choice(bios),
            "username": None,
            "mutual": quality_score
        })
    
    return suggestions

async def fetch_real_news():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "category": "technology",
                    "language": "en",
                    "pageSize": 5
                },
                timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("articles", [])
    except:
        pass
    return []

def generate_ai_posts(limit: int = 20, interests: Optional[List[str]] = None):
    posts = []
    
    post_templates = [
        {
            "content": "Just discovered an amazing new AI tool that completely transforms the way we think about {topic}. The future is here! 🚀",
            "type": "text"
        },
        {
            "content": "Working on something exciting in the {topic} space. Can't wait to share more soon!",
            "type": "text"
        },
        {
            "content": "Hot take: {topic} is going to be the next big thing in 2025. Here's why...",
            "type": "text"
        },
        {
            "content": "Beautiful {topic} inspiration for today ✨",
            "type": "image"
        },
        {
            "content": "5 things I learned about {topic} this week that changed my perspective",
            "type": "text"
        }
    ]
    
    image_urls = [
        "https://images.unsplash.com/photo-1518780664697-55e3ad937233?w=600&h=400&fit=crop",
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=600&h=400&fit=crop",
        "https://images.unsplash.com/photo-1561070791-2526d30994b5?w=600&h=400&fit=crop",
        "https://images.unsplash.com/photo-1542831371-29b0f74f9713?w=600&h=400&fit=crop",
    ]
    
    for i in range(limit):
        template = random.choice(post_templates)
        topic = random.choice(interests or TOPICS)
        
        hours_ago = random.randint(1, 48)
        if hours_ago == 1:
            time_str = "1h"
        elif hours_ago < 24:
            time_str = f"{hours_ago}h"
        else:
            time_str = f"{hours_ago // 24}d"
        
        views = random.randint(1000, 100000)
        engagement = random.randint(40, 95)
        quality_score = random.randint(60, 98)
        
        post = {
            "id": i + 1,
            "user": {
                "name": "Anonymous",
                "username": None,
                "avatar": None,
                "verified": False
            },
            "time": time_str,
            "content": template["content"].format(topic=topic),
            "type": "post",
            "views": views,
            "engagement": engagement,
            "qualityScore": quality_score,
            "trending": quality_score > 85 and engagement > 75
        }
        
        if template["type"] == "image" and random.random() > 0.5:
            post["image"] = random.choice(image_urls)
        
        if random.random() > 0.7:
            post["tags"] = [f"#{topic.lower()}", f"#{random.choice(TOPICS).lower()}"]
        
        posts.append(post)
    
    return posts

@router.get("/posts")
async def get_feed_posts(
    limit: int = Query(20, ge=1, le=100),
    interests: Optional[str] = Query(None)
):
    interest_list = interests.split(",") if interests else None
    posts = generate_ai_posts(limit, interest_list)
    return posts

@router.get("/trending")
async def get_trending_topics():
    return generate_trending_topics()

@router.get("/events")
async def get_events(
    limit: int = Query(10, ge=1, le=50),
    location: Optional[str] = Query(None)
):
    events = generate_events(limit)
    
    if location:
        events = [e for e in events if location.lower() in e["location"].lower()]
    
    return events

@router.get("/suggestions")
async def get_suggestions(
    limit: int = Query(10, ge=1, le=50)
):
    return generate_suggestions(limit)

@router.get("/news")
async def get_news_feed(
    limit: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query("technology")
):
    real_news = await fetch_real_news()
    
    news_posts = []
    for i, article in enumerate(real_news[:limit]):
        news_posts.append({
            "id": i + 1,
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "url": article.get("url", ""),
            "image": article.get("urlToImage"),
            "source": article.get("source", {}).get("name", "Unknown"),
            "publishedAt": article.get("publishedAt", ""),
            "category": category
        })
    
    if len(news_posts) < limit:
        ai_news_topics = [
            "AI breakthrough transforms healthcare industry",
            "New sustainable technology could change everything",
            "Startup raises $50M to revolutionize {topic}",
            "Study reveals surprising findings about {topic}",
            "Experts predict major shift in {topic} by 2026"
        ]
        
        for i in range(len(news_posts), limit):
            topic = random.choice(TOPICS)
            template = random.choice(ai_news_topics)
            
            news_posts.append({
                "id": i + 1,
                "title": template.format(topic=topic),
                "description": f"A comprehensive look at the latest developments in {topic} and what it means for the future.",
                "url": "#",
                "image": None,
                "source": "AI Generated",
                "publishedAt": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                "category": category,
                "ai_generated": True
            })
    
    return news_posts
