from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    context: Optional[dict] = None

class ShoppingRequest(QueryRequest):
    budget: Optional[float] = None
    category: Optional[str] = None
    preferences: Optional[List[str]] = []

class TravelRequest(QueryRequest):
    destination: Optional[str] = None
    dates: Optional[dict] = None
    budget: Optional[float] = None
    travelers: Optional[int] = 1

class NewsRequest(QueryRequest):
    categories: Optional[List[str]] = []
    sources: Optional[List[str]] = []

class ContentRequest(QueryRequest):
    content_type: str
    tone: Optional[str] = "professional"
    length: Optional[str] = "medium"

class ScraperRequest(BaseModel):
    query: str
    sources: Optional[List[str]] = []
    max_results: Optional[int] = 5

class AIResponse(BaseModel):
    response: str
    source: str
    confidence: float
    metadata: Optional[dict] = None
    timestamp: datetime

class User(BaseModel):
    id: str
    name: str
    email: str
    preferences: dict
    created_at: datetime

class UserPreferences(BaseModel):
    user_id: str
    categories: List[str]
    interests: List[str]
    settings: dict
