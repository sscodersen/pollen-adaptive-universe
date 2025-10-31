from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Float, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/newfrontier")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Content(Base):
    __tablename__ = "content"
    
    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(String, unique=True, index=True, nullable=False)
    content_type = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    url = Column(String)
    source = Column(String)
    image_url = Column(String)
    raw_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    published_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_content_type_created', 'content_type', 'created_at'),
    )

class ContentScore(Base):
    __tablename__ = "content_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(String, nullable=False, index=True)
    
    scope_score = Column(Float, default=0)
    intensity_score = Column(Float, default=0)
    originality_score = Column(Float, default=0)
    immediacy_score = Column(Float, default=0)
    practicability_score = Column(Float, default=0)
    positivity_score = Column(Float, default=0)
    credibility_score = Column(Float, default=0)
    
    overall_score = Column(Float, nullable=False, index=True)
    trending = Column(Boolean, default=False)
    
    scored_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_score_overall', 'overall_score'),
    )

class TrainingData(Base):
    __tablename__ = "training_data"
    
    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(String, nullable=False, index=True)
    quality_score = Column(Float, nullable=False)
    user_feedback = Column(JSON)
    engagement_metrics = Column(JSON)
    used_for_training = Column(Boolean, default=False)
    training_batch_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_training_quality', 'quality_score', 'used_for_training'),
    )

class UserPreferences(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    interests = Column(JSON, default=[])
    categories = Column(JSON, default=[])
    algorithm_preference = Column(String, default="personalized")
    min_quality_score = Column(Float, default=7.0)
    display_settings = Column(JSON, default={})
    engagement_history = Column(JSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ScraperJob(Base):
    __tablename__ = "scraper_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, nullable=False)
    job_type = Column(String, nullable=False)
    status = Column(String, default="pending")
    source = Column(String)
    items_scraped = Column(Integer, default=0)
    items_scored = Column(Integer, default=0)
    items_passed = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_job_status_type', 'status', 'job_type'),
    )

class AnonymousSession(Base):
    __tablename__ = "anonymous_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    preferences = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class QueryCache(Base):
    __tablename__ = "query_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(String, unique=True, index=True)
    feature = Column(String, nullable=False)
    response = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
