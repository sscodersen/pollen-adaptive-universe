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

class IoTDevice(Base):
    __tablename__ = "iot_devices"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True, nullable=False)
    device_type = Column(String, nullable=False)
    device_name = Column(String, nullable=False)
    manufacturer = Column(String)
    model = Column(String)
    os_version = Column(String)
    firmware_version = Column(String)
    status = Column(String, default="offline")
    location = Column(JSON)
    capabilities = Column(JSON)
    device_metadata = Column(JSON)
    last_seen = Column(DateTime)
    registered_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_device_type_status', 'device_type', 'status'),
    )

class DeviceTelemetry(Base):
    __tablename__ = "device_telemetry"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, nullable=False, index=True)
    telemetry_type = Column(String, nullable=False)
    sensor_data = Column(JSON)
    value = Column(Float)
    unit = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    telemetry_metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_device_telemetry_time', 'device_id', 'timestamp'),
    )

class AIContentDetection(Base):
    __tablename__ = "ai_content_detection"
    
    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(String, unique=True, index=True, nullable=False)
    content_hash = Column(String, index=True)
    content_type = Column(String, nullable=False)
    ai_generated_probability = Column(Float, nullable=False)
    human_generated_probability = Column(Float, nullable=False)
    detection_confidence = Column(Float, nullable=False)
    model_used = Column(String)
    features_detected = Column(JSON)
    detection_metadata = Column(JSON)
    verified_by_human = Column(Boolean, default=False)
    verification_status = Column(String, default="unverified")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_ai_detection_prob', 'ai_generated_probability'),
    )

class AgricultureData(Base):
    __tablename__ = "agriculture_data"
    
    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(String, nullable=False, index=True)
    field_id = Column(String, index=True)
    crop_type = Column(String, nullable=False)
    location = Column(JSON, nullable=False)
    soil_data = Column(JSON)
    climate_data = Column(JSON)
    crop_health_score = Column(Float)
    growth_stage = Column(String)
    estimated_yield = Column(Float)
    irrigation_level = Column(Float)
    fertilizer_data = Column(JSON)
    pest_detection = Column(JSON)
    farm_metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_farm_field_time', 'farm_id', 'field_id', 'timestamp'),
    )

class CropRecommendation(Base):
    __tablename__ = "crop_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    recommendation_id = Column(String, unique=True, index=True, nullable=False)
    farm_id = Column(String, nullable=False, index=True)
    field_id = Column(String)
    recommended_crop = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    expected_yield = Column(Float)
    optimal_conditions = Column(JSON)
    seasonal_timing = Column(JSON)
    risk_factors = Column(JSON)
    reasoning = Column(Text)
    recommendation_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class GeoOptimization(Base):
    __tablename__ = "geo_optimization"
    
    id = Column(Integer, primary_key=True, index=True)
    optimization_id = Column(String, unique=True, index=True, nullable=False)
    industry = Column(String, nullable=False, index=True)
    location = Column(JSON, nullable=False)
    optimization_type = Column(String, nullable=False)
    parameters = Column(JSON)
    recommendations = Column(JSON)
    efficiency_score = Column(Float)
    cost_savings = Column(Float)
    environmental_impact = Column(JSON)
    implementation_steps = Column(JSON)
    optimization_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_geo_industry_time', 'industry', 'created_at'),
    )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
