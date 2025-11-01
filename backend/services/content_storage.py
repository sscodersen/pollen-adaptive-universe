"""
Content Storage Service
Manages persistent storage of scraped content, scores, and training data
"""

from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import hashlib
import json

from backend.database import Content, ContentScore, TrainingData, UserPreferences, ScraperJob
from backend.services.adaptive_scorer import adaptive_scorer


class ContentStorageService:
    def __init__(self):
        self.min_quality_threshold = 50.0
    
    def store_content_with_score(
        self,
        db: Session,
        content_data: Dict[str, Any],
        content_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Store content and its score in the database
        Only stores if score meets threshold
        
        Returns stored content with score if successful, None if filtered out
        """
        try:
            content_id = self._generate_content_id(content_data)
            
            existing = db.query(Content).filter(Content.content_id == content_id).first()
            if existing:
                return None
            
            score = adaptive_scorer.score_content(content_data)
            
            if score.overall < self.min_quality_threshold:
                return None
            
            content = Content(
                content_id=content_id,
                content_type=content_type,
                title=content_data.get("title", "")[:500],
                description=content_data.get("description", ""),
                url=content_data.get("url", ""),
                source=content_data.get("source", ""),
                image_url=content_data.get("image_url"),
                raw_data=content_data,
                published_at=self._parse_published_date(content_data.get("published_at"))
            )
            
            db.add(content)
            db.flush()
            
            content_score = ContentScore(
                content_id=content_id,
                scope_score=score.scope,
                intensity_score=score.intensity,
                originality_score=score.originality,
                immediacy_score=score.immediacy,
                practicability_score=score.practicability,
                positivity_score=score.positivity,
                credibility_score=score.credibility,
                overall_score=score.overall,
                trending=score.overall >= 85
            )
            
            db.add(content_score)
            
            if score.overall >= 80:
                training_data = TrainingData(
                    content_id=content_id,
                    quality_score=score.overall,
                    engagement_metrics=content_data.get("engagement_metrics", {}),
                    user_feedback={}
                )
                db.add(training_data)
            
            db.commit()
            
            return {
                "content": content_data,
                "score": score.to_dict(),
                "stored": True
            }
            
        except Exception as e:
            db.rollback()
            print(f"Error storing content: {e}")
            return None
    
    def get_quality_content(
        self,
        db: Session,
        content_type: Optional[str] = None,
        min_score: float = 70.0,
        limit: int = 50,
        hours_old: int = 168
    ) -> List[Dict[str, Any]]:
        """
        Retrieve high-quality content from database
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_old)
            
            query = db.query(Content, ContentScore).join(
                ContentScore, Content.content_id == ContentScore.content_id
            ).filter(
                ContentScore.overall_score >= min_score,
                Content.created_at >= cutoff_time
            )
            
            if content_type:
                query = query.filter(Content.content_type == content_type)
            
            query = query.order_by(ContentScore.overall_score.desc())
            results = query.limit(limit).all()
            
            content_list = []
            for content, score in results:
                content_dict = content.raw_data if content.raw_data else {}
                content_dict.update({
                    "id": content.id,
                    "content_id": content.content_id,
                    "adaptive_score": {
                        "scope": score.scope_score,
                        "intensity": score.intensity_score,
                        "originality": score.originality_score,
                        "immediacy": score.immediacy_score,
                        "practicability": score.practicability_score,
                        "positivity": score.positivity_score,
                        "credibility": score.credibility_score,
                        "overall": score.overall_score,
                        "quality_tier": "HIGH QUALITY" if score.overall_score >= 70 else "MEDIUM QUALITY"
                    },
                    "trending": score.trending,
                    "stored_at": content.created_at.isoformat()
                })
                content_list.append(content_dict)
            
            return content_list
            
        except Exception as e:
            print(f"Error retrieving content: {e}")
            return []
    
    def get_training_dataset(
        self,
        db: Session,
        min_quality: float = 80.0,
        max_items: int = 1000,
        unused_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get high-quality content for Pollen AI training
        """
        try:
            query = db.query(TrainingData, Content).join(
                Content, TrainingData.content_id == Content.content_id
            ).filter(
                TrainingData.quality_score >= min_quality
            )
            
            if unused_only:
                query = query.filter(TrainingData.used_for_training == False)
            
            query = query.order_by(TrainingData.quality_score.desc())
            results = query.limit(max_items).all()
            
            dataset = []
            for training_data, content in results:
                dataset.append({
                    "content_id": content.content_id,
                    "title": content.title,
                    "description": content.description,
                    "url": content.url,
                    "source": content.source,
                    "quality_score": training_data.quality_score,
                    "category": content.raw_data.get("category", "general") if content.raw_data else "general",
                    "engagement_metrics": training_data.engagement_metrics,
                    "content_type": content.content_type
                })
            
            return dataset
            
        except Exception as e:
            print(f"Error getting training dataset: {e}")
            return []
    
    def get_content_by_category(
        self,
        db: Session,
        category: str,
        limit: int = 10,
        min_quality: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Get content by category (trend, event, product, news)
        """
        try:
            query = db.query(Content, ContentScore).join(
                ContentScore, Content.content_id == ContentScore.content_id
            ).filter(
                Content.content_type == category,
                ContentScore.overall_score >= min_quality
            ).order_by(ContentScore.overall_score.desc())
            
            results = query.limit(limit).all()
            
            content_list = []
            for content, score in results:
                content_dict = content.raw_data if content.raw_data else {}
                content_dict.update({
                    "id": content.id,
                    "content_id": content.content_id,
                    "title": content.title,
                    "description": content.description,
                    "url": content.url,
                    "source": content.source,
                    "category": content.content_type,
                    "quality_score": score.overall_score,
                    "engagement_metrics": content.raw_data.get("engagement_metrics", {}) if content.raw_data else {},
                    "image_url": content.raw_data.get("image_url") if content.raw_data else None
                })
                content_list.append(content_dict)
            
            return content_list
            
        except Exception as e:
            print(f"Error getting content by category: {e}")
            return []
    
    def mark_training_data_used(
        self,
        db: Session,
        content_ids: List[str],
        batch_id: str
    ) -> int:
        """
        Mark content as used for training
        """
        try:
            updated = db.query(TrainingData).filter(
                TrainingData.content_id.in_(content_ids)
            ).update(
                {
                    "used_for_training": True,
                    "training_batch_id": batch_id
                },
                synchronize_session=False
            )
            db.commit()
            return updated
        except Exception as e:
            db.rollback()
            print(f"Error marking training data: {e}")
            return 0
    
    def create_scraper_job(
        self,
        db: Session,
        job_type: str,
        source: Optional[str] = None
    ) -> str:
        """
        Create a new scraper job record
        """
        try:
            job_id = f"{job_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            job = ScraperJob(
                job_id=job_id,
                job_type=job_type,
                source=source,
                status="running",
                started_at=datetime.utcnow()
            )
            
            db.add(job)
            db.commit()
            
            return job_id
        except Exception as e:
            db.rollback()
            print(f"Error creating scraper job: {e}")
            return ""
    
    def update_scraper_job(
        self,
        db: Session,
        job_id: str,
        status: str,
        items_scraped: int = 0,
        items_scored: int = 0,
        items_passed: int = 0,
        error_message: Optional[str] = None
    ):
        """
        Update scraper job status
        """
        try:
            job = db.query(ScraperJob).filter(ScraperJob.job_id == job_id).first()
            if job:
                job.status = status
                job.items_scraped = items_scraped
                job.items_scored = items_scored
                job.items_passed = items_passed
                if error_message:
                    job.error_message = error_message
                if status == "completed" or status == "failed":
                    job.completed_at = datetime.utcnow()
                db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error updating scraper job: {e}")
    
    def get_user_preferences(
        self,
        db: Session,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get user personalization preferences
        """
        try:
            prefs = db.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).first()
            
            if not prefs:
                prefs = UserPreferences(user_id=user_id)
                db.add(prefs)
                db.commit()
            
            return {
                "interests": prefs.interests or [],
                "categories": prefs.categories or [],
                "algorithm_preference": prefs.algorithm_preference,
                "min_quality_score": prefs.min_quality_score,
                "display_settings": prefs.display_settings or {},
                "engagement_history": prefs.engagement_history or []
            }
        except Exception as e:
            print(f"Error getting user preferences: {e}")
            return {
                "interests": [],
                "categories": [],
                "algorithm_preference": "personalized",
                "min_quality_score": 70.0,
                "display_settings": {},
                "engagement_history": []
            }
    
    def update_user_preferences(
        self,
        db: Session,
        user_id: str,
        preferences: Dict[str, Any]
    ):
        """
        Update user personalization preferences
        """
        try:
            prefs = db.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).first()
            
            if not prefs:
                prefs = UserPreferences(user_id=user_id)
                db.add(prefs)
            
            if "interests" in preferences:
                prefs.interests = preferences["interests"]
            if "categories" in preferences:
                prefs.categories = preferences["categories"]
            if "algorithm_preference" in preferences:
                prefs.algorithm_preference = preferences["algorithm_preference"]
            if "min_quality_score" in preferences:
                prefs.min_quality_score = preferences["min_quality_score"]
            if "display_settings" in preferences:
                prefs.display_settings = preferences["display_settings"]
            
            prefs.updated_at = datetime.utcnow()
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error updating user preferences: {e}")
    
    def store_synthetic_training_data(
        self,
        db: Session,
        synthetic_data: Dict[str, Any],
        batch_id: str
    ):
        """
        Store synthetic training data generated by AI
        """
        try:
            content_id = self._generate_content_id({
                "title": synthetic_data.get("original_query", ""),
                "url": f"synthetic_{batch_id}"
            })
            
            content = Content(
                content_id=content_id,
                content_type="synthetic",
                title=synthetic_data.get("original_query", "Synthetic Data"),
                description=synthetic_data.get("response", ""),
                url=f"synthetic://{batch_id}",
                source="AI Generated",
                raw_data=synthetic_data
            )
            
            db.add(content)
            
            training_data = TrainingData(
                content_id=content_id,
                quality_score=synthetic_data.get("quality_score", 70.0),
                engagement_metrics={"synthetic": True, "batch_id": batch_id},
                user_feedback={},
                training_batch_id=batch_id
            )
            
            db.add(training_data)
            db.commit()
            
        except Exception as e:
            db.rollback()
            print(f"Error storing synthetic training data: {e}")
    
    def delete_synthetic_batch(self, db: Session, batch_id: str):
        """
        Delete a batch of synthetic training data (rollback)
        """
        try:
            db.query(TrainingData).filter(
                TrainingData.training_batch_id == batch_id
            ).delete()
            
            db.query(Content).filter(
                Content.url.like(f"synthetic://{batch_id}%")
            ).delete()
            
            db.commit()
            
        except Exception as e:
            db.rollback()
            print(f"Error deleting synthetic batch: {e}")
    
    def _generate_content_id(self, content_data: Dict[str, Any]) -> str:
        """
        Generate unique content ID from URL or title
        """
        identifier = content_data.get("url") or content_data.get("title", "")
        return hashlib.md5(identifier.encode()).hexdigest()
    
    def _parse_published_date(self, date_str: Any) -> Optional[datetime]:
        """
        Parse published date string to datetime
        """
        if not date_str:
            return None
        
        try:
            if isinstance(date_str, datetime):
                return date_str
            if isinstance(date_str, str):
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            pass
        
        return None


content_storage = ContentStorageService()
