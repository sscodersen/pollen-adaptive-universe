"""
Content Processing Pipeline for Adaptive Intelligence Worker Bee
Handles content filtering, quality assessment, and training data preparation
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from .adaptive_scorer import adaptive_scorer, AdaptiveScore
import json


class ContentProcessor:
    def __init__(self):
        self.quality_thresholds = {
            "high": 70,
            "medium": 50,
            "low": 30
        }
        
    def process_batch(
        self,
        content_items: List[Dict[str, Any]],
        min_quality: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of content items
        
        Args:
            content_items: List of content dictionaries
            min_quality: Minimum quality threshold ("high", "medium", "low")
        
        Returns:
            Filtered and scored content items
        """
        min_score = self.quality_thresholds.get(min_quality, 50)
        
        processed = []
        for item in content_items:
            processed_item = self.process_single(item)
            
            if processed_item and processed_item["adaptive_score"]["overall"] >= min_score:
                processed.append(processed_item)
        
        # Sort by score
        processed.sort(key=lambda x: x["adaptive_score"]["overall"], reverse=True)
        
        return processed
    
    def process_single(self, content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single content item
        """
        try:
            # Clean and validate content
            cleaned = self._clean_content(content)
            
            if not self._validate_content(cleaned):
                return None
            
            # Score content
            score = adaptive_scorer.score_content(cleaned)
            
            # Add processing metadata
            cleaned["adaptive_score"] = score.to_dict()
            cleaned["processed_at"] = datetime.now().isoformat()
            cleaned["content_id"] = self._generate_content_id(cleaned)
            
            # Extract keywords for training
            cleaned["keywords"] = self._extract_keywords(cleaned)
            
            # Determine content category if missing
            if not cleaned.get("category"):
                cleaned["category"] = self._classify_category(cleaned)
            
            return cleaned
        except Exception as e:
            print(f"Error processing content: {e}")
            return None
    
    def _clean_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean content fields
        """
        cleaned = content.copy()
        
        # Clean title
        if "title" in cleaned:
            cleaned["title"] = self._clean_text(cleaned["title"])
        
        # Clean description
        if "description" in cleaned:
            cleaned["description"] = self._clean_text(cleaned["description"])
        
        # Ensure required fields
        cleaned.setdefault("source", "Unknown")
        cleaned.setdefault("category", "general")
        cleaned.setdefault("published_at", datetime.now().isoformat())
        
        return cleaned
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text content
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        
        return text.strip()
    
    def _validate_content(self, content: Dict[str, Any]) -> bool:
        """
        Validate content has required fields
        """
        required_fields = ["title", "source"]
        
        for field in required_fields:
            if not content.get(field):
                return False
        
        # Title should be meaningful
        if len(content.get("title", "")) < 10:
            return False
        
        return True
    
    def _generate_content_id(self, content: Dict[str, Any]) -> str:
        """
        Generate unique content ID
        """
        title = content.get("title", "")
        source = content.get("source", "")
        timestamp = content.get("published_at", datetime.now().isoformat())
        
        # Simple hash-like ID
        id_str = f"{source}:{title}:{timestamp}"
        return str(hash(id_str))
    
    def _extract_keywords(self, content: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from content for training
        """
        text = f"{content.get('title', '')} {content.get('description', '')}"
        text = text.lower()
        
        # Common words to ignore
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "should", "could", "may", "might", "can", "this", "that"
        }
        
        # Extract words
        words = text.split()
        keywords = [
            word.strip(".,!?;:\"'()[]{}") 
            for word in words 
            if len(word) > 3 and word not in stop_words
        ]
        
        # Get unique keywords
        unique_keywords = list(set(keywords))
        
        return unique_keywords[:20]  # Top 20 keywords
    
    def _classify_category(self, content: Dict[str, Any]) -> str:
        """
        Classify content into a category
        """
        text = f"{content.get('title', '')} {content.get('description', '')}".lower()
        
        categories = {
            "technology": ["tech", "ai", "software", "computer", "digital", "app", "startup", "code"],
            "business": ["business", "market", "economy", "finance", "stock", "company", "corporate"],
            "science": ["science", "research", "study", "scientists", "discovery", "nature"],
            "health": ["health", "medical", "disease", "treatment", "wellness", "fitness"],
            "politics": ["politics", "government", "election", "policy", "congress", "senate"],
            "entertainment": ["entertainment", "movie", "music", "celebrity", "film", "show"],
            "sports": ["sports", "game", "team", "player", "championship", "league"]
        }
        
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    def prepare_training_data(
        self,
        content_items: List[Dict[str, Any]],
        min_score: float = 70.0
    ) -> Dict[str, Any]:
        """
        Prepare high-quality content for Pollen AI training
        
        Args:
            content_items: List of scored content items
            min_score: Minimum Adaptive Intelligence score for training inclusion
        
        Returns:
            Training data package
        """
        training_items = [
            item for item in content_items
            if item.get("adaptive_score", {}).get("overall", 0) >= min_score
        ]
        
        # Extract training examples
        training_examples = []
        for item in training_items:
            example = {
                "text": f"{item.get('title', '')}\n\n{item.get('description', '')}",
                "metadata": {
                    "source": item.get("source"),
                    "category": item.get("category"),
                    "score": item.get("adaptive_score", {}).get("overall"),
                    "keywords": item.get("keywords", [])
                }
            }
            training_examples.append(example)
        
        return {
            "total_items": len(content_items),
            "training_items": len(training_items),
            "examples": training_examples,
            "created_at": datetime.now().isoformat(),
            "quality_threshold": min_score
        }


# Global instance
content_processor = ContentProcessor()
