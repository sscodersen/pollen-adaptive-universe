from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime
import hashlib
import re
import uuid

from backend.database import AIContentDetection


class AIContentDetector:
    def __init__(self):
        self.ai_patterns = [
            r"as an ai",
            r"i don't have personal",
            r"i cannot feel",
            r"my training data",
            r"as a language model",
            r"i apologize, but i",
            r"unfortunately, i cannot",
            r"i'm not able to"
        ]
        
        self.human_indicators = [
            r"personally,",
            r"in my experience",
            r"i feel that",
            r"i remember when",
            r"tbh",
            r"imo",
            r"imho"
        ]
    
    def analyze_content(
        self,
        db: Session,
        content: str,
        content_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        content_id = str(uuid.uuid4())
        
        existing = db.query(AIContentDetection).filter(
            AIContentDetection.content_hash == content_hash
        ).first()
        
        if existing:
            return self._detection_to_dict(existing)
        
        ai_score = self._calculate_ai_probability(content)
        human_score = 100 - ai_score
        confidence = abs(ai_score - 50) * 2
        
        features = self._extract_features(content)
        
        detection = AIContentDetection(
            content_id=content_id,
            content_hash=content_hash,
            content_type=content_type,
            ai_generated_probability=ai_score,
            human_generated_probability=human_score,
            detection_confidence=confidence,
            model_used="pollen_detector_v1",
            features_detected=features,
            detection_metadata=metadata or {},
            verification_status="pending"
        )
        
        db.add(detection)
        db.commit()
        db.refresh(detection)
        
        return self._detection_to_dict(detection)
    
    def verify_content(
        self,
        db: Session,
        content_id: str,
        is_ai_generated: bool,
        verifier_notes: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        detection = db.query(AIContentDetection).filter(
            AIContentDetection.content_id == content_id
        ).first()
        
        if not detection:
            return None
        
        detection.verified_by_human = True
        detection.verification_status = "ai_generated" if is_ai_generated else "human_generated"
        
        if verifier_notes:
            current_metadata = detection.detection_metadata if detection.detection_metadata else {}
            detection.detection_metadata = {
                **current_metadata,
                "verifier_notes": verifier_notes,
                "verified_at": datetime.utcnow().isoformat()
            }
        
        db.commit()
        db.refresh(detection)
        
        return self._detection_to_dict(detection)
    
    def get_detections(
        self,
        db: Session,
        min_ai_probability: Optional[float] = None,
        verification_status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        query = db.query(AIContentDetection)
        
        if min_ai_probability is not None:
            query = query.filter(
                AIContentDetection.ai_generated_probability >= min_ai_probability
            )
        
        if verification_status:
            query = query.filter(
                AIContentDetection.verification_status == verification_status
            )
        
        detections = query.order_by(
            AIContentDetection.created_at.desc()
        ).limit(limit).all()
        
        return [self._detection_to_dict(d) for d in detections]
    
    def _calculate_ai_probability(self, content: str) -> float:
        content_lower = content.lower()
        
        ai_matches = sum(
            1 for pattern in self.ai_patterns
            if re.search(pattern, content_lower)
        )
        
        human_matches = sum(
            1 for pattern in self.human_indicators
            if re.search(pattern, content_lower)
        )
        
        words = content.split()
        word_count = len(words)
        
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / max(word_count, 1)
        
        base_score = 50.0
        
        base_score += ai_matches * 15
        base_score -= human_matches * 15
        
        if avg_sentence_length > 25:
            base_score += 10
        elif avg_sentence_length < 10:
            base_score -= 5
        
        if vocabulary_diversity > 0.7:
            base_score -= 10
        elif vocabulary_diversity < 0.4:
            base_score += 10
        
        if len(content) > 1000 and content.count('\n\n') / len(content) > 0.01:
            base_score += 5
        
        return max(0, min(100, base_score))
    
    def _extract_features(self, content: str) -> Dict[str, Any]:
        words = content.split()
        sentences = content.count('.') + content.count('!') + content.count('?')
        
        return {
            "word_count": len(words),
            "sentence_count": sentences,
            "avg_sentence_length": len(words) / max(sentences, 1),
            "vocabulary_diversity": len(set(words)) / max(len(words), 1),
            "has_ai_patterns": any(
                re.search(pattern, content.lower())
                for pattern in self.ai_patterns
            ),
            "has_human_indicators": any(
                re.search(pattern, content.lower())
                for pattern in self.human_indicators
            )
        }
    
    def _detection_to_dict(self, detection: AIContentDetection) -> Dict[str, Any]:
        return {
            "content_id": detection.content_id,
            "content_hash": detection.content_hash,
            "content_type": detection.content_type,
            "ai_generated_probability": detection.ai_generated_probability,
            "human_generated_probability": detection.human_generated_probability,
            "detection_confidence": detection.detection_confidence,
            "model_used": detection.model_used,
            "features_detected": detection.features_detected,
            "metadata": detection.detection_metadata,
            "verified_by_human": detection.verified_by_human,
            "verification_status": detection.verification_status,
            "created_at": detection.created_at.isoformat(),
            "label": self._get_label(detection.ai_generated_probability)
        }
    
    def _get_label(self, ai_probability: float) -> str:
        if ai_probability >= 75:
            return "Likely AI-Generated"
        elif ai_probability >= 60:
            return "Possibly AI-Generated"
        elif ai_probability >= 40:
            return "Uncertain"
        elif ai_probability >= 25:
            return "Possibly Human-Generated"
        else:
            return "Likely Human-Generated"


ai_content_detector = AIContentDetector()
