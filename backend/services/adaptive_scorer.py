"""
Adaptive Intelligence Worker Bee Content Scoring System

Evaluates content using multidimensional criteria:
- Scope: The number of individuals affected by the event
- Intensity: The magnitude of the event's impact
- Originality: The unexpected or distinctive nature of the event
- Immediacy: The temporal proximity of the event
- Practicability: The likelihood that readers can take actionable steps
- Positivity: An evaluation of the event's positive aspects
- Credibility: An assessment of the source's reliability
"""

from typing import Dict, Any
from datetime import datetime, timedelta
import re
from dataclasses import dataclass


@dataclass
class AdaptiveScore:
    scope: float  # 0-100
    intensity: float  # 0-100
    originality: float  # 0-100
    immediacy: float  # 0-100
    practicability: float  # 0-100
    positivity: float  # 0-100
    credibility: float  # 0-100
    overall: float  # Weighted average
    quality_tier: str  # "HIGH QUALITY", "MEDIUM QUALITY", "LOW QUALITY"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "intensity": self.intensity,
            "originality": self.originality,
            "immediacy": self.immediacy,
            "practicability": self.practicability,
            "positivity": self.positivity,
            "credibility": self.credibility,
            "overall": self.overall,
            "quality_tier": self.quality_tier
        }


class AdaptiveScorer:
    def __init__(self):
        # Weighted importance of each criterion
        self.weights = {
            "scope": 0.15,
            "intensity": 0.15,
            "originality": 0.15,
            "immediacy": 0.10,
            "practicability": 0.15,
            "positivity": 0.10,
            "credibility": 0.20
        }
        
        # Credible sources database
        self.credible_sources = {
            "reuters": 95, "bbc": 90, "ap": 95, "npr": 85,
            "the guardian": 85, "new york times": 85, "wall street journal": 85,
            "economist": 90, "nature": 95, "science": 95, "ieee": 90,
            "techcrunch": 75, "wired": 80, "ars technica": 80,
            "bloomberg": 85, "financial times": 85
        }
        
    def score_content(self, content: Dict[str, Any]) -> AdaptiveScore:
        """
        Score content using Adaptive Intelligence Worker Bee multidimensional criteria
        
        Args:
            content: Dictionary with keys: title, description, source, published_at, 
                    category, engagement_metrics (optional)
        
        Returns:
            AdaptiveScore object with all scores
        """
        scope_score = self._calculate_scope(content)
        intensity_score = self._calculate_intensity(content)
        originality_score = self._calculate_originality(content)
        immediacy_score = self._calculate_immediacy(content)
        practicability_score = self._calculate_practicability(content)
        positivity_score = self._calculate_positivity(content)
        credibility_score = self._calculate_credibility(content)
        
        # Calculate weighted overall score
        overall = (
            scope_score * self.weights["scope"] +
            intensity_score * self.weights["intensity"] +
            originality_score * self.weights["originality"] +
            immediacy_score * self.weights["immediacy"] +
            practicability_score * self.weights["practicability"] +
            positivity_score * self.weights["positivity"] +
            credibility_score * self.weights["credibility"]
        )
        
        # Determine quality tier
        if overall >= 70:
            quality_tier = "HIGH QUALITY"
        elif overall >= 50:
            quality_tier = "MEDIUM QUALITY"
        else:
            quality_tier = "LOW QUALITY"
        
        return AdaptiveScore(
            scope=round(scope_score, 2),
            intensity=round(intensity_score, 2),
            originality=round(originality_score, 2),
            immediacy=round(immediacy_score, 2),
            practicability=round(practicability_score, 2),
            positivity=round(positivity_score, 2),
            credibility=round(credibility_score, 2),
            overall=round(overall, 2),
            quality_tier=quality_tier
        )
    
    def _calculate_scope(self, content: Dict[str, Any]) -> float:
        """
        Calculate scope: number of individuals affected
        """
        title = content.get("title", "").lower()
        description = content.get("description", "").lower()
        text = f"{title} {description}"
        
        score = 50  # Base score
        
        # Global/international keywords
        global_keywords = ["global", "worldwide", "international", "world", "billions", "millions", "humanity"]
        if any(keyword in text for keyword in global_keywords):
            score += 30
        
        # National keywords
        national_keywords = ["national", "country", "nation", "federal", "government"]
        if any(keyword in text for keyword in national_keywords):
            score += 20
        
        # Regional keywords
        regional_keywords = ["regional", "state", "province", "city", "local"]
        if any(keyword in text for keyword in regional_keywords):
            score += 10
        
        # Industry/sector keywords
        sector_keywords = ["industry", "sector", "market", "businesses", "companies"]
        if any(keyword in text for keyword in sector_keywords):
            score += 15
        
        return min(score, 100)
    
    def _calculate_intensity(self, content: Dict[str, Any]) -> float:
        """
        Calculate intensity: magnitude of impact
        """
        title = content.get("title", "").lower()
        description = content.get("description", "").lower()
        text = f"{title} {description}"
        
        score = 50  # Base score
        
        # High impact keywords
        high_impact = ["breakthrough", "revolutionary", "unprecedented", "crisis", "emergency", 
                      "critical", "massive", "dramatic", "shocking", "devastating"]
        score += sum(10 for keyword in high_impact if keyword in text)
        
        # Medium impact keywords
        medium_impact = ["significant", "major", "important", "notable", "substantial"]
        score += sum(5 for keyword in medium_impact if keyword in text)
        
        # Numbers indicating scale
        if re.search(r'\b(billion|trillion|million)\b', text):
            score += 20
        
        # Engagement metrics if available
        engagement = content.get("engagement_metrics", {})
        if engagement.get("views", 0) > 100000:
            score += 15
        
        return min(score, 100)
    
    def _calculate_originality(self, content: Dict[str, Any]) -> float:
        """
        Calculate originality: unexpected or distinctive nature
        """
        title = content.get("title", "").lower()
        description = content.get("description", "").lower()
        text = f"{title} {description}"
        
        score = 50  # Base score
        
        # Innovation keywords
        innovation_keywords = ["first", "new", "novel", "innovative", "discovery", "invention",
                             "breakthrough", "unique", "unprecedented", "pioneering"]
        score += sum(8 for keyword in innovation_keywords if keyword in text)
        
        # Research and scientific advancement
        research_keywords = ["study", "research", "scientists", "discovered", "found"]
        if any(keyword in text for keyword in research_keywords):
            score += 15
        
        # Avoid common/repetitive topics
        common_keywords = ["another", "again", "continues", "ongoing", "usual"]
        score -= sum(5 for keyword in common_keywords if keyword in text)
        
        return max(min(score, 100), 0)
    
    def _calculate_immediacy(self, content: Dict[str, Any]) -> float:
        """
        Calculate immediacy: temporal proximity
        """
        published_at = content.get("published_at")
        
        if not published_at:
            return 50  # Default if no timestamp
        
        try:
            if isinstance(published_at, str):
                pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            else:
                pub_date = published_at
            
            now = datetime.now(pub_date.tzinfo) if pub_date.tzinfo else datetime.now()
            age = now - pub_date
            
            # Score based on age
            if age < timedelta(hours=1):
                return 100
            elif age < timedelta(hours=6):
                return 90
            elif age < timedelta(hours=24):
                return 80
            elif age < timedelta(days=2):
                return 70
            elif age < timedelta(days=7):
                return 50
            else:
                return max(30 - (age.days * 2), 10)
        except:
            return 50
    
    def _calculate_practicability(self, content: Dict[str, Any]) -> float:
        """
        Calculate practicability: actionable steps for readers
        """
        title = content.get("title", "").lower()
        description = content.get("description", "").lower()
        text = f"{title} {description}"
        
        score = 40  # Base score
        
        # Actionable keywords
        action_keywords = ["how to", "guide", "tips", "ways to", "steps", "can help",
                          "should", "could", "advice", "recommendation", "solution"]
        score += sum(8 for keyword in action_keywords if keyword in text)
        
        # Personal benefit indicators
        benefit_keywords = ["save", "earn", "improve", "protect", "benefit", "advantage",
                          "opportunity", "cheaper", "better", "faster"]
        score += sum(6 for keyword in benefit_keywords if keyword in text)
        
        # Questions (often lead to actionable answers)
        if "?" in title:
            score += 10
        
        # Category-based adjustments
        category = content.get("category", "").lower()
        if category in ["education", "health", "finance", "technology"]:
            score += 15
        
        return min(score, 100)
    
    def _calculate_positivity(self, content: Dict[str, Any]) -> float:
        """
        Calculate positivity: counteracting negativity bias
        """
        title = content.get("title", "").lower()
        description = content.get("description", "").lower()
        text = f"{title} {description}"
        
        score = 50  # Neutral baseline
        
        # Positive keywords
        positive_keywords = ["success", "achievement", "progress", "improvement", "solution",
                           "innovation", "breakthrough", "celebrate", "wins", "growth",
                           "hope", "opportunity", "benefit", "advance", "prosper"]
        score += sum(5 for keyword in positive_keywords if keyword in text)
        
        # Negative keywords (reduce score)
        negative_keywords = ["crisis", "disaster", "failure", "collapse", "threat",
                           "danger", "risk", "problem", "issue", "concern", "fear",
                           "worse", "decline", "loss", "death"]
        score -= sum(4 for keyword in negative_keywords if keyword in text)
        
        # Balanced reporting (good sign)
        if "however" in text or "but" in text or "although" in text:
            score += 5
        
        return max(min(score, 100), 0)
    
    def _calculate_credibility(self, content: Dict[str, Any]) -> float:
        """
        Calculate credibility: source reliability
        """
        source = content.get("source", "").lower()
        
        # Check against credible sources database
        for credible_source, base_score in self.credible_sources.items():
            if credible_source in source:
                return base_score
        
        # Default scoring for unknown sources
        score = 50
        
        # Check for credibility indicators in content
        title = content.get("title", "").lower()
        description = content.get("description", "").lower()
        text = f"{title} {description}"
        
        # Academic/research indicators
        academic_keywords = ["study", "research", "university", "journal", "peer-reviewed",
                           "scientists", "researchers", "professor"]
        score += sum(5 for keyword in academic_keywords if keyword in text)
        
        # Official sources
        official_keywords = ["official", "government", "agency", "department", "ministry"]
        score += sum(5 for keyword in official_keywords if keyword in text)
        
        # Red flags for low credibility
        red_flags = ["rumor", "allegedly", "unconfirmed", "claims", "conspiracy"]
        score -= sum(10 for flag in red_flags if flag in text)
        
        return max(min(score, 100), 0)


# Global instance
adaptive_scorer = AdaptiveScorer()
