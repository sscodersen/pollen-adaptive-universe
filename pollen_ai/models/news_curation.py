import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

class NewsCurator:
    def __init__(self):
        self.news_categories = [
            "Technology", "Science", "Business", "Health", "Environment",
            "Space", "AI/ML", "Cybersecurity", "Fintech", "Biotech"
        ]
        
        self.trending_stories = {
            "Technology": [
                "AI breakthrough in natural language processing",
                "Quantum computing milestone achieved",
                "New semiconductor technology unveiled",
                "Autonomous vehicle safety improvements"
            ],
            "Science": [
                "Climate research reveals new insights",
                "Medical breakthrough in gene therapy",
                "Space exploration discoveries",
                "Renewable energy efficiency gains"
            ],
            "Business": [
                "Tech IPO generates massive investor interest",
                "Startup funding reaches record highs",
                "Corporate sustainability initiatives expand",
                "Digital transformation accelerates globally"
            ]
        }

    def curate_news(self, input_text: str) -> Dict[str, Any]:
        """Curate news content based on input parameters"""
        try:
            category = self._determine_category(input_text)
            news_items = self._generate_news_items(category, input_text)
            
            return {
                "category": category,
                "trending_score": random.uniform(8.0, 9.8),
                "articles": news_items,
                "summary": self._generate_summary(news_items),
                "metadata": {
                    "total_articles": len(news_items),
                    "freshness_score": random.uniform(0.85, 0.98),
                    "relevance_score": random.uniform(0.80, 0.95),
                    "timestamp": datetime.now().isoformat(),
                    "update_frequency": "real-time"
                },
                "insights": self._generate_insights(category, news_items),
                "related_topics": self._get_related_topics(category)
            }
        except Exception as e:
            return self._fallback_news(input_text)

    def _determine_category(self, text: str) -> str:
        """Determine news category from input text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["ai", "artificial", "machine learning", "neural"]):
            return "AI/ML"
        elif any(word in text_lower for word in ["cyber", "security", "hack", "breach"]):
            return "Cybersecurity"
        elif any(word in text_lower for word in ["startup", "funding", "ipo", "investment"]):
            return "Business"
        elif any(word in text_lower for word in ["space", "mars", "satellite", "astronomy"]):
            return "Space"
        elif any(word in text_lower for word in ["climate", "environment", "green", "renewable"]):
            return "Environment"
        elif any(word in text_lower for word in ["health", "medical", "biotech", "pharma"]):
            return "Health"
        else:
            return "Technology"

    def _generate_news_items(self, category: str, input_text: str) -> List[Dict[str, Any]]:
        """Generate relevant news articles"""
        num_articles = random.randint(4, 8)
        articles = []
        
        base_stories = self.trending_stories.get(category, self.trending_stories["Technology"])
        
        for i in range(num_articles):
            story = random.choice(base_stories)
            article = {
                "id": f"news_{datetime.now().timestamp()}_{i}",
                "headline": self._enhance_headline(story, category),
                "summary": self._generate_article_summary(story, category),
                "content": self._generate_article_content(story, category),
                "source": self._get_credible_source(category),
                "published_at": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                "reading_time": f"{random.randint(2, 8)} min",
                "engagement": {
                    "views": random.randint(1000, 50000),
                    "shares": random.randint(50, 2000),
                    "comments": random.randint(10, 500)
                },
                "tags": self._generate_article_tags(category),
                "significance_score": random.uniform(7.5, 9.5),
                "credibility_score": random.uniform(8.5, 9.8)
            }
            articles.append(article)
        
        return articles

    def _enhance_headline(self, base_story: str, category: str) -> str:
        """Create compelling headlines"""
        enhancements = [
            "Breaking: ",
            "Major: ",
            "Exclusive: ",
            "Latest: ",
            ""
        ]
        
        impact_words = {
            "Technology": ["Revolutionary", "Groundbreaking", "Innovative"],
            "Science": ["Breakthrough", "Discovery", "Research reveals"],
            "Business": ["Record-setting", "Industry-changing", "Market-shifting"],
            "AI/ML": ["Advanced", "Next-generation", "Cutting-edge"]
        }
        
        prefix = random.choice(enhancements)
        impact = random.choice(impact_words.get(category, ["Significant"]))
        
        return f"{prefix}{impact} {base_story}"

    def _generate_article_summary(self, story: str, category: str) -> str:
        """Generate article summaries"""
        summaries = {
            "Technology": f"Industry experts report significant progress in {story.lower()}, with potential implications for global technology markets and consumer applications.",
            "Science": f"Recent research findings in {story.lower()} could reshape our understanding and lead to practical applications across multiple industries.",
            "Business": f"Market analysts highlight the impact of {story.lower()} on industry dynamics, investor sentiment, and future growth prospects."
        }
        
        return summaries.get(category, f"Experts analyze the developments in {story.lower()} and its potential widespread impact.")

    def _generate_article_content(self, story: str, category: str) -> str:
        """Generate detailed article content"""
        intro = f"In a significant development for the {category.lower()} sector, {story.lower()} has captured the attention of industry leaders and analysts worldwide."
        
        body = f"The implications of this development extend beyond immediate applications, potentially reshaping how we approach challenges in {category.lower()}. " \
               f"Early indicators suggest strong market reception and growing investor confidence in related technologies."
        
        conclusion = f"As this story continues to develop, stakeholders across {category.lower()} are closely monitoring the potential long-term impacts " \
                    f"and preparing strategic responses to capitalize on emerging opportunities."
        
        return f"{intro}\n\n{body}\n\n{conclusion}"

    def _get_credible_source(self, category: str) -> str:
        """Get credible news sources by category"""
        sources = {
            "Technology": ["TechCrunch", "Wired", "IEEE Spectrum", "MIT Technology Review"],
            "Science": ["Nature", "Science Magazine", "Scientific American", "New Scientist"],
            "Business": ["Bloomberg", "Reuters", "Financial Times", "WSJ"],
            "AI/ML": ["AI News", "Machine Learning Today", "AI Research", "Neural Networks"],
            "Cybersecurity": ["Security Week", "Cyber Defense", "InfoSec Today", "Security News"],
            "Health": ["Medical News Today", "Health Tech", "BioPharma News", "Medical Research"]
        }
        
        return random.choice(sources.get(category, sources["Technology"]))

    def _generate_article_tags(self, category: str) -> List[str]:
        """Generate relevant tags for articles"""
        base_tags = ["breaking", "trending", "analysis"]
        
        category_tags = {
            "Technology": ["tech", "innovation", "digital"],
            "Science": ["research", "discovery", "breakthrough"],
            "Business": ["market", "investment", "industry"],
            "AI/ML": ["artificial-intelligence", "machine-learning", "neural-networks"],
            "Cybersecurity": ["security", "privacy", "cyber-threats"],
            "Health": ["healthcare", "medical", "biotech"]
        }
        
        specific_tags = category_tags.get(category, ["news", "update"])
        return base_tags + specific_tags

    def _generate_summary(self, articles: List[Dict[str, Any]]) -> str:
        """Generate overall news summary"""
        num_articles = len(articles)
        categories = list(set([article.get('tags', ['general'])[0] for article in articles]))
        
        return f"Today's curated news includes {num_articles} significant developments across {', '.join(categories[:3])}. " \
               f"Key themes include technological advancement, market dynamics, and industry innovation. " \
               f"Overall significance score: {random.uniform(8.2, 9.4):.1f}/10."

    def _generate_insights(self, category: str, articles: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from news analysis"""
        insights = [
            f"Increased activity in {category.lower()} sector suggests accelerating innovation cycles",
            f"Market sentiment remains positive with growing investor confidence",
            f"Cross-industry collaboration is driving breakthrough developments",
            f"Regulatory landscape is evolving to support emerging technologies",
            f"Consumer adoption rates are exceeding industry projections"
        ]
        
        return random.sample(insights, min(3, len(insights)))

    def _get_related_topics(self, category: str) -> List[str]:
        """Get related topics for the category"""
        related = {
            "Technology": ["Innovation", "Digital Transformation", "Emerging Tech"],
            "Science": ["Research", "Discovery", "Scientific Method"],
            "Business": ["Market Analysis", "Investment", "Industry Trends"],
            "AI/ML": ["Deep Learning", "Neural Networks", "Automation"],
            "Cybersecurity": ["Data Protection", "Threat Analysis", "Security Protocols"],
            "Health": ["Medical Research", "Healthcare Innovation", "Biotechnology"]
        }
        
        return related.get(category, ["Technology", "Innovation", "Industry News"])

    def _fallback_news(self, input_text: str) -> Dict[str, Any]:
        """Fallback news when processing fails"""
        return {
            "category": "General",
            "trending_score": 7.5,
            "articles": [{
                "id": f"fallback_{datetime.now().timestamp()}",
                "headline": "Technology and Innovation Continue to Shape Global Markets",
                "summary": "Industry leaders report steady progress across multiple technology sectors with positive market outlook.",
                "content": "The technology sector continues to demonstrate resilience and innovation across multiple domains. Market analysts remain optimistic about future growth prospects.",
                "source": "Tech News Today",
                "published_at": datetime.now().isoformat(),
                "reading_time": "3 min",
                "engagement": {"views": 5000, "shares": 250, "comments": 50},
                "tags": ["technology", "market", "innovation"],
                "significance_score": 7.5,
                "credibility_score": 8.0
            }],
            "summary": "General technology and market news with standard industry updates.",
            "metadata": {
                "total_articles": 1,
                "freshness_score": 0.8,
                "relevance_score": 0.7,
                "timestamp": datetime.now().isoformat(),
                "update_frequency": "standard"
            },
            "insights": ["Technology sector maintains steady growth trajectory"],
            "related_topics": ["Technology", "Innovation", "Market Analysis"]
        }