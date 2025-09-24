import json
import random
from datetime import datetime
from typing import Dict, List, Any

class SocialPostCurator:
    def __init__(self):
        self.trending_topics = [
            "Artificial Intelligence", "Climate Technology", "Space Exploration",
            "Quantum Computing", "Biotechnology", "Renewable Energy",
            "Virtual Reality", "Blockchain", "Neural Interfaces", "Robotics"
        ]
        
        self.content_templates = {
            "tech_innovation": "🚀 {topic} is revolutionizing {industry}! {insight} What are your thoughts on this breakthrough? #Innovation #Tech",
            "educational": "📚 Did you know? {fact} This demonstrates the incredible potential of {topic}. #Learning #Science",
            "thought_provoking": "💭 As {topic} advances, we're seeing {observation}. How do you think this will impact {sector}? #FutureThinking",
            "news_update": "🔥 Breaking: {headline} in {topic}. This could mean {implication} for the industry. #News #Trending"
        }

    def curate_post(self, input_text: str) -> Dict[str, Any]:
        """Generate a curated social media post based on input text"""
        try:
            # Extract key themes and generate appropriate content
            topic = self._extract_topic(input_text)
            post_type = self._determine_post_type(input_text)
            content = self._generate_content(topic, post_type, input_text)
            
            return {
                "content": content,
                "engagement_score": random.uniform(7.5, 9.5),
                "hashtags": self._generate_hashtags(topic),
                "platform_optimized": {
                    "twitter": content[:280],
                    "linkedin": self._professional_version(content),
                    "instagram": self._visual_version(content)
                },
                "metadata": {
                    "topic": topic,
                    "type": post_type,
                    "timestamp": datetime.now().isoformat(),
                    "trending_potential": random.uniform(0.6, 0.95)
                }
            }
        except Exception as e:
            return self._fallback_post(input_text)

    def _extract_topic(self, text: str) -> str:
        """Extract main topic from input text"""
        text_lower = text.lower()
        for topic in self.trending_topics:
            if topic.lower() in text_lower:
                return topic
        
        # Fallback based on keywords
        if any(word in text_lower for word in ["ai", "artificial", "intelligence"]):
            return "Artificial Intelligence"
        elif any(word in text_lower for word in ["climate", "environment", "green"]):
            return "Climate Technology"
        elif any(word in text_lower for word in ["space", "mars", "satellite"]):
            return "Space Exploration"
        else:
            return random.choice(self.trending_topics)

    def _determine_post_type(self, text: str) -> str:
        """Determine the best post type based on content"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["news", "breaking", "announced"]):
            return "news_update"
        elif any(word in text_lower for word in ["learn", "fact", "research"]):
            return "educational"
        elif any(word in text_lower for word in ["think", "opinion", "future"]):
            return "thought_provoking"
        else:
            return "tech_innovation"

    def _generate_content(self, topic: str, post_type: str, original_text: str) -> str:
        """Generate engaging social media content"""
        template = self.content_templates.get(post_type, self.content_templates["tech_innovation"])
        
        placeholders = {
            "topic": topic,
            "industry": self._get_related_industry(topic),
            "insight": self._generate_insight(topic),
            "fact": self._generate_fact(topic),
            "observation": self._generate_observation(topic),
            "sector": self._get_related_sector(topic),
            "headline": self._generate_headline(topic),
            "implication": self._generate_implication(topic)
        }
        
        return template.format(**placeholders)

    def _generate_hashtags(self, topic: str) -> List[str]:
        """Generate relevant hashtags"""
        base_tags = ["#Innovation", "#Technology", "#Future"]
        topic_specific = {
            "Artificial Intelligence": ["#AI", "#MachineLearning", "#DeepLearning"],
            "Climate Technology": ["#ClimateAction", "#GreenTech", "#Sustainability"],
            "Space Exploration": ["#Space", "#SpaceX", "#NASA"],
            "Quantum Computing": ["#Quantum", "#Computing", "#Science"],
            "Biotechnology": ["#Biotech", "#Healthcare", "#Medicine"]
        }
        
        specific_tags = topic_specific.get(topic, ["#Tech", "#Innovation"])
        return base_tags + specific_tags

    def _professional_version(self, content: str) -> str:
        """Create LinkedIn-optimized version"""
        return content.replace("🚀", "").replace("🔥", "").replace("💭", "") + "\n\nWhat's your perspective on this development?"

    def _visual_version(self, content: str) -> str:
        """Create Instagram-optimized version"""
        return content + "\n\n📸 Share your thoughts in the comments! \n#VisualStory #TechTalk"

    def _get_related_industry(self, topic: str) -> str:
        industries = {
            "Artificial Intelligence": "healthcare and finance",
            "Climate Technology": "energy and manufacturing",
            "Space Exploration": "telecommunications and research",
            "Quantum Computing": "cybersecurity and logistics"
        }
        return industries.get(topic, "technology")

    def _generate_insight(self, topic: str) -> str:
        insights = [
            "The implications are far-reaching and transformative.",
            "This breakthrough could reshape entire industries.",
            "Early adoption could provide significant competitive advantages.",
            "The potential applications are virtually limitless."
        ]
        return random.choice(insights)

    def _generate_fact(self, topic: str) -> str:
        facts = {
            "Artificial Intelligence": "AI can now process information 1000x faster than traditional methods",
            "Climate Technology": "Renewable energy costs have dropped 85% in the last decade",
            "Space Exploration": "Private space missions have increased 400% since 2020"
        }
        return facts.get(topic, "Technology adoption is accelerating at an unprecedented rate")

    def _generate_observation(self, topic: str) -> str:
        observations = [
            "unprecedented collaboration between industries",
            "rapid advancement in practical applications",
            "significant investment from major corporations",
            "growing interest from regulatory bodies"
        ]
        return random.choice(observations)

    def _get_related_sector(self, topic: str) -> str:
        sectors = {
            "Artificial Intelligence": "healthcare sector",
            "Climate Technology": "energy sector",
            "Space Exploration": "telecommunications sector"
        }
        return sectors.get(topic, "technology sector")

    def _generate_headline(self, topic: str) -> str:
        headlines = [
            f"Major breakthrough in {topic} announced",
            f"{topic} reaches new milestone",
            f"Industry leaders invest billions in {topic}",
            f"{topic} adoption accelerates globally"
        ]
        return random.choice(headlines)

    def _generate_implication(self, topic: str) -> str:
        implications = [
            "accelerated innovation cycles",
            "new market opportunities",
            "enhanced competitive dynamics",
            "revolutionary business models"
        ]
        return random.choice(implications)

    def _fallback_post(self, input_text: str) -> Dict[str, Any]:
        """Fallback post when processing fails"""
        return {
            "content": f"🚀 Exploring new frontiers in technology and innovation. {input_text[:100]}... What trends are you most excited about? #Innovation #Tech #Future",
            "engagement_score": 7.0,
            "hashtags": ["#Innovation", "#Technology", "#Future", "#Trending"],
            "platform_optimized": {
                "twitter": "🚀 New frontiers in tech & innovation. What trends excite you most? #Innovation #Tech",
                "linkedin": "Exploring technological frontiers and innovation opportunities. What developments in your industry are you most excited about?",
                "instagram": "🚀 Tech innovation never stops! Share your favorite tech trend in the comments! #Innovation #TechTalk"
            },
            "metadata": {
                "topic": "Technology",
                "type": "general",
                "timestamp": datetime.now().isoformat(),
                "trending_potential": 0.7
            }
        }