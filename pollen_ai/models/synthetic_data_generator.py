"""
Synthetic Data Generation for Pollen AI
Generates training data for continuous learning across all domains
"""

from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta
import json


class SyntheticDataGenerator:
    """Generate synthetic training data for Pollen AI"""
    
    def __init__(self):
        self.generated_samples = []
        self.generation_stats = {
            "text": 0,
            "audio": 0,
            "image": 0,
            "video": 0,
            "code": 0,
            "game": 0,
            "smart_home": 0,
            "robot": 0
        }
    
    def generate_text_data(self, domain: str, count: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic text data for specific domain"""
        samples = []
        
        templates = self._get_text_templates(domain)
        
        for i in range(count):
            template = random.choice(templates)
            sample = {
                "id": f"text_{domain}_{len(self.generated_samples) + i}",
                "domain": domain,
                "type": "text",
                "content": template.format(
                    topic=self._generate_topic(domain),
                    detail=self._generate_detail(domain)
                ),
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "quality_score": random.uniform(0.7, 0.95),
                    "difficulty": random.choice(["beginner", "intermediate", "advanced"])
                },
                "tags": self._generate_tags(domain)
            }
            samples.append(sample)
        
        self.generated_samples.extend(samples)
        self.generation_stats["text"] += count
        
        return samples
    
    def generate_audio_data(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic audio training data"""
        samples = []
        
        audio_types = ["music", "speech", "ambient", "sound_effect"]
        
        for i in range(count):
            audio_type = random.choice(audio_types)
            sample = {
                "id": f"audio_{len(self.generated_samples) + i}",
                "type": "audio",
                "audio_type": audio_type,
                "description": self._generate_audio_description(audio_type),
                "metadata": {
                    "duration_seconds": random.randint(30, 300),
                    "sample_rate": random.choice([44100, 48000]),
                    "channels": random.choice([1, 2]),
                    "format": random.choice(["mp3", "wav", "ogg"]),
                    "generated_at": datetime.now().isoformat()
                },
                "tags": self._generate_tags("audio")
            }
            samples.append(sample)
        
        self.generated_samples.extend(samples)
        self.generation_stats["audio"] += count
        
        return samples
    
    def generate_image_data(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic image training data"""
        samples = []
        
        image_types = ["product", "scene", "portrait", "abstract", "3d_render"]
        
        for i in range(count):
            image_type = random.choice(image_types)
            sample = {
                "id": f"image_{len(self.generated_samples) + i}",
                "type": "image",
                "image_type": image_type,
                "description": self._generate_image_description(image_type),
                "metadata": {
                    "width": random.choice([512, 768, 1024, 2048]),
                    "height": random.choice([512, 768, 1024, 2048]),
                    "format": random.choice(["png", "jpg", "webp"]),
                    "style": random.choice(["realistic", "artistic", "technical", "abstract"]),
                    "generated_at": datetime.now().isoformat()
                },
                "tags": self._generate_tags("image")
            }
            samples.append(sample)
        
        self.generated_samples.extend(samples)
        self.generation_stats["image"] += count
        
        return samples
    
    def generate_code_data(self, language: str = "python", count: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic code training data"""
        samples = []
        
        code_patterns = self._get_code_patterns(language)
        
        for i in range(count):
            pattern = random.choice(code_patterns)
            sample = {
                "id": f"code_{language}_{len(self.generated_samples) + i}",
                "type": "code",
                "language": language,
                "code": pattern,
                "description": f"Code example for {language}",
                "metadata": {
                    "complexity": random.choice(["simple", "moderate", "complex"]),
                    "lines": len(pattern.split("\n")),
                    "generated_at": datetime.now().isoformat()
                },
                "tags": self._generate_tags("code")
            }
            samples.append(sample)
        
        self.generated_samples.extend(samples)
        self.generation_stats["code"] += count
        
        return samples
    
    def generate_game_data(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic game design data"""
        samples = []
        
        game_genres = ["action", "puzzle", "strategy", "rpg", "simulation"]
        
        for i in range(count):
            genre = random.choice(game_genres)
            sample = {
                "id": f"game_{len(self.generated_samples) + i}",
                "type": "game",
                "genre": genre,
                "title": self._generate_game_title(genre),
                "description": self._generate_game_description(genre),
                "mechanics": self._generate_game_mechanics(genre),
                "metadata": {
                    "target_platform": random.choice(["mobile", "desktop", "web", "vr"]),
                    "estimated_dev_time": f"{random.randint(1, 12)} months",
                    "difficulty": random.choice(["casual", "moderate", "hardcore"]),
                    "generated_at": datetime.now().isoformat()
                },
                "tags": self._generate_tags("game")
            }
            samples.append(sample)
        
        self.generated_samples.extend(samples)
        self.generation_stats["game"] += count
        
        return samples
    
    def generate_training_batch(self, batch_size: int = 50) -> Dict[str, Any]:
        """Generate balanced batch of training data across all domains"""
        batch = {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "batch_size": batch_size,
            "domains": {},
            "generated_at": datetime.now().isoformat()
        }
        
        samples_per_domain = batch_size // 8  # 8 domains
        
        batch["domains"]["text_wellness"] = self.generate_text_data("wellness", samples_per_domain)
        batch["domains"]["text_product"] = self.generate_text_data("product", samples_per_domain)
        batch["domains"]["audio"] = self.generate_audio_data(samples_per_domain)
        batch["domains"]["image"] = self.generate_image_data(samples_per_domain)
        batch["domains"]["code"] = self.generate_code_data("python", samples_per_domain)
        batch["domains"]["game"] = self.generate_game_data(samples_per_domain)
        
        return batch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "total_samples": len(self.generated_samples),
            "by_type": self.generation_stats,
            "latest_batch_time": datetime.now().isoformat()
        }
    
    # Helper methods for generating realistic synthetic data
    
    def _get_text_templates(self, domain: str) -> List[str]:
        """Get text templates for domain"""
        templates = {
            "wellness": [
                "Practice {topic} for better {detail}. Start with small steps and build consistency.",
                "Discover the benefits of {topic} in improving your {detail}.",
                "{topic} techniques can significantly enhance your {detail} and overall wellbeing."
            ],
            "product": [
                "Introducing the revolutionary {topic} designed to enhance your {detail}.",
                "Premium {topic} featuring advanced {detail} for modern users.",
                "Experience the next generation of {topic} with innovative {detail}."
            ],
            "entertainment": [
                "An exciting exploration of {topic} that will keep you engaged with {detail}.",
                "{topic}: A compelling story featuring {detail} and unexpected twists.",
                "Discover the world of {topic} through stunning {detail}."
            ]
        }
        return templates.get(domain, ["Sample content about {topic} and {detail}."])
    
    def _generate_topic(self, domain: str) -> str:
        """Generate topic for domain"""
        topics = {
            "wellness": ["mindfulness", "nutrition", "exercise", "sleep", "stress management"],
            "product": ["smart technology", "eco-friendly materials", "ergonomic design", "AI integration"],
            "entertainment": ["adventure", "mystery", "sci-fi", "drama", "comedy"]
        }
        return random.choice(topics.get(domain, ["general topic"]))
    
    def _generate_detail(self, domain: str) -> str:
        """Generate detail for domain"""
        details = {
            "wellness": ["mental health", "physical fitness", "emotional balance", "energy levels"],
            "product": ["user experience", "performance", "sustainability", "efficiency"],
            "entertainment": ["visual effects", "storytelling", "character development", "immersive experience"]
        }
        return random.choice(details.get(domain, ["quality"]))
    
    def _generate_tags(self, domain: str) -> List[str]:
        """Generate relevant tags"""
        tag_pool = {
            "text": ["quality", "informative", "engaging", "relevant"],
            "audio": ["professional", "clear", "balanced", "dynamic"],
            "image": ["high_resolution", "detailed", "vibrant", "professional"],
            "code": ["efficient", "clean", "documented", "tested"],
            "game": ["fun", "challenging", "creative", "balanced"]
        }
        tags = tag_pool.get(domain, ["general"])
        return random.sample(tags, min(3, len(tags)))
    
    def _generate_audio_description(self, audio_type: str) -> str:
        """Generate audio description"""
        descriptions = {
            "music": "Melodic composition with rich harmonies and dynamic rhythm",
            "speech": "Clear spoken content with natural intonation and pacing",
            "ambient": "Atmospheric soundscape creating immersive environment",
            "sound_effect": "High-quality audio effect for interactive applications"
        }
        return descriptions.get(audio_type, "Audio content")
    
    def _generate_image_description(self, image_type: str) -> str:
        """Generate image description"""
        descriptions = {
            "product": "Professional product photography with optimal lighting",
            "scene": "Detailed environmental scene with realistic atmosphere",
            "portrait": "High-quality portrait with professional composition",
            "abstract": "Creative abstract design with balanced color palette",
            "3d_render": "Photorealistic 3D rendering with accurate materials"
        }
        return descriptions.get(image_type, "Image content")
    
    def _get_code_patterns(self, language: str) -> List[str]:
        """Get code patterns for language"""
        patterns = {
            "python": [
                "def process_data(data):\n    result = [x * 2 for x in data]\n    return result",
                "class DataProcessor:\n    def __init__(self):\n        self.data = []\n    \n    def add(self, item):\n        self.data.append(item)",
                "import numpy as np\n\ndef calculate_stats(arr):\n    return {\n        'mean': np.mean(arr),\n        'std': np.std(arr)\n    }"
            ],
            "javascript": [
                "function processData(data) {\n    return data.map(x => x * 2);\n}",
                "const fetchData = async () => {\n    const response = await fetch('/api/data');\n    return response.json();\n}",
                "class DataManager {\n    constructor() {\n        this.items = [];\n    }\n    add(item) {\n        this.items.push(item);\n    }\n}"
            ]
        }
        return patterns.get(language, ["# Sample code"])
    
    def _generate_game_title(self, genre: str) -> str:
        """Generate game title"""
        prefixes = ["Epic", "Mystic", "Cosmic", "Ancient", "Digital", "Quantum"]
        suffixes = ["Quest", "Warriors", "Odyssey", "Chronicles", "Legends", "Adventure"]
        return f"{random.choice(prefixes)} {random.choice(suffixes)}"
    
    def _generate_game_description(self, genre: str) -> str:
        """Generate game description"""
        templates = {
            "action": "Fast-paced action gameplay with intense combat and strategic elements",
            "puzzle": "Mind-bending puzzles that challenge logic and problem-solving skills",
            "strategy": "Deep strategic gameplay requiring planning and resource management",
            "rpg": "Immersive role-playing experience with character progression and storyline",
            "simulation": "Realistic simulation providing authentic management experience"
        }
        return templates.get(genre, "Engaging gameplay experience")
    
    def _generate_game_mechanics(self, genre: str) -> List[str]:
        """Generate game mechanics"""
        mechanics = {
            "action": ["real-time combat", "power-ups", "combo system", "boss battles"],
            "puzzle": ["pattern matching", "logic chains", "time pressure", "progressive difficulty"],
            "strategy": ["resource management", "unit control", "base building", "tech trees"],
            "rpg": ["character stats", "skill trees", "quest system", "inventory management"],
            "simulation": ["realistic physics", "economy system", "progression", "customization"]
        }
        return random.sample(mechanics.get(genre, ["basic gameplay"]), 3)
