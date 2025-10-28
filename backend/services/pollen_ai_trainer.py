"""
Pollen AI Training Service
Handles incremental training and knowledge base updates
"""

from typing import Dict, List, Any, AsyncGenerator, Optional
from datetime import datetime
import json
import asyncio


class PollenAITrainer:
    def __init__(self):
        self.training_queue = []
        self.training_history = []
        self.knowledge_base = {}
        
    async def stream_incremental_training(
        self,
        training_data: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Stream incremental training updates via SSE
        
        Args:
            training_data: Prepared training data from ContentProcessor
        """
        try:
            examples = training_data.get("examples", [])
            total = len(examples)
            
            yield json.dumps({
                "type": "training_start",
                "message": f"🧠 Starting Pollen AI training with {total} examples...",
                "total_examples": total
            }) + "\n"
            
            # Process training examples
            for i, example in enumerate(examples):
                # Simulate training step
                await asyncio.sleep(0.1)
                
                # Update knowledge base
                category = example.get("metadata", {}).get("category", "general")
                keywords = example.get("metadata", {}).get("keywords", [])
                
                if category not in self.knowledge_base:
                    self.knowledge_base[category] = {
                        "examples": [],
                        "keywords": set(),
                        "last_updated": datetime.now().isoformat()
                    }
                
                self.knowledge_base[category]["examples"].append(example)
                self.knowledge_base[category]["keywords"].update(keywords)
                self.knowledge_base[category]["last_updated"] = datetime.now().isoformat()
                
                # Stream progress
                yield json.dumps({
                    "type": "training_progress",
                    "progress": ((i + 1) / total) * 100,
                    "current": i + 1,
                    "total": total,
                    "category": category,
                    "score": example.get("metadata", {}).get("score", 0)
                }) + "\n"
            
            # Training complete
            training_session = {
                "session_id": hash(datetime.now().isoformat()),
                "timestamp": datetime.now().isoformat(),
                "examples_processed": total,
                "categories_updated": list(self.knowledge_base.keys()),
                "quality_threshold": training_data.get("quality_threshold", 70)
            }
            
            self.training_history.append(training_session)
            
            yield json.dumps({
                "type": "training_complete",
                "message": "✅ Training session complete!",
                "session": training_session,
                "knowledge_base_size": sum(
                    len(kb["examples"]) for kb in self.knowledge_base.values()
                )
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": f"Training error: {str(e)}"
            }) + "\n"
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Get summary of current knowledge base
        """
        return {
            "total_categories": len(self.knowledge_base),
            "categories": {
                category: {
                    "examples": len(data["examples"]),
                    "keywords": len(data["keywords"]),
                    "last_updated": data["last_updated"]
                }
                for category, data in self.knowledge_base.items()
            },
            "training_sessions": len(self.training_history),
            "last_training": self.training_history[-1] if self.training_history else None
        }
    
    async def generate_enhanced_response(
        self,
        query: str,
        category: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate AI response enhanced with trained knowledge
        """
        try:
            # Find relevant knowledge
            relevant_knowledge = []
            
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for cat, data in self.knowledge_base.items():
                if category and cat != category:
                    continue
                
                # Find examples with matching keywords
                cat_keywords = data.get("keywords", set())
                keyword_overlap = len(query_words & cat_keywords)
                
                if keyword_overlap > 0:
                    relevant_knowledge.append({
                        "category": cat,
                        "relevance": keyword_overlap,
                        "examples": data["examples"][:3]  # Top 3 examples
                    })
            
            # Sort by relevance
            relevant_knowledge.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Generate response
            yield json.dumps({
                "type": "response_start",
                "message": "Generating response with trained knowledge..."
            }) + "\n"
            
            if relevant_knowledge:
                yield json.dumps({
                    "type": "knowledge_context",
                    "message": f"Found {len(relevant_knowledge)} relevant knowledge areas",
                    "categories": [k["category"] for k in relevant_knowledge[:3]]
                }) + "\n"
            
            # Simulate enhanced response generation
            response = f"Based on my knowledge from {len(self.training_history)} training sessions"
            if relevant_knowledge:
                response += f" and {sum(k['relevance'] for k in relevant_knowledge)} relevant examples"
            response += f", here's what I can tell you about '{query}':\n\n"
            
            yield json.dumps({
                "type": "response_chunk",
                "text": response
            }) + "\n"
            
            yield json.dumps({
                "type": "response_complete",
                "message": "Response generated"
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": str(e)
            }) + "\n"


# Global instance
pollen_ai_trainer = PollenAITrainer()
