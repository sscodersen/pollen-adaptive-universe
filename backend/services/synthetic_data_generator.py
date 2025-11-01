"""
Synthetic Data Generation Service
Generates training data when Pollen AI encounters unknown requests
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json

from backend.database import SessionLocal
from backend.services.ai_service import ai_service
from backend.services.content_storage import content_storage
from backend.services.pollen_ai_trainer import pollen_ai_trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    def __init__(self):
        self.unknown_intents = []
        self.generation_queue = []
        self.quality_threshold = 70.0
        self.rollback_history = []
    
    async def capture_unknown_intent(self, query: str, mode: str, context: Optional[Dict] = None):
        """
        Capture an unknown intent that Pollen AI couldn't handle well
        """
        intent = {
            "query": query,
            "mode": mode,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        
        self.unknown_intents.append(intent)
        logger.info(f"📝 Captured unknown intent: {query[:50]}... (mode: {mode})")
        
        if len(self.unknown_intents) >= 5:
            await self.generate_synthetic_batch()
    
    async def generate_synthetic_batch(self):
        """
        Generate synthetic training data from unknown intents using scraped context
        """
        if not self.unknown_intents:
            return
        
        logger.info(f"🧬 Generating synthetic training data from {len(self.unknown_intents)} intents")
        
        db = SessionLocal()
        try:
            recent_content = content_storage.get_training_dataset(
                db,
                min_quality=60.0,
                max_items=100,
                unused_only=False
            )
            
            synthetic_examples = []
            
            for intent in self.unknown_intents[:10]:
                try:
                    synthetic_data = await self._generate_synthetic_example(
                        intent,
                        recent_content
                    )
                    
                    if synthetic_data and synthetic_data.get("quality_score", 0) >= self.quality_threshold:
                        synthetic_examples.append(synthetic_data)
                        intent["status"] = "generated"
                        logger.info(f"✅ Generated synthetic data: quality {synthetic_data['quality_score']:.1f}")
                    else:
                        logger.warn(f"⚠️  Low quality synthetic data, skipping")
                        intent["status"] = "low_quality"
                        
                except Exception as e:
                    logger.error(f"Error generating synthetic data: {e}")
                    intent["status"] = "error"
            
            if synthetic_examples:
                await self._store_and_train(db, synthetic_examples)
            
            self.unknown_intents = [i for i in self.unknown_intents if i["status"] == "pending"]
            
        except Exception as e:
            logger.error(f"Synthetic batch generation failed: {e}")
        finally:
            db.close()
    
    async def _generate_synthetic_example(
        self,
        intent: Dict,
        context_data: List[Dict]
    ) -> Optional[Dict]:
        """
        Generate a single synthetic training example
        """
        context_text = self._prepare_context(context_data, intent)
        
        prompt = f"""Generate a high-quality training example for this query:

Query: {intent['query']}
Mode: {intent['mode']}

Context from recent high-quality content:
{context_text}

Generate:
1. An improved, detailed response
2. Key concepts and entities
3. Related topics
4. Quality score (0-100)

Format as JSON with: response, concepts, related_topics, quality_score"""
        
        try:
            full_response = ""
            async for chunk in ai_service.stream_response(
                prompt,
                context={"mode": "synthetic_generation"}
            ):
                data = json.loads(chunk)
                if "text" in data:
                    full_response += data["text"]
            
            if "```json" in full_response:
                json_str = full_response.split("```json")[1].split("```")[0].strip()
                synthetic_data = json.loads(json_str)
            elif "{" in full_response:
                json_start = full_response.index("{")
                json_end = full_response.rindex("}") + 1
                synthetic_data = json.loads(full_response[json_start:json_end])
            else:
                synthetic_data = {
                    "response": full_response,
                    "quality_score": 60.0
                }
            
            synthetic_data["original_query"] = intent["query"]
            synthetic_data["mode"] = intent["mode"]
            synthetic_data["generated_at"] = datetime.utcnow().isoformat()
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic example: {e}")
            return None
    
    def _prepare_context(self, context_data: List[Dict], intent: Dict) -> str:
        """
        Prepare context from recent high-quality content
        """
        if not context_data:
            return "No recent context available"
        
        mode = intent.get("mode", "general")
        relevant_content = [
            c for c in context_data 
            if mode in c.get("category", "").lower() or mode in c.get("title", "").lower()
        ][:5]
        
        if not relevant_content:
            relevant_content = context_data[:5]
        
        context_lines = []
        for item in relevant_content:
            context_lines.append(
                f"- {item.get('title', '')}: {item.get('description', '')[:100]}"
            )
        
        return "\n".join(context_lines)
    
    async def _store_and_train(self, db, synthetic_examples: List[Dict]):
        """
        Store synthetic examples and trigger training
        """
        logger.info(f"💾 Storing {len(synthetic_examples)} synthetic examples")
        
        batch_id = f"synthetic_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        for example in synthetic_examples:
            content_storage.store_synthetic_training_data(
                db,
                example,
                batch_id=batch_id
            )
        
        self.rollback_history.append({
            "batch_id": batch_id,
            "count": len(synthetic_examples),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"✅ Synthetic data stored with batch_id: {batch_id}")
        logger.info(f"🧠 Training Pollen AI with synthetic examples...")
    
    async def rollback_batch(self, batch_id: str):
        """
        Rollback a synthetic data batch if quality issues detected
        """
        db = SessionLocal()
        try:
            content_storage.delete_synthetic_batch(db, batch_id)
            logger.info(f"↩️  Rolled back synthetic batch: {batch_id}")
            
            self.rollback_history = [
                h for h in self.rollback_history 
                if h["batch_id"] != batch_id
            ]
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
        finally:
            db.close()
    
    def get_stats(self) -> Dict:
        """
        Get statistics about synthetic data generation
        """
        return {
            "pending_intents": len([i for i in self.unknown_intents if i["status"] == "pending"]),
            "total_captured": len(self.unknown_intents),
            "rollback_history": len(self.rollback_history),
            "quality_threshold": self.quality_threshold
        }


synthetic_data_generator = SyntheticDataGenerator()
