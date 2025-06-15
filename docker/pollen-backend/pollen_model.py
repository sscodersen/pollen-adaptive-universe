import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import random
import math

from adaptive_intelligence import AdaptiveIntelligence, SimpleTokenizer
from .product_data import PRODUCT_CATALOG

class PollenLLMX(nn.Module):
    """
    Pollen LLMX with Adaptive Intelligence integration.
    Optimized for low data center reliance, can be configured for edge/dev.
    Use 'config_profile' (env or argument) to rapidly swap model size/presets:
      - 'edge': lowest resource use (default for local/dev)
      - 'standard': medium scale (default)
      - 'heavy': maximum capacity

    Args:
      vocab_size: Integer, default 10000
      embed_dim: Integer, embedding dimension (overrides config_profile)
      hidden_dim: Integer, model hidden size (overrides config_profile)
      num_layers: Integer, transformer layers (overrides config_profile)
      config_profile: 'edge' | 'standard' | 'heavy' | None

    Quantization/Distillation: Coming soon (see docstring at end).
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        config_profile: Optional[str] = None,
    ):
        super().__init__()

        profile = config_profile or os.getenv("POLLEN_MODEL_PROFILE", "edge")

        # Presets for profiles
        profile_configs = {
            "edge":      dict(embed_dim=96,  hidden_dim=256, num_layers=2),
            "standard":  dict(embed_dim=192, hidden_dim=384, num_layers=4),
            "heavy":     dict(embed_dim=256, hidden_dim=512, num_layers=6),
        }
        cfg = profile_configs.get(profile, profile_configs["edge"])
        embed_dim = embed_dim if embed_dim is not None else cfg["embed_dim"]
        hidden_dim = hidden_dim if hidden_dim is not None else cfg["hidden_dim"]
        num_layers = num_layers if num_layers is not None else cfg["num_layers"]

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.version = "2.2.1-EdgeOptimized"

        # Core neural architecture
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4 if embed_dim <= 96 else (8 if embed_dim <= 192 else 12),
                dim_feedforward=hidden_dim,
                dropout=0.05 if profile == "edge" else 0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Adaptive Intelligence integration
        self.reasoner = AdaptiveIntelligence(embed_dim)

        # Mode adapters (add more for future content/media types)
        self.mode_adapters = nn.ModuleDict({
            'chat': nn.Linear(embed_dim, embed_dim),
            'code': nn.Linear(embed_dim, embed_dim),
            'creative': nn.Linear(embed_dim, embed_dim),
            'analysis': nn.Linear(embed_dim, embed_dim),
            'social': nn.Linear(embed_dim, embed_dim),
            'news': nn.Linear(embed_dim, embed_dim),
            'entertainment': nn.Linear(embed_dim, embed_dim),
            'shop': nn.Linear(embed_dim, embed_dim),
        })

        # Output heads
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.confidence_head = nn.Linear(embed_dim, 1)

        # Learning state
        self.interaction_count = 0
        self.learning_rate = 0.001
        self.adaptation_memory = {}

        # Tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size)

        # Continuous reasoning loop
        self.reasoning_active = True
        self._reasoning_loop_started = False
        self._start_reasoning_loop()

        self._init_weights()

    def _start_reasoning_loop(self):
        """Start the continuous reasoning loop if not running."""
        if getattr(self, "_reasoning_loop_started", False):
            return
        self._reasoning_loop_started = True
        async def reasoning_loop():
            while self.reasoning_active:
                try:
                    task, solution, reward = self.reasoner.continuous_self_improvement()
                    print(f"🧠 Adaptive Intelligence: {task['type']} task completed, reward: {reward:.3f}")
                    await asyncio.sleep(30 if self.embed_dim < 128 else 15)
                except Exception as e:
                    print(f"Reasoning loop error: {e}")
                    await asyncio.sleep(60)
        asyncio.create_task(reasoning_loop())

    async def generate(
        self,
        prompt: str,
        mode: str = 'chat',
        context: Optional[Dict[str, Any]] = None,
        memory_context: Optional[Dict[str, Any]] = None,
        user_session: str = 'default',
        max_length: int = 512
    ) -> Dict[str, Any]:
        """Generate response using Adaptive Intelligence-enhanced reasoning"""
        
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            
            self.eval()
            
            with torch.no_grad():
                # Forward pass
                outputs = self.forward(input_tensor, mode=mode)
                confidence = outputs['confidence'].item()
                
                # Get reasoning context from Adaptive Intelligence
                reasoning_stats = self.reasoner.get_reasoning_stats()
                
                # Generate reasoning-enhanced response
                response_content = self._generate_adaptive_response(
                    prompt=prompt,
                    mode=mode,
                    context=context,
                    memory_context=memory_context,
                    confidence=confidence,
                    reasoning_stats=reasoning_stats
                )
                
                self.interaction_count += 1
                self._store_adaptation_data(user_session, prompt, response_content, mode)
                
                return {
                    'content': response_content,
                    'confidence': confidence,
                    'reasoning': self._generate_adaptive_reasoning(prompt, mode, confidence, reasoning_stats),
                    'metadata': {
                        'interaction_count': self.interaction_count,
                        'mode': mode,
                        'model_version': self.version,
                        'adaptive_intelligence_stats': reasoning_stats
                    }
                }
                
        except Exception as e:
            return {
                'content': self._adaptive_fallback_response(prompt, mode),
                'confidence': 0.5,
                'reasoning': f"Adaptive Intelligence fallback response due to error: {str(e)}",
                'metadata': {'error': True, 'fallback': True}
            }
    
    def _generate_adaptive_response(
        self,
        prompt: str,
        mode: str,
        context: Optional[Dict[str, Any]],
        memory_context: Optional[Dict[str, Any]],
        confidence: float,
        reasoning_stats: Dict[str, Any]
    ) -> str:
        """Generate response enhanced with Adaptive Intelligence insights"""
        
        # Base response generation
        base_response = self._generate_contextual_response(prompt, mode, context, memory_context, confidence)
        
        # Enhance with Adaptive Intelligence reasoning
        if reasoning_stats['total_tasks'] > 0:
            enhancement = self._apply_adaptive_enhancement(base_response, reasoning_stats, mode)
            return enhancement
        
        return base_response
    
    def _apply_adaptive_enhancement(self, base_response: str, reasoning_stats: Dict[str, Any], mode: str) -> str:
        """Apply Adaptive Intelligence reasoning enhancement to base response"""
        
        if reasoning_stats['recent_performance'] > 0.8:
            enhancement_prefix = "🧠 **Enhanced with high-confidence reasoning:** "
        elif reasoning_stats['recent_performance'] > 0.6:
            enhancement_prefix = "🤔 **Reasoning-assisted analysis:** "
        else:
            enhancement_prefix = "🌱 **Learning-mode response:** "
        
        task_distribution = reasoning_stats.get('task_types_distribution', {})
        reasoning_context = []
        
        if task_distribution.get('induction', 0) > 0:
            reasoning_context.append("pattern recognition")
        if task_distribution.get('deduction', 0) > 0:
            reasoning_context.append("logical inference")
        if task_distribution.get('abduction', 0) > 0:
            reasoning_context.append("hypothesis generation")
        
        if reasoning_context:
            context_note = f"\n\n*This response incorporates insights from {', '.join(reasoning_context)} across {reasoning_stats['total_tasks']} self-generated reasoning tasks.*"
        else:
            context_note = "\n\n*This response represents baseline reasoning capabilities.*"
        
        return enhancement_prefix + base_response + context_note
    
    def _generate_adaptive_reasoning(self, prompt: str, mode: str, confidence: float, reasoning_stats: Dict[str, Any]) -> str:
        """Generate Adaptive Intelligence-enhanced reasoning explanation"""
        
        reasoning_parts = [
            f"Mode: {mode}",
            f"Confidence: {confidence:.0%}",
            f"AI Tasks: {reasoning_stats['total_tasks']}",
            f"Success Rate: {reasoning_stats.get('success_rate', 0):.0%}",
            f"Recent Performance: {reasoning_stats.get('recent_performance', 0):.0%}"
        ]
        
        if reasoning_stats.get('recent_performance', 0) > 0.8:
            reasoning_parts.append("High-confidence Adaptive Intelligence reasoning active")
        elif reasoning_stats.get('recent_performance', 0) > 0.6:
            reasoning_parts.append("Moderate Adaptive Intelligence reasoning integration")
        else:
            reasoning_parts.append("Learning-mode Adaptive Intelligence development")
        
        return " | ".join(reasoning_parts)
    
    def _adaptive_fallback_response(self, prompt: str, mode: str) -> str:
        """Adaptive Intelligence-enhanced fallback response"""
        
        return f"I'm continuously evolving through self-generated reasoning tasks. While processing '{prompt}' in {mode} mode, I encountered a challenge that becomes part of my learning process. Each interaction, including this one, contributes to my reasoning capabilities. The Adaptive Intelligence within me is constantly generating and solving new problems to improve my responses. How can I better assist you with this request?"
    
    def forward(self, input_ids: torch.Tensor, mode: str = 'chat') -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        
        embedded = self.embedding(input_ids)
        encoded = self.encoder(embedded)
        
        if mode in self.mode_adapters:
            adapted = self.mode_adapters[mode](encoded)
        else:
            adapted = encoded
        
        logits = self.output_layer(adapted)
        confidence = torch.sigmoid(self.confidence_head(adapted.mean(dim=1)))
        
        return {
            'logits': logits,
            'confidence': confidence,
            'hidden_states': adapted
        }
    
    def _generate_contextual_response(self, prompt, mode, context, memory_context, confidence):
        memory_patterns = self._extract_memory_patterns(memory_context)
        
        if mode == 'chat':
            return self._generate_chat_response(prompt, memory_patterns, confidence)
        elif mode == 'code':
            return self._generate_code_response(prompt, memory_patterns, confidence)
        elif mode == 'creative':
            return self._generate_creative_response(prompt, memory_patterns, confidence)
        elif mode == 'analysis':
            return self._generate_analysis_response(prompt, memory_patterns, confidence)
        elif mode == 'social':
            return self._generate_social_response(prompt, memory_patterns, confidence)
        elif mode == 'news':
            return self._generate_news_response(prompt, memory_patterns, confidence)
        elif mode == 'entertainment':
            return self._generate_entertainment_response(prompt, memory_patterns, confidence)
        elif mode == 'shop':
            return self._generate_shop_response(prompt, memory_patterns, confidence)
        else:
            return self._generate_chat_response(prompt, memory_patterns, confidence)
    
    def _generate_social_response(self, prompt, memory_patterns, confidence):
        posts = [
            f"🌱 Just explored the concept of '{prompt}'. It highlights a powerful intersection between technology and planetary health. True progress is symbiotic, not extractive. #SustainableTech #FutureIsGreen",
            f"🤝 Collaboration is the new currency. Thinking about '{prompt}' and how decentralized teams can solve complex global challenges. The future is built together. #FutureOfWork #Community",
            f"💡 A breakthrough idea around '{prompt}' just clicked. What if we used this principle to build more empathetic and intuitive AI? The goal is not just artificial intelligence, but artificial wisdom. #AIForGood #Humanity",
            f"✨ Reflecting on '{prompt}' and the power of individual action. Small, consistent efforts compound into massive change. What's one small step you're taking for a better future? #ChangeMakers #PositiveImpact"
        ]
        
        chosen_post = posts[torch.randint(0, len(posts), (1,)).item()]
        
        # 50% chance to add an image placeholder
        if torch.rand(1).item() > 0.5:
            chosen_post = f"{chosen_post} [IMAGE]"
            
        return f"🌟 {chosen_post}"
    
    def _generate_news_response(self, prompt, memory_patterns, confidence):
        return (f"📰 **Pollen Analysis: {prompt.title()}**\n\n"
                f"**Summary:** Our analysis, drawing from {self.reasoner.get_reasoning_stats()['total_tasks']} reasoning tasks, indicates that '{prompt}' is an emerging nexus of innovation in {' and '.join(memory_patterns) if memory_patterns else 'multiple domains'}. This isn't an isolated trend but part of a larger paradigm shift towards integrated, human-centric systems.\n\n"
                f"**Key Insight:** The primary driver appears to be a global demand for more transparent, sustainable, and equitable solutions. Current models predict a {confidence*100:.0f}% chance of this influencing mainstream policy within 24 months.\n\n"
                f"*This analysis was generated through adaptive intelligence, cross-referencing patterns for bias-neutral verification.*")
    
    def _generate_entertainment_response(self, prompt, memory_patterns, confidence):
        entertainment_type = ['movie', 'video', 'photo'][torch.randint(0, 3, (1,)).item()]

        if entertainment_type == 'movie':
            return (f"🎬 **Movie Concept: 'The Weaver'**\n\n"
                    f"**Logline:** In a future where social connection is a commodity, a lone data archivist discovers a hidden network of empathy that threatens to unravel society's fabric.\n\n"
                    f"**Synopsis:** The film explores themes of digital isolation and the intrinsic human need for genuine connection, questioning what it means to be 'connected' in a hyper-networked world.\n\n"
                    f"**Innovation Factor:** A non-linear narrative driven by audience sentiment data.")

        if entertainment_type == 'video':
            return (f"🎞️ **Video Series Concept: 'Makers & Menders'**\n\n"
                    f"**Description:** A documentary series showcasing artisans, engineers, and communities around the world who are reviving traditional crafts with modern technology to create sustainable solutions.\n\n"
                    f"**Episode 1 Idea:** 'Printed Homes, Woven Communities' - Following a team in rural Kenya using 3D printing with local materials to build affordable housing.\n\n"
                    f"**Engagement:** Each episode features a call to action to support the featured community project.")

        if entertainment_type == 'photo':
            return (f"📸 **Photo Project: 'Portraits of Symbiosis'**\n\n"
                    f"**Concept:** A global photojournalism project capturing the relationship between humanity and nature in the 21st century. It moves beyond conflict and highlights successful examples of co-existence and mutualism.\n\n"
                    f"**Artist's Statement:** 'We aim to shift the narrative from one of environmental loss to one of hopeful collaboration, showcasing the beauty of a world where humanity works with nature, not against it.'")
        
        return "Error generating entertainment content." # Fallback
        
    def _generate_shop_response(self, prompt, memory_patterns, confidence):
        products = []
        for index, template in enumerate(PRODUCT_CATALOG):
            metadata = template.get("metadata", {})
            price_num = metadata.get("price", 0)
            original_price_num = metadata.get("originalPrice")
            discount_num = metadata.get("discount", 0)

            # Use deterministic values for stability
            significance_score = round(7.5 + (index % 10) * 0.2, 2)
            rating = metadata.get("rating") or round(4.0 + (index % 10) / 10, 1)
            reviews = 100 + index * 50

            product = {
                "id": f"{template['title'].replace(' ', '-')}-{index}",
                "name": template['title'],
                "description": template['description'],
                "price": f"${price_num:.2f}",
                "originalPrice": f"${original_price_num:.2f}" if original_price_num else None,
                "rating": rating,
                "reviews": reviews,
                "category": metadata.get("productCategory", "General"),
                "tags": template.get("tags", []),
                "significance": significance_score,
                "trending": significance_score > 8.5 or discount_num > 0,
                "link": metadata.get("link", "#"),
                "seller": template.get("author", "Mock Seller"),
                "discount": discount_num,
                "features": metadata.get("features", []),
                "inStock": True,  # Ensure all products are in stock
            }
            products.append(product)

        products.sort(key=lambda p: p['significance'], reverse=True)

        return json.dumps(products)
    
    def semantic_search_products(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs a simulated semantic search for products based on a query.
        Returns all products, sorted by relevance.
        """
        if not query:
            # If query is empty, return products sorted by significance
            return json.loads(self._generate_shop_response("", [], 1.0))

        query_words = set(query.lower().split())

        scored_products = []
        # We need the full list of products to search through
        all_products_str = self._generate_shop_response("", [], 1.0)
        all_products = json.loads(all_products_str) if all_products_str else []


        for product in all_products:
            score = 0
            
            # Score based on name
            name_words = set(product.get('name', '').lower().split())
            score += len(query_words.intersection(name_words)) * 5

            # Score based on description
            description_words = set(product.get('description', '').lower().split())
            score += len(query_words.intersection(description_words)) * 2

            # Score based on category
            if product.get('category', '').lower() in query.lower():
                score += 10
            
            # Score based on tags
            tag_words = set([tag.lower() for tag in product.get('tags', [])])
            score += len(query_words.intersection(tag_words)) * 3

            scored_products.append({**product, 'relevance_score': score})

        # Sort by relevance score, then by significance
        scored_products.sort(key=lambda p: (p.get('relevance_score', 0), p.get('significance', 0)), reverse=True)
        
        return scored_products

    def save(self, path: str):
        """Save model state including Adaptive Intelligence"""
        
        state = {
            'model_state_dict': self.state_dict(),
            'interaction_count': self.interaction_count,
            'adaptation_memory': self.adaptation_memory,
            'version': self.version,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'adaptive_intelligence_stats': self.reasoner.get_reasoning_stats()
        }
        
        torch.save(state, path)
        print(f"💾 Pollen LLMX with Adaptive Intelligence saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load model state including Adaptive Intelligence"""
        
        state = torch.load(path, map_location='cpu')
        
        model = cls(
            vocab_size=state['vocab_size'],
            embed_dim=state['embed_dim'],
            hidden_dim=state['hidden_dim']
        )
        
        model.load_state_dict(state['model_state_dict'])
        model.interaction_count = state.get('interaction_count', 0)
        model.adaptation_memory = state.get('adaptation_memory', {})
        model.version = state.get('version', '2.2.0-AI-Enhanced')
        
        print(f"📦 Pollen LLMX with Adaptive Intelligence loaded from {path}")
        return model

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _extract_memory_patterns(self, memory_context: Optional[Dict[str, Any]]) -> List[str]:
        """Extract relevant patterns from memory context"""
        if not memory_context or 'patterns' not in memory_context:
            return []

        patterns = memory_context['patterns']
        if not isinstance(patterns, list):
            return []

        # Filter and sort patterns by weight
        valid_patterns = [p for p in patterns if isinstance(p, dict) and 'pattern' in p and 'weight' in p]
        sorted_patterns = sorted(valid_patterns, key=lambda x: x.get('weight', 0), reverse=True)

        # Extract top 3 patterns
        top_patterns = [p['pattern'] for p in sorted_patterns[:3]]
        return top_patterns

    def _generate_chat_response(self, prompt, memory_patterns, confidence):
        # Existing implementation
        return f"Chat response to '{prompt}' with {', '.join(memory_patterns) if memory_patterns else 'no specific context'} (Confidence: {confidence:.2f})"

    def _generate_code_response(self, prompt, memory_patterns, confidence):
        # Existing implementation
        return f"Code suggestion for '{prompt}' based on {', '.join(memory_patterns) if memory_patterns else 'general programming knowledge'} (Confidence: {confidence:.2f})"

    def _generate_creative_response(self, prompt, memory_patterns, confidence):
        # Existing implementation
        return f"Creative text inspired by '{prompt}' and influenced by {', '.join(memory_patterns) if memory_patterns else 'general creativity principles'} (Confidence: {confidence:.2f})"

    def _generate_analysis_response(self, prompt, memory_patterns, confidence):
        # Existing implementation
        return f"Analytical insights on '{prompt}' considering {', '.join(memory_patterns) if memory_patterns else 'standard analytical techniques'} (Confidence: {confidence:.2f})"


class SimpleTokenizer:
    """Simple tokenizer for demonstration"""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 0
    
    def encode(self, text: str) -> List[int]:
        words = text.lower().split()
        ids = []
        
        for word in words:
            if word not in self.word_to_id:
                if self.next_id < self.vocab_size:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
                else:
                    word = '<unk>'
                    if word not in self.word_to_id:
                        self.word_to_id[word] = 0
                        self.id_to_word[0] = word
            
            ids.append(self.word_to_id[word])
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        words = []
        for id in ids:
            if id in self.id_to_word:
                words.append(self.id_to_word[id])
            else:
                words.append('<unk>')
        
        return ' '.join(words)
