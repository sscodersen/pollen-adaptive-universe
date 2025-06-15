
import asyncio
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
# === Refactor: import feedback persistence and adaptation signals ===
from .feedback_persistence import save_feedback_buffer, load_feedback_buffer
from .adaptation_signals import extract_adaptation_signals
import os

class LearningEngine:
    """
    Learning engine for Pollen LLMX
    
    Implements real-time learning, feedback processing, and model adaptation
    based on user interactions and feedback.
    """

    def __init__(self):
        self.learning_enabled = True
        self.auto_learning_enabled = True
        self.adaptation_rate = 0.001
        self.feedback_buffer = []
        self.learning_stats = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'last_adaptation': None
        }
        # Local feedback buffer persistence
        self.feedback_store_path = "./pollen_feedback_buffer.json"
        # Learning parameters
        self.min_confidence_threshold = 0.6
        self.adaptation_frequency = 10
        self.max_feedback_buffer = 1000
        # Load feedback buffer if exists:
        self.load_feedback_buffer()

    async def process_interaction(
        self,
        model: 'PollenLLMX',
        user_session: str,
        prompt: str,
        response: Dict[str, Any],
        mode: str
    ):
        """Process interaction for learning opportunities"""
        if not self.learning_enabled or not self.auto_learning_enabled:
            return
        try:
            confidence = response.get('confidence', 0.5)
            interaction_data = {
                'user_session': user_session,
                'prompt': prompt,
                'response': response,
                'mode': mode,
                'confidence': confidence,
                'timestamp': datetime.utcnow().isoformat()
            }
            # Add feedback & persist
            self.feedback_buffer.append(interaction_data)
            self.save_feedback_buffer()
            # Maintain buffer size
            if len(self.feedback_buffer) > self.max_feedback_buffer:
                self.feedback_buffer = self.feedback_buffer[-self.max_feedback_buffer:]
                self.save_feedback_buffer()
            # Check if adaptation is needed
            if len(self.feedback_buffer) % self.adaptation_frequency == 0:
                await self._adapt_model(model)
            # Learn from low confidence interactions
            if confidence < self.min_confidence_threshold:
                await self._process_low_confidence_interaction(model, interaction_data)
        except Exception as e:
            print(f"Learning engine error: {e}")
            self.learning_stats['failed_adaptations'] += 1

    async def _adapt_model(self, model: 'PollenLLMX'):
        """Adapt model based on recent interactions"""
        try:
            print("🧠 Starting model adaptation...")
            recent_feedback = self.feedback_buffer[-self.adaptation_frequency:]
            # Use refactored helper logic
            adaptation_signals = extract_adaptation_signals(recent_feedback, self.min_confidence_threshold)
            if adaptation_signals:
                await self._apply_adaptations(model, adaptation_signals)
                self.learning_stats['successful_adaptations'] += 1
                self.learning_stats['last_adaptation'] = datetime.utcnow().isoformat()
                print("✅ Model adaptation completed")
            self.learning_stats['total_adaptations'] += 1
        except Exception as e:
            print(f"Model adaptation failed: {e}")
            self.learning_stats['failed_adaptations'] += 1

    # _extract_adaptation_signals is now removed and replaced by helper import

    async def _apply_adaptations(self, model: 'PollenLLMX', signals: Dict[str, Any]):
        """Apply adaptations to the model"""
        for mode, performance in signals['mode_performance'].items():
            if performance['needs_improvement'] and mode in model.mode_adapters:
                await self._adapt_mode_layer(model, mode, performance)
        overall_confidence = np.mean([
            perf['avg_confidence'] 
            for perf in signals['mode_performance'].values()
        ])
        if overall_confidence < 0.5:
            model.learning_rate = min(0.01, model.learning_rate * 1.1)
        elif overall_confidence > 0.8:
            model.learning_rate = max(0.0001, model.learning_rate * 0.9)

    async def _adapt_mode_layer(self, model: 'PollenLLMX', mode: str, performance: Dict[str, Any]):
        """Adapt a specific mode layer"""
        try:
            mode_adapter = model.mode_adapters[mode]
            with torch.no_grad():
                for param in mode_adapter.parameters():
                    noise = torch.randn_like(param) * self.adaptation_rate
                    param.add_(noise)
            print(f"🔧 Adapted {mode} mode layer (confidence: {performance['avg_confidence']:.2f})")
        except Exception as e:
            print(f"Failed to adapt {mode} mode layer: {e}")

    async def _process_low_confidence_interaction(self, model: 'PollenLLMX', interaction_data: Dict[str, Any]):
        """Process interactions with low confidence for immediate learning"""
        try:
            mode = interaction_data['mode']
            confidence = interaction_data['confidence']
            if not hasattr(model, 'low_confidence_samples'):
                model.low_confidence_samples = []
            model.low_confidence_samples.append(interaction_data)
            if len(model.low_confidence_samples) > 100:
                model.low_confidence_samples = model.low_confidence_samples[-100:]
            print(f"⚠️ Low confidence interaction in {mode} mode: {confidence:.2f}")
        except Exception as e:
            print(f"Failed to process low confidence interaction: {e}")

    def process_user_feedback(
        self,
        user_session: str,
        interaction_id: str,
        feedback_type: str,
        feedback_data: Dict[str, Any]
    ):
        """Process explicit user feedback"""
        feedback_entry = {
            'user_session': user_session,
            'interaction_id': interaction_id,
            'feedback_type': feedback_type,
            'feedback_data': feedback_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.feedback_buffer.insert(0, feedback_entry)
        self.save_feedback_buffer()
        print(f"📝 User feedback received: {feedback_type} from {user_session[:8]}")

    def toggle_learning(self, user_session: str) -> bool:
        self.learning_enabled = not self.learning_enabled
        return self.learning_enabled

    def toggle_auto_learning(self) -> bool:
        self.auto_learning_enabled = not self.auto_learning_enabled
        print(f"Auto-learning {'enabled' if self.auto_learning_enabled else 'disabled'}.")
        return self.auto_learning_enabled

    def get_learning_stats(self) -> Dict[str, Any]:
        recent_feedback = self.feedback_buffer[-100:] if self.feedback_buffer else []
        recent_confidences = [
            fb.get('confidence', 0.5) 
            for fb in recent_feedback 
            if 'confidence' in fb
        ]
        avg_recent_confidence = (
            sum(recent_confidences) / len(recent_confidences) 
            if recent_confidences else 0.5
        )
        return {
            **self.learning_stats,
            'learning_enabled': self.learning_enabled,
            'feedback_buffer_size': len(self.feedback_buffer),
            'recent_avg_confidence': avg_recent_confidence,
            'adaptation_rate': self.adaptation_rate,
            'min_confidence_threshold': self.min_confidence_threshold
        }
    
    def clear_feedback_buffer(self):
        self.feedback_buffer = []
        print("🗑️ Feedback buffer cleared")
        self.save_feedback_buffer()
    
    def export_learning_data(self) -> Dict[str, Any]:
        return {
            'feedback_buffer': self.feedback_buffer,
            'learning_stats': self.learning_stats,
            'config': {
                'adaptation_rate': self.adaptation_rate,
                'min_confidence_threshold': self.min_confidence_threshold,
                'adaptation_frequency': self.adaptation_frequency
            }
        }

    def save_feedback_buffer(self):
        save_feedback_buffer(self.feedback_buffer, self.feedback_store_path)

    def load_feedback_buffer(self):
        self.feedback_buffer = load_feedback_buffer(self.feedback_store_path)
