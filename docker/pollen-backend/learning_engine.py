
import asyncio
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class LearningEngine:
    """
    Learning engine for Pollen LLMX
    
    Implements real-time learning, feedback processing, and model adaptation
    based on user interactions and feedback.
    """
    
    def __init__(self):
        self.learning_enabled = True
        self.adaptation_rate = 0.001
        self.feedback_buffer = []
        self.learning_stats = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'last_adaptation': None
        }
        
        # Learning parameters
        self.min_confidence_threshold = 0.6
        self.adaptation_frequency = 10  # Adapt every N interactions
        self.max_feedback_buffer = 1000
    
    async def process_interaction(
        self,
        model: 'PollenLLMX',
        user_session: str,
        prompt: str,
        response: Dict[str, Any],
        mode: str
    ):
        """Process interaction for learning opportunities"""
        
        if not self.learning_enabled:
            return
        
        try:
            # Extract learning signals
            confidence = response.get('confidence', 0.5)
            interaction_data = {
                'user_session': user_session,
                'prompt': prompt,
                'response': response,
                'mode': mode,
                'confidence': confidence,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add to feedback buffer
            self.feedback_buffer.append(interaction_data)
            
            # Maintain buffer size
            if len(self.feedback_buffer) > self.max_feedback_buffer:
                self.feedback_buffer = self.feedback_buffer[-self.max_feedback_buffer:]
            
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
            
            # Analyze recent feedback
            recent_feedback = self.feedback_buffer[-self.adaptation_frequency:]
            adaptation_signals = self._extract_adaptation_signals(recent_feedback)
            
            # Apply adaptations
            if adaptation_signals:
                await self._apply_adaptations(model, adaptation_signals)
                self.learning_stats['successful_adaptations'] += 1
                self.learning_stats['last_adaptation'] = datetime.utcnow().isoformat()
                print("✅ Model adaptation completed")
            
            self.learning_stats['total_adaptations'] += 1
            
        except Exception as e:
            print(f"Model adaptation failed: {e}")
            self.learning_stats['failed_adaptations'] += 1
    
    def _extract_adaptation_signals(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract signals for model adaptation"""
        
        signals = {
            'mode_performance': {},
            'confidence_patterns': {},
            'user_patterns': {},
            'improvement_areas': []
        }
        
        # Analyze performance by mode
        mode_confidences = {}
        for interaction in feedback_data:
            mode = interaction['mode']
            confidence = interaction['confidence']
            
            if mode not in mode_confidences:
                mode_confidences[mode] = []
            mode_confidences[mode].append(confidence)
        
        # Calculate average confidence per mode
        for mode, confidences in mode_confidences.items():
            avg_confidence = sum(confidences) / len(confidences)
            signals['mode_performance'][mode] = {
                'avg_confidence': avg_confidence,
                'sample_count': len(confidences),
                'needs_improvement': avg_confidence < self.min_confidence_threshold
            }
            
            if avg_confidence < self.min_confidence_threshold:
                signals['improvement_areas'].append(mode)
        
        # Analyze user interaction patterns
        user_interactions = {}
        for interaction in feedback_data:
            user = interaction['user_session']
            if user not in user_interactions:
                user_interactions[user] = []
            user_interactions[user].append(interaction)
        
        signals['user_patterns'] = {
            'unique_users': len(user_interactions),
            'avg_interactions_per_user': len(feedback_data) / len(user_interactions) if user_interactions else 0,
            'most_active_users': sorted(
                user_interactions.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5]
        }
        
        return signals
    
    async def _apply_adaptations(self, model: 'PollenLLMX', signals: Dict[str, Any]):
        """Apply adaptations to the model"""
        
        # Adapt mode-specific layers for poor performing modes
        for mode, performance in signals['mode_performance'].items():
            if performance['needs_improvement'] and mode in model.mode_adapters:
                await self._adapt_mode_layer(model, mode, performance)
        
        # Update learning rate based on overall performance
        overall_confidence = np.mean([
            perf['avg_confidence'] 
            for perf in signals['mode_performance'].values()
        ])
        
        if overall_confidence < 0.5:
            # Increase learning rate for poor performance
            model.learning_rate = min(0.01, model.learning_rate * 1.1)
        elif overall_confidence > 0.8:
            # Decrease learning rate for good performance
            model.learning_rate = max(0.0001, model.learning_rate * 0.9)
    
    async def _adapt_mode_layer(self, model: 'PollenLLMX', mode: str, performance: Dict[str, Any]):
        """Adapt a specific mode layer"""
        
        try:
            mode_adapter = model.mode_adapters[mode]
            
            # Simple adaptation: add small random noise to encourage exploration
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
            
            # Store for analysis
            if not hasattr(model, 'low_confidence_samples'):
                model.low_confidence_samples = []
            
            model.low_confidence_samples.append(interaction_data)
            
            # Keep only recent samples
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
            'feedback_type': feedback_type,  # 'positive', 'negative', 'correction'
            'feedback_data': feedback_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add to feedback buffer with high priority
        self.feedback_buffer.insert(0, feedback_entry)
        
        print(f"📝 User feedback received: {feedback_type} from {user_session[:8]}")
    
    def toggle_learning(self, user_session: str) -> bool:
        """Toggle learning for specific user or globally"""
        
        self.learning_enabled = not self.learning_enabled
        return self.learning_enabled
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning engine statistics"""
        
        recent_feedback = self.feedback_buffer[-100:] if self.feedback_buffer else []
        
        # Calculate recent performance
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
        """Clear the feedback buffer"""
        
        self.feedback_buffer = []
        print("🗑️ Feedback buffer cleared")
    
    def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for analysis"""
        
        return {
            'feedback_buffer': self.feedback_buffer,
            'learning_stats': self.learning_stats,
            'config': {
                'adaptation_rate': self.adaptation_rate,
                'min_confidence_threshold': self.min_confidence_threshold,
                'adaptation_frequency': self.adaptation_frequency
            }
        }
