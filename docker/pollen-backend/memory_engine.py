
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

class MemoryEngine:
    """
    Memory engine for Pollen LLMX
    
    Handles short-term and long-term memory storage, pattern extraction,
    and context retrieval for personalized AI interactions.
    """
    
    def __init__(self):
        # User-specific memory stores
        self.user_memories = defaultdict(lambda: {
            'short_term': deque(maxlen=100),  # Recent interactions
            'long_term': {},  # Learned patterns and preferences
            'preferences': {},  # User preferences
            'session_data': {},  # Session-specific data
            'learning_enabled': True
        })
        
        # Global pattern storage
        self.global_patterns = defaultdict(float)
        self.interaction_counts = defaultdict(int)
        
        # Memory persistence
        self.last_save = time.time()
        self.save_interval = 300  # Save every 5 minutes
    
    def add_interaction(
        self,
        user_session: str,
        input_text: str,
        output_text: str,
        mode: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a new interaction to user memory"""
        
        if not self.user_memories[user_session]['learning_enabled']:
            return
        
        interaction = {
            'id': str(uuid.uuid4()),
            'input': input_text,
            'output': output_text,
            'mode': mode,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        # Add to short-term memory
        self.user_memories[user_session]['short_term'].append(interaction)
        
        # Extract patterns for long-term memory
        self._extract_patterns(user_session, input_text, output_text, mode)
        
        # Update global statistics
        self.interaction_counts[user_session] += 1
        
        # Auto-save if needed
        if time.time() - self.last_save > self.save_interval:
            self._save_memory_state()
    
    def _extract_patterns(self, user_session: str, input_text: str, output_text: str, mode: str):
        """Extract patterns from interaction for long-term memory"""
        
        user_memory = self.user_memories[user_session]
        
        # Extract keywords and phrases
        input_words = self._extract_keywords(input_text)
        
        # Update pattern weights
        for word in input_words:
            pattern_key = f"{mode}:{word}"
            
            if pattern_key not in user_memory['long_term']:
                user_memory['long_term'][pattern_key] = {
                    'weight': 1.0,
                    'count': 1,
                    'last_seen': datetime.utcnow().isoformat(),
                    'mode': mode,
                    'examples': []
                }
            else:
                user_memory['long_term'][pattern_key]['weight'] += 0.1
                user_memory['long_term'][pattern_key]['count'] += 1
                user_memory['long_term'][pattern_key]['last_seen'] = datetime.utcnow().isoformat()
            
            # Store example interactions
            if len(user_memory['long_term'][pattern_key]['examples']) < 3:
                user_memory['long_term'][pattern_key]['examples'].append({
                    'input': input_text[:100],  # Truncate for storage
                    'output': output_text[:100],
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        # Update global patterns
        for word in input_words:
            self.global_patterns[f"{mode}:{word}"] += 1
        
        # Decay old patterns
        self._decay_patterns(user_session)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        
        # Simple keyword extraction (would use NLP in production)
        words = text.lower().split()
        
        # Filter out common words and keep meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
        
        keywords = []
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            
            # Keep meaningful words
            if len(word) > 2 and word not in stop_words:
                keywords.append(word)
        
        return keywords[:10]  # Return top 10 keywords
    
    def _decay_patterns(self, user_session: str):
        """Decay old patterns to prevent memory bloat"""
        
        user_memory = self.user_memories[user_session]
        current_time = datetime.utcnow()
        
        patterns_to_remove = []
        
        for pattern_key, pattern_data in user_memory['long_term'].items():
            last_seen = datetime.fromisoformat(pattern_data['last_seen'])
            days_since_seen = (current_time - last_seen).days
            
            # Decay weight based on time
            if days_since_seen > 7:
                pattern_data['weight'] *= 0.9
            
            # Remove very weak patterns
            if pattern_data['weight'] < 0.1:
                patterns_to_remove.append(pattern_key)
        
        # Remove weak patterns
        for pattern_key in patterns_to_remove:
            del user_memory['long_term'][pattern_key]
    
    def get_relevant_context(
        self,
        user_session: str,
        current_input: str,
        mode: str,
        max_short_term: int = 5,
        max_long_term: int = 10
    ) -> Dict[str, Any]:
        """Get relevant context for current interaction"""
        
        user_memory = self.user_memories[user_session]
        
        # Get recent short-term memories
        recent_memories = list(user_memory['short_term'])[-max_short_term:]
        
        # Get relevant long-term patterns
        input_keywords = self._extract_keywords(current_input)
        relevant_patterns = []
        
        for keyword in input_keywords:
            pattern_key = f"{mode}:{keyword}"
            if pattern_key in user_memory['long_term']:
                pattern_data = user_memory['long_term'][pattern_key]
                relevant_patterns.append({
                    'pattern': keyword,
                    'weight': pattern_data['weight'],
                    'count': pattern_data['count'],
                    'category': mode,
                    'examples': pattern_data['examples']
                })
        
        # Sort by weight and get top patterns
        relevant_patterns.sort(key=lambda x: x['weight'], reverse=True)
        relevant_patterns = relevant_patterns[:max_long_term]
        
        return {
            'recent': [
                {
                    'input': mem['input'],
                    'output': mem['output'],
                    'timestamp': mem['timestamp']
                }
                for mem in recent_memories
            ],
            'relevant': relevant_patterns,
            'preferences': user_memory['preferences']
        }
    
    def get_user_stats(self, user_session: str) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        
        user_memory = self.user_memories[user_session]
        
        # Get top patterns
        top_patterns = []
        for pattern_key, pattern_data in user_memory['long_term'].items():
            if ':' in pattern_key:
                category, pattern = pattern_key.split(':', 1)
                top_patterns.append({
                    'pattern': pattern,
                    'weight': pattern_data['weight'],
                    'category': category,
                    'count': pattern_data['count']
                })
        
        top_patterns.sort(key=lambda x: x['weight'], reverse=True)
        
        return {
            'short_term_size': len(user_memory['short_term']),
            'long_term_patterns': len(user_memory['long_term']),
            'top_patterns': top_patterns[:10],
            'is_learning': user_memory['learning_enabled'],
            'total_interactions': self.interaction_counts[user_session],
            'session_start': user_memory.get('session_start', datetime.utcnow().isoformat())
        }
    
    def clear_user_memory(self, user_session: str):
        """Clear all memory for a user"""
        
        self.user_memories[user_session] = {
            'short_term': deque(maxlen=100),
            'long_term': {},
            'preferences': {},
            'session_data': {},
            'learning_enabled': True
        }
        
        self.interaction_counts[user_session] = 0
    
    def toggle_learning(self, user_session: str) -> bool:
        """Toggle learning for a user"""
        
        current_state = self.user_memories[user_session]['learning_enabled']
        self.user_memories[user_session]['learning_enabled'] = not current_state
        
        return not current_state
    
    def update_user_preferences(self, user_session: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        
        self.user_memories[user_session]['preferences'].update(preferences)
    
    def _save_memory_state(self):
        """Save memory state to persistent storage"""
        
        # In production, this would save to a database
        # For now, we just update the timestamp
        self.last_save = time.time()
        print(f"💾 Memory state saved at {datetime.utcnow().isoformat()}")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global memory statistics"""
        
        total_users = len(self.user_memories)
        total_interactions = sum(self.interaction_counts.values())
        total_patterns = sum(len(mem['long_term']) for mem in self.user_memories.values())
        
        # Top global patterns
        top_global_patterns = sorted(
            self.global_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        return {
            'total_users': total_users,
            'total_interactions': total_interactions,
            'total_patterns': total_patterns,
            'top_global_patterns': [
                {'pattern': pattern.split(':', 1)[1] if ':' in pattern else pattern,
                 'category': pattern.split(':', 1)[0] if ':' in pattern else 'unknown',
                 'count': count}
                for pattern, count in top_global_patterns
            ],
            'last_save': self.last_save
        }
