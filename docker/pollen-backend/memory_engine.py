
import json
import time
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional

from .memory_persistence import save_user_memory, load_user_memory, save_global_stats, load_global_stats
from .keyword_extraction import extract_keywords
from .pattern_extraction import update_patterns, decay_patterns

class MemoryEngine:
    """
    Memory engine for Pollen LLMX

    Handles short-term and long-term memory storage, pattern extraction,
    and context retrieval for personalized AI interactions.
    """

    def __init__(self):
        self.user_memories = defaultdict(lambda: {
            'short_term': deque(maxlen=100),
            'long_term': {},
            'preferences': {},
            'session_data': {},
            'learning_enabled': True
        })
        self.global_patterns = defaultdict(float)
        self.interaction_counts = defaultdict(int)
        self.last_save = time.time()
        self.save_interval = 300  # Save every 5 minutes
        # Paths for local persistence
        self.user_memory_path = "./pollen_user_memories.json"
        self.global_stats_path = "./pollen_global_stats.json"
        self._load_persistence()

    def add_interaction(
        self,
        user_session: str,
        input_text: str,
        output_text: str,
        mode: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
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
        self.user_memories[user_session]['short_term'].append(interaction)
        self._extract_patterns(user_session, input_text, output_text, mode)
        self.interaction_counts[user_session] += 1
        if time.time() - self.last_save > self.save_interval:
            self._save_memory_state()

    def _extract_patterns(self, user_session: str, input_text: str, output_text: str, mode: str):
        user_memory = self.user_memories[user_session]
        input_words = extract_keywords(input_text)
        update_patterns(user_memory, input_words, input_text, output_text, mode)
        for word in input_words:
            self.global_patterns[f"{mode}:{word}"] += 1
        decay_patterns(user_memory)

    def get_relevant_context(
        self,
        user_session: str,
        current_input: str,
        mode: str,
        max_short_term: int = 5,
        max_long_term: int = 10
    ) -> Dict[str, Any]:
        user_memory = self.user_memories[user_session]
        recent_memories = list(user_memory['short_term'])[-max_short_term:]
        input_keywords = extract_keywords(current_input)
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
        user_memory = self.user_memories[user_session]
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
        self.user_memories[user_session] = {
            'short_term': deque(maxlen=100),
            'long_term': {},
            'preferences': {},
            'session_data': {},
            'learning_enabled': True
        }
        self.interaction_counts[user_session] = 0

    def toggle_learning(self, user_session: str) -> bool:
        current_state = self.user_memories[user_session]['learning_enabled']
        self.user_memories[user_session]['learning_enabled'] = not current_state
        return not current_state

    def update_user_preferences(self, user_session: str, preferences: Dict[str, Any]):
        self.user_memories[user_session]['preferences'].update(preferences)

    def _save_memory_state(self):
        save_user_memory(self.user_memories, self.user_memory_path)
        save_global_stats(dict(self.global_patterns), dict(self.interaction_counts), self.global_stats_path)
        self.last_save = time.time()
        print(f"💾 Memory state saved at {datetime.utcnow().isoformat()}")

    def _load_persistence(self):
        user_memories_loaded = load_user_memory(self.user_memory_path)
        if user_memories_loaded:
            for user, data in user_memories_loaded.items():
                # Convert short_term to deque if needed
                data['short_term'] = deque(data.get('short_term', []), maxlen=100)
                self.user_memories[user] = data
        global_patterns, interaction_counts = load_global_stats(self.global_stats_path)
        if global_patterns:
            self.global_patterns.update(global_patterns)
        if interaction_counts:
            self.interaction_counts.update(interaction_counts)

    def get_global_stats(self) -> Dict[str, Any]:
        total_users = len(self.user_memories)
        total_interactions = sum(self.interaction_counts.values())
        total_patterns = sum(len(mem['long_term']) for mem in self.user_memories.values())
        top_global_patterns = sorted(
            self.global_patterns.items(), key=lambda x: x[1], reverse=True
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
