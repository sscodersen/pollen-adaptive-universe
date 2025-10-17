"""
Memory Persistence Layer for Pollen AI
Manages storage and retrieval of AI memory systems
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from pathlib import Path

class MemoryPersistence:
    """
    Handles persistence of Pollen AI memory systems.
    Currently uses JSON files, will migrate to PostgreSQL database.
    """
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Memory file paths
        self.episodic_path = self.base_path / "episodic_memory.json"
        self.longterm_path = self.base_path / "lt_memory.json"
        self.contextual_path = self.base_path / "contextual_memory.json"
        
        # In-memory cache for performance
        self.cache = {
            'episodic': [],
            'longterm': {},
            'contextual': []
        }
        
        # Load existing memories
        self.load_all_memories()
    
    def load_all_memories(self):
        """Load all memory types from disk"""
        try:
            # Load episodic memory
            if self.episodic_path.exists():
                with open(self.episodic_path, 'r') as f:
                    self.cache['episodic'] = json.load(f)
            
            # Load long-term memory
            if self.longterm_path.exists():
                with open(self.longterm_path, 'r') as f:
                    self.cache['longterm'] = json.load(f)
            
            # Load contextual memory
            if self.contextual_path.exists():
                with open(self.contextual_path, 'r') as f:
                    self.cache['contextual'] = json.load(f)
            
            print(f"✅ Loaded memories: {len(self.cache['episodic'])} episodic, "
                  f"{len(self.cache['longterm'])} long-term, "
                  f"{len(self.cache['contextual'])} contextual")
        
        except Exception as e:
            print(f"⚠️ Error loading memories: {e}")
    
    def save_episodic_memory(self, episode: Dict[str, Any]):
        """Save an episodic memory"""
        self.cache['episodic'].append({
            **episode,
            'timestamp': datetime.now().isoformat(),
            'id': f"ep_{int(time.time() * 1000)}"
        })
        
        # Keep only last 1000 episodes
        if len(self.cache['episodic']) > 1000:
            self.cache['episodic'] = self.cache['episodic'][-1000:]
        
        self._save_to_file(self.episodic_path, self.cache['episodic'])
    
    def save_longterm_memory(self, key: str, value: Any):
        """Save to long-term memory"""
        self.cache['longterm'][key] = {
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'access_count': self.cache['longterm'].get(key, {}).get('access_count', 0) + 1
        }
        
        self._save_to_file(self.longterm_path, self.cache['longterm'])
    
    def save_contextual_memory(self, context: Dict[str, Any]):
        """Save contextual memory with embedding"""
        self.cache['contextual'].append({
            **context,
            'timestamp': datetime.now().isoformat(),
            'id': f"ctx_{int(time.time() * 1000)}"
        })
        
        # Keep only last 5000 contexts
        if len(self.cache['contextual']) > 5000:
            self.cache['contextual'] = self.cache['contextual'][-5000:]
        
        self._save_to_file(self.contextual_path, self.cache['contextual'])
    
    def get_episodic_memories(self, limit: int = 100) -> List[Dict]:
        """Retrieve recent episodic memories"""
        return self.cache['episodic'][-limit:]
    
    def get_longterm_memory(self, key: str) -> Optional[Any]:
        """Retrieve from long-term memory"""
        memory = self.cache['longterm'].get(key)
        if memory:
            # Update access count
            memory['access_count'] = memory.get('access_count', 0) + 1
            memory['last_accessed'] = datetime.now().isoformat()
            self._save_to_file(self.longterm_path, self.cache['longterm'])
            return memory['value']
        return None
    
    def search_contextual_memory(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search contextual memory by embedding similarity"""
        import numpy as np
        
        results = []
        query_vec = np.array(query_embedding)
        
        for ctx in self.cache['contextual']:
            if 'embedding' in ctx:
                ctx_vec = np.array(ctx['embedding'])
                similarity = np.dot(query_vec, ctx_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(ctx_vec)
                )
                results.append((ctx, similarity))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:top_k]]
    
    def _save_to_file(self, path: Path, data: Any):
        """Save data to JSON file"""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving to {path}: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            'episodic_count': len(self.cache['episodic']),
            'longterm_count': len(self.cache['longterm']),
            'contextual_count': len(self.cache['contextual']),
            'total_memories': (
                len(self.cache['episodic']) + 
                len(self.cache['longterm']) + 
                len(self.cache['contextual'])
            ),
            'storage_path': str(self.base_path.absolute())
        }
    
    async def consolidate_memories(self):
        """
        Consolidate episodic memories into long-term storage.
        This simulates the memory consolidation process.
        """
        # Get recent episodic memories
        recent_episodes = self.cache['episodic'][-100:]
        
        # Extract patterns and store in long-term memory
        patterns = {}
        for episode in recent_episodes:
            if 'input' in episode and 'output' in episode:
                key = f"pattern_{hash(episode['input']) % 10000}"
                if key not in patterns:
                    patterns[key] = []
                patterns[key].append(episode)
        
        # Save consolidated patterns
        for key, episodes in patterns.items():
            if len(episodes) > 2:  # Only consolidate if pattern appears multiple times
                self.save_longterm_memory(key, {
                    'pattern': episodes[0]['input'][:100],
                    'frequency': len(episodes),
                    'examples': episodes[:5]
                })
        
        print(f"✅ Consolidated {len(patterns)} patterns into long-term memory")
    
    def cleanup_old_memories(self, days: int = 30):
        """Remove memories older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        # Clean episodic
        self.cache['episodic'] = [
            e for e in self.cache['episodic']
            if datetime.fromisoformat(e.get('timestamp', '2020-01-01')).timestamp() > cutoff
        ]
        
        # Clean contextual
        self.cache['contextual'] = [
            c for c in self.cache['contextual']
            if datetime.fromisoformat(c.get('timestamp', '2020-01-01')).timestamp() > cutoff
        ]
        
        # Save cleaned data
        self._save_to_file(self.episodic_path, self.cache['episodic'])
        self._save_to_file(self.contextual_path, self.cache['contextual'])
        
        print(f"✅ Cleaned up memories older than {days} days")


# Global memory persistence instance
memory_persistence = MemoryPersistence()
