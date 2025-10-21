"""
Memory modules for Pollen AI - Absolute Zero Reasoner
Lightweight implementation with optional heavy dependencies
"""

import json
import os
import time
import numpy as np
from typing import List, Tuple, Any, Dict, Optional


class EpisodicMemory:
    """Short-term memory for recent experiences"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.logs: List[Dict] = []
    
    def add(self, experience: Dict):
        """Add experience to episodic memory"""
        if len(self.logs) >= self.capacity:
            self.logs.pop(0)
        self.logs.append(experience)
    
    def recall(self) -> List[Dict]:
        """Recall all episodic memories"""
        return self.logs
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get n most recent experiences"""
        return self.logs[-n:] if self.logs else []
    
    def size(self) -> int:
        """Get size of episodic memory"""
        return len(self.logs)
    
    def clear(self):
        """Clear episodic memory"""
        self.logs = []


class LongTermMemory:
    """Persistent memory storage"""
    
    def __init__(self, path: str = "data/lt_memory.json"):
        self.path = path
        self.memory: Dict[str, Any] = self.load_memory()
    
    def load_memory(self) -> Dict[str, Any]:
        """Load memory from file"""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            if os.path.exists(self.path):
                with open(self.path, 'r') as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return {}
    
    def update(self, key: str, value: Any):
        """Update memory with key-value pair"""
        self.memory[key] = value
        self.save_memory()
    
    def recall(self, key: str) -> Optional[Any]:
        """Recall value from memory"""
        return self.memory.get(key, None)
    
    def save_memory(self):
        """Save memory to file"""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save memory: {e}")
    
    def keys(self) -> List[str]:
        """Get all memory keys"""
        return list(self.memory.keys())
    
    def size(self) -> int:
        """Get number of stored memories"""
        return len(self.memory)


class ContextualMemory:
    """Semantic memory with embedding-based search"""
    
    def __init__(self):
        self.memory: Dict[Tuple, str] = {}
        self.embeddings: List[Tuple[np.ndarray, str]] = []
    
    def add(self, embedding: np.ndarray, text: str):
        """Add text with its embedding"""
        # Store both ways for compatibility
        self.memory[tuple(embedding.tolist())] = text
        self.embeddings.append((embedding, text))
    
    def retrieve(self, embedding: np.ndarray) -> Optional[str]:
        """Retrieve exact match by embedding"""
        return self.memory.get(tuple(embedding.tolist()), None)
    
    def find_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar embeddings using cosine similarity"""
        if not self.embeddings:
            return []
        
        similarities = []
        for stored_embedding, text in self.embeddings:
            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            similarities.append((text, float(similarity)))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def size(self) -> int:
        """Get number of stored embeddings"""
        return len(self.embeddings)
    
    def clear(self):
        """Clear contextual memory"""
        self.memory.clear()
        self.embeddings.clear()
