"""
Memory Systems for Pollen AI
Implements Episodic, Long-term, and Contextual Memory
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class EpisodicMemory:
    """
    Episodic Memory stores short-term experiences and interactions.
    Implements a circular buffer with a fixed capacity.
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.logs: List[Dict[str, Any]] = []
    
    def add(self, experience: Dict[str, Any]) -> None:
        """Add an experience to episodic memory"""
        if len(self.logs) >= self.capacity:
            self.logs.pop(0)
        self.logs.append(experience)
    
    def recall(self) -> List[Dict[str, Any]]:
        """Recall all experiences from episodic memory"""
        return self.logs
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent experiences"""
        return self.logs[-n:] if len(self.logs) >= n else self.logs
    
    def clear(self) -> None:
        """Clear all episodic memories"""
        self.logs = []
    
    def size(self) -> int:
        """Return the current number of memories"""
        return len(self.logs)


class LongTermMemory:
    """
    Long-term Memory stores persistent knowledge and patterns.
    Backed by JSON file storage for persistence.
    """
    
    def __init__(self, path: str = "data/lt_memory.json"):
        self.path = path
        self._ensure_directory_exists()
        self.memory: Dict[str, Any] = self.load_memory()
    
    def _ensure_directory_exists(self) -> None:
        """Ensure the data directory exists"""
        directory = os.path.dirname(self.path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def load_memory(self) -> Dict[str, Any]:
        """Load memory from disk"""
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r') as f:
                    return json.load(f)
            return {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def update(self, key: str, value: Any) -> None:
        """Update a key-value pair in long-term memory"""
        self.memory[key] = value
        self.save_memory()
    
    def recall(self, key: str) -> Optional[Any]:
        """Recall a value from long-term memory"""
        return self.memory.get(key, None)
    
    def save_memory(self) -> None:
        """Save memory to disk"""
        try:
            with open(self.path, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Error saving long-term memory: {e}")
    
    def clear(self) -> None:
        """Clear all long-term memories"""
        self.memory = {}
        self.save_memory()
    
    def keys(self) -> List[str]:
        """Get all keys in long-term memory"""
        return list(self.memory.keys())


class ContextualMemory:
    """
    Contextual Memory stores embeddings and their associated text.
    Enables semantic search and context-aware retrieval.
    """
    
    def __init__(self):
        self.memory: Dict[Tuple, str] = {}
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []
    
    def add(self, embedding: np.ndarray, text: str) -> None:
        """Add an embedding-text pair to contextual memory"""
        # Store with tuple key for dict lookup
        embedding_tuple = tuple(embedding.flatten().tolist())
        self.memory[embedding_tuple] = text
        
        # Also store in lists for similarity search
        self.embeddings.append(embedding)
        self.texts.append(text)
    
    def retrieve(self, embedding: np.ndarray) -> Optional[str]:
        """Retrieve text by exact embedding match"""
        embedding_tuple = tuple(embedding.flatten().tolist())
        return self.memory.get(embedding_tuple, None)
    
    def find_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar texts based on embedding similarity.
        Returns list of (text, similarity_score) tuples.
        """
        if not self.embeddings:
            return []
        
        query_flat = query_embedding.flatten()
        similarities = []
        
        for i, stored_embedding in enumerate(self.embeddings):
            stored_flat = stored_embedding.flatten()
            # Cosine similarity
            similarity = np.dot(query_flat, stored_flat) / (
                np.linalg.norm(query_flat) * np.linalg.norm(stored_flat) + 1e-8
            )
            similarities.append((self.texts[i], float(similarity)))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def clear(self) -> None:
        """Clear all contextual memories"""
        self.memory = {}
        self.embeddings = []
        self.texts = []
    
    def size(self) -> int:
        """Return the number of stored embeddings"""
        return len(self.embeddings)
