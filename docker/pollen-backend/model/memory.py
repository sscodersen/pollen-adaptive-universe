
import json
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class MemoryManager:
    """Manages episodic, long-term, and contextual memory for Pollen AI"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.lt_memory_file = os.path.join(data_dir, "lt_memory.json")
        self.logs_dir = os.path.join(data_dir, "logs")
        
        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Memory structures
        self.episodic_memory = {}  # User session -> recent interactions
        self.long_term_memory = self.load_long_term_memory()
        self.contextual_memory = {}  # Current context per user
        
    def load_long_term_memory(self) -> Dict[str, Any]:
        """Load long-term memory from file"""
        try:
            if os.path.exists(self.lt_memory_file):
                with open(self.lt_memory_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading long-term memory: {e}")
        
        return {
            "patterns": [],
            "successful_strategies": [],
            "user_preferences": {},
            "reasoning_insights": []
        }
    
    def save_long_term_memory(self):
        """Save long-term memory to file"""
        try:
            with open(self.lt_memory_file, 'w') as f:
                json.dump(self.long_term_memory, f, indent=2)
        except Exception as e:
            print(f"Error saving long-term memory: {e}")
    
    def add_interaction(self, user_session: str, input_text: str, output_text: str, mode: str, metadata: Dict[str, Any]):
        """Add new interaction to memory"""
        
        # Add to episodic memory
        if user_session not in self.episodic_memory:
            self.episodic_memory[user_session] = []
        
        interaction = {
            "timestamp": time.time(),
            "input": input_text,
            "output": output_text,
            "mode": mode,
            "metadata": metadata
        }
        
        self.episodic_memory[user_session].append(interaction)
        
        # Keep only last 100 interactions per user
        if len(self.episodic_memory[user_session]) > 100:
            self.episodic_memory[user_session] = self.episodic_memory[user_session][-100:]
        
        # Extract patterns for long-term memory
        self.extract_patterns(user_session, interaction)
        
        # Log interaction
        self.log_interaction(user_session, interaction)
    
    def extract_patterns(self, user_session: str, interaction: Dict[str, Any]):
        """Extract patterns from interactions for long-term learning"""
        
        # Extract keywords from input
        keywords = interaction["input"].lower().split()
        significant_keywords = [w for w in keywords if len(w) > 3]
        
        # Update patterns in long-term memory
        for keyword in significant_keywords:
            pattern_found = False
            for pattern in self.long_term_memory["patterns"]:
                if pattern["keyword"] == keyword:
                    pattern["frequency"] += 1
                    pattern["last_seen"] = time.time()
                    pattern["modes"].add(interaction["mode"])
                    pattern_found = True
                    break
            
            if not pattern_found:
                self.long_term_memory["patterns"].append({
                    "keyword": keyword,
                    "frequency": 1,
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "modes": {interaction["mode"]}
                })
        
        # Keep top 1000 patterns
        self.long_term_memory["patterns"] = sorted(
            self.long_term_memory["patterns"],
            key=lambda x: x["frequency"],
            reverse=True
        )[:1000]
        
        self.save_long_term_memory()
    
    def get_relevant_context(self, user_session: str, prompt: str, mode: str) -> Dict[str, Any]:
        """Get relevant context for generation"""
        
        context = {
            "recent_interactions": [],
            "relevant_patterns": [],
            "user_preferences": {},
            "session_context": {}
        }
        
        # Get recent interactions
        if user_session in self.episodic_memory:
            context["recent_interactions"] = self.episodic_memory[user_session][-5:]
        
        # Get relevant patterns
        prompt_keywords = prompt.lower().split()
        for pattern in self.long_term_memory["patterns"][:20]:
            if any(keyword in prompt_keywords for keyword in [pattern["keyword"]]):
                context["relevant_patterns"].append(pattern)
        
        # Get user preferences
        if user_session in self.long_term_memory["user_preferences"]:
            context["user_preferences"] = self.long_term_memory["user_preferences"][user_session]
        
        return context
    
    def get_user_stats(self, user_session: str) -> Dict[str, Any]:
        """Get statistics for a user session"""
        
        total_interactions = 0
        if user_session in self.episodic_memory:
            total_interactions = len(self.episodic_memory[user_session])
        
        return {
            "total_interactions": total_interactions,
            "learning_tasks": len(self.long_term_memory["patterns"]),
            "success_rate": 0.85,  # Placeholder
            "recent_performance": 0.78  # Placeholder
        }
    
    def clear_user_memory(self, user_session: str):
        """Clear memory for a specific user session"""
        if user_session in self.episodic_memory:
            del self.episodic_memory[user_session]
        
        if user_session in self.long_term_memory["user_preferences"]:
            del self.long_term_memory["user_preferences"][user_session]
        
        self.save_long_term_memory()
    
    def log_interaction(self, user_session: str, interaction: Dict[str, Any]):
        """Log interaction to file"""
        try:
            log_file = os.path.join(self.logs_dir, f"{user_session}.json")
            
            logs = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            
            logs.append(interaction)
            
            # Keep only last 1000 logs
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"Error logging interaction: {e}")
