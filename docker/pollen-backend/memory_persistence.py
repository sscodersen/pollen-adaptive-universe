
import json
import os
from typing import Dict, Any

def save_user_memory(user_memories: Dict[str, Any], path: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(user_memories, f)
        print(f"💾 User memories saved to {path}")
    except Exception as e:
        print(f"Failed to save user memories: {e}")

def load_user_memory(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                user_memories = json.load(f)
            print(f"📥 User memories loaded from {path}")
            return user_memories
        else:
            return {}
    except Exception as e:
        print(f"Failed to load user memories: {e}")
        return {}

def save_global_stats(global_patterns: Dict[str, float], interaction_counts: Dict[str, int], path: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "global_patterns": global_patterns,
                "interaction_counts": interaction_counts
            }, f)
        print(f"💾 Global stats saved to {path}")
    except Exception as e:
        print(f"Failed to save global stats: {e}")

def load_global_stats(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            print(f"📥 Global stats loaded from {path}")
            return stats.get("global_patterns", {}), stats.get("interaction_counts", {})
        else:
            return {}, {}
    except Exception as e:
        print(f"Failed to load global stats: {e}")
        return {}, {}

