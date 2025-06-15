
import json
import os
from typing import List, Dict, Any

def save_feedback_buffer(feedback_buffer: List[Dict[str, Any]], path: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(feedback_buffer, f)
        print(f"💾 Feedback buffer saved locally to {path}")
    except Exception as e:
        print(f"Failed to save feedback buffer: {e}")

def load_feedback_buffer(path: str) -> List[Dict[str, Any]]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                feedback_buffer = json.load(f)
            print(f"📥 Feedback buffer loaded from {path}")
            return feedback_buffer
        else:
            return []
    except Exception as e:
        print(f"Failed to load feedback buffer: {e}")
        return []
