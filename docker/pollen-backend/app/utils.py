
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
import uuid
import time

security = HTTPBearer(auto_error=False)

def get_user_session(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if credentials:
        return credentials.credentials
    return f"anon-{uuid.uuid4().hex[:8]}"

def process_request(request: Any, user_session: str) -> Dict[str, Any]:
    return {
        "prompt": request.prompt,
        "mode": request.mode,
        "context": request.context or {},
        "user_session": user_session,
        "timestamp": time.time()
    }

def format_response(response: Dict[str, Any], generation_time: float, user_session: str) -> Dict[str, Any]:
    return {
        "content": response["content"],
        "confidence": response.get("confidence", 0.8),
        "reasoning": response.get("reasoning"),
        "metadata": {
            "generation_time": generation_time,
            "user_session": user_session[:8] + "...",
            "model_version": response.get("model_version", "2.0.0")
        }
    }
