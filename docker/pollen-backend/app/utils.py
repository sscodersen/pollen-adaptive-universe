
from typing import Optional, Dict, Any
from fastapi import Header

def get_user_session(x_session_id: Optional[str] = Header(None)) -> str:
    """Gets user session from header or provides a default."""
    return x_session_id or "default_user"

def process_request(request: Any, user_session: str) -> Dict[str, Any]:
    """Processes the incoming request data."""
    processed = request.dict()
    processed["user_session"] = user_session
    return processed

def format_response(response: Dict[str, Any], generation_time: float, user_session: str) -> Dict[str, Any]:
    """Formats the final response."""
    if 'metadata' not in response:
        response['metadata'] = {}
    
    response['metadata'].update({
        "generation_time": generation_time,
        "user_session": user_session
    })
    
    # Ensure top-level keys are present
    response.setdefault('content', 'No content generated.')
    response.setdefault('confidence', 0.5)
    
    return response

