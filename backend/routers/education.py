from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.get("/learn")
async def learning_assistant(
    query: str = Query(...),
    subject: str = Query("general"),
    level: str = Query("intermediate"),
    learning_style: str = Query(None)
):
    """
    Stream educational content and tutoring using SSE
    """
    async def generate():
        prompt = f"Educational assistance: {query}"
        if subject:
            prompt += f" in {subject}"
        prompt += f" at {level} level"
        if learning_style:
            prompt += f" optimized for {learning_style} learning style"
        
        prompt += ". Provide clear explanations, examples, and practice suggestions."
        
        try:
            yield {"event": "message", "data": json.dumps({"text": "Preparing your lesson...\n\n"})}
            async for chunk in ai_service.stream_response(prompt, {}):
                yield {"event": "message", "data": chunk}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            yield {"event": "done", "data": ""}
    
    return EventSourceResponse(generate())

@router.get("/subjects")
async def get_subjects():
    return {
        "subjects": [
            "Mathematics",
            "Science",
            "Programming",
            "Languages",
            "History",
            "Arts"
        ]
    }
