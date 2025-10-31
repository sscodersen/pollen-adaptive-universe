"""
Pollen AI Playground Router
Handles voice, image, video, code, tasks, automation, and reAct modes
"""

from fastapi import APIRouter, Query, File, UploadFile
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json
import asyncio
from typing import Optional

router = APIRouter()


@router.get("/chat")
async def chat_mode(message: str = Query(..., description="Chat message")):
    """
    General AI chat mode with SSE streaming
    """
    async def generate():
        async for chunk in ai_service.stream_response(
            prompt=f"Chat with the user: {message}",
            context={"mode": "chat"}
        ):
            yield {"event": "message", "data": chunk}
    
    return EventSourceResponse(generate())


@router.get("/code-assist")
async def code_assist_mode(request: str = Query(..., description="Code request")):
    """
    Code assistance mode - generates code and explanations
    """
    async def generate():
        try:
            yield {"event": "message", "data": json.dumps({
                "type": "status",
                "message": "Generating code..."
            })}
            
            prompt = f"""You are a code assistant. The user needs: {request}

Generate clean, well-documented code with explanations.
Format:
1. First send the code
2. Then send the explanation"""
            
            full_response = ""
            async for chunk in ai_service.stream_response(prompt, context={"mode": "code"}):
                data = json.loads(chunk)
                if "text" in data:
                    full_response += data["text"]
            
            if "```" in full_response:
                parts = full_response.split("```")
                if len(parts) >= 2:
                    code = parts[1].strip()
                    if code.startswith(('python', 'javascript', 'java', 'cpp', 'rust')):
                        code = '\n'.join(code.split('\n')[1:])
                    
                    yield {"event": "message", "data": json.dumps({
                        "type": "code",
                        "content": code
                    })}
                    
                    explanation = parts[2].strip() if len(parts) > 2 else "Code generated successfully"
                    yield {"event": "message", "data": json.dumps({
                        "type": "explanation",
                        "content": explanation
                    })}
            else:
                yield {"event": "message", "data": json.dumps({
                    "type": "code",
                    "content": full_response
                })}
            
            yield {"event": "message", "data": json.dumps({
                "type": "complete"
            })}
            
        except Exception as e:
            yield {"event": "message", "data": json.dumps({
                "type": "error",
                "error": str(e)
            })}
    
    return EventSourceResponse(generate())


@router.get("/react-mode")
async def react_mode(query: str = Query(..., description="Query for ReAct mode")):
    """
    ReAct mode: Reasoning + Acting
    AI reasons through the problem and takes actions
    """
    async def generate():
        try:
            prompt = f"""You are in ReAct mode (Reasoning + Acting).

User Query: {query}

Think through this step by step:
1. THOUGHT: What is the user asking?
2. REASONING: How should I approach this?
3. ACTION: What steps should I take?
4. OBSERVATION: What did I learn?
5. FINAL ANSWER: Provide the solution

Be thorough and show your reasoning."""
            
            async for chunk in ai_service.stream_response(prompt, context={"mode": "react"}):
                yield {"event": "message", "data": chunk}
            
            yield {"event": "message", "data": json.dumps({
                "type": "complete"
            })}
            
        except Exception as e:
            yield {"event": "message", "data": json.dumps({
                "type": "error",
                "error": str(e)
            })}
    
    return EventSourceResponse(generate())


@router.get("/automate-task")
async def automate_task(task: str = Query(..., description="Task to automate")):
    """
    Task automation mode - breaks down tasks into actionable steps
    """
    async def generate():
        try:
            prompt = f"""You are a task automation expert. The user wants to automate: {task}

Provide:
1. Task Analysis
2. Required Tools/Technologies
3. Step-by-step Automation Plan
4. Code/Script Examples
5. Implementation Tips"""
            
            async for chunk in ai_service.stream_response(prompt, context={"mode": "automation"}):
                yield {"event": "message", "data": chunk}
            
            yield {"event": "message", "data": json.dumps({
                "type": "complete"
            })}
            
        except Exception as e:
            yield {"event": "message", "data": json.dumps({
                "type": "error",
                "error": str(e)
            })}
    
    return EventSourceResponse(generate())


@router.post("/generate-image")
async def generate_image(prompt: dict):
    """
    Image generation endpoint
    Note: Returns a placeholder until Pollen AI image generation is integrated
    """
    return {
        "image_url": "https://via.placeholder.com/512x512/667eea/ffffff?text=Image+Generation+Coming+Soon",
        "description": f"Image generation for: {prompt.get('prompt', '')}",
        "note": "Connect your Pollen AI image generation model to enable this feature"
    }


@router.post("/generate-video")
async def generate_video(prompt: dict):
    """
    Video generation endpoint
    Note: Returns a placeholder until Pollen AI video generation is integrated
    """
    return {
        "video_url": None,
        "description": f"Video generation for: {prompt.get('prompt', '')}",
        "note": "Connect your Pollen AI video generation model to enable this feature"
    }


@router.post("/text-to-speech")
async def text_to_speech(data: dict):
    """
    Text-to-speech endpoint
    Note: Returns a placeholder until Pollen AI TTS is integrated
    """
    return StreamingResponse(
        iter([b""]),
        media_type="audio/mpeg",
        headers={
            "X-Note": "Connect your Pollen AI TTS model to enable this feature"
        }
    )


@router.post("/voice-to-text")
async def voice_to_text(audio: UploadFile = File(...)):
    """
    Voice-to-text (speech recognition) endpoint
    Note: Returns a placeholder until Pollen AI STT is integrated
    """
    return {
        "text": "Voice-to-text transcription will appear here once Pollen AI STT is connected",
        "note": "Upload your audio and it will be transcribed when the service is configured"
    }


@router.get("/health")
async def playground_health():
    """
    Health check for playground endpoints
    """
    return {
        "status": "operational",
        "service": "Pollen AI Playground",
        "version": "1.0.0",
        "modes": {
            "chat": "active",
            "voice": "pending_integration",
            "image": "pending_integration",
            "video": "pending_integration",
            "code": "active",
            "tasks": "active",
            "react": "active"
        }
    }
