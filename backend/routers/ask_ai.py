from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from backend.services.ai_service import ai_service
import json

router = APIRouter()

@router.get("/{mode}")
async def ask_ai_unified(
    mode: str,
    query: str = Query(..., description="User query"),
    context: str = Query(None, description="Additional context")
):
    """
    Unified Ask AI endpoint that dispatches to appropriate mode handlers
    Supports: chat, code, shopping, travel, health, education, finance, 
              image, video, audio, music, react, automation
    """
    
    async def generate():
        try:
            if mode == "chat":
                async for chunk in ai_service.stream_response(
                    prompt=f"Chat with the user: {query}",
                    context={"mode": "chat"}
                ):
                    yield {"event": "message", "data": chunk}
                    
            elif mode == "code":
                yield {"event": "message", "data": json.dumps({
                    "type": "status",
                    "message": "Generating code..."
                })}
                
                prompt = f"""You are a code assistant. The user needs: {query}

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
                    
            elif mode == "shopping":
                prompt = f"""You are a shopping assistant. Help the user find products.

User request: {query}

Provide:
1. Product recommendations with details
2. Price ranges
3. Where to buy
4. Key features to consider"""
                
                async for chunk in ai_service.stream_response(prompt, context={"mode": "shopping"}):
                    yield {"event": "message", "data": chunk}
                    
            elif mode == "travel":
                prompt = f"""You are a travel planning assistant.

User request: {query}

Provide:
1. Destination recommendations
2. Itinerary suggestions
3. Budget estimates
4. Travel tips"""
                
                async for chunk in ai_service.stream_response(prompt, context={"mode": "travel"}):
                    yield {"event": "message", "data": chunk}
                    
            elif mode == "health":
                prompt = f"""You are a health and wellness advisor. Provide evidence-based guidance.

IMPORTANT: This is general wellness information, not medical advice.

User request: {query}

Provide helpful wellness information."""
                
                async for chunk in ai_service.stream_response(prompt, context={"mode": "health"}):
                    yield {"event": "message", "data": chunk}
                    
            elif mode == "education":
                prompt = f"""You are an educational assistant helping users learn.

User request: {query}

Provide:
1. Clear explanations
2. Examples
3. Practice suggestions
4. Learning resources"""
                
                async for chunk in ai_service.stream_response(prompt, context={"mode": "education"}):
                    yield {"event": "message", "data": chunk}
                    
            elif mode == "finance":
                prompt = f"""You are a financial advisor providing budgeting and investment guidance.

IMPORTANT: This is general financial information, not professional advice.

User request: {query}

Provide helpful financial information."""
                
                async for chunk in ai_service.stream_response(prompt, context={"mode": "finance"}):
                    yield {"event": "message", "data": chunk}
                    
            elif mode == "image":
                yield {"event": "message", "data": json.dumps({
                    "type": "status",
                    "message": "Generating image..."
                })}
                
                yield {"event": "message", "data": json.dumps({
                    "type": "image",
                    "url": "/api/placeholder-image",
                    "prompt": query,
                    "message": "Image generation capability - integration pending"
                })}
                
            elif mode == "video":
                yield {"event": "message", "data": json.dumps({
                    "type": "status",
                    "message": "Generating video..."
                })}
                
                yield {"event": "message", "data": json.dumps({
                    "type": "video",
                    "message": "Video generation capability - integration pending",
                    "prompt": query
                })}
                
            elif mode == "audio":
                yield {"event": "message", "data": json.dumps({
                    "type": "status",
                    "message": "Processing audio..."
                })}
                
                yield {"event": "message", "data": json.dumps({
                    "type": "audio",
                    "message": "Audio generation capability - integration pending",
                    "prompt": query
                })}
                
            elif mode == "music":
                yield {"event": "message", "data": json.dumps({
                    "type": "status",
                    "message": "Composing music..."
                })}
                
                yield {"event": "message", "data": json.dumps({
                    "type": "music",
                    "message": "Music generation capability - integration pending",
                    "prompt": query
                })}
                
            elif mode == "react":
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
                    
            elif mode == "automation":
                prompt = f"""You are a task automation expert. The user wants to automate: {query}

Provide:
1. Task Analysis
2. Required Tools/Technologies
3. Step-by-step Automation Plan
4. Code/Script Examples
5. Implementation Tips"""
                
                async for chunk in ai_service.stream_response(prompt, context={"mode": "automation"}):
                    yield {"event": "message", "data": chunk}
                    
            else:
                yield {"event": "message", "data": json.dumps({
                    "type": "error",
                    "error": f"Unknown mode: {mode}"
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
