import os
import httpx
from typing import AsyncGenerator
import json

class AIService:
    def __init__(self):
        self.model_url = os.getenv("AI_MODEL_URL", "")
        
    async def stream_response(self, prompt: str, context: dict | None = None) -> AsyncGenerator[str, None]:
        """
        Stream AI responses using SSE - Pollen AI only
        """
        ctx = context if context is not None else {}
        if self.model_url:
            async for chunk in self._stream_from_pollen_ai(prompt, ctx):
                yield chunk
        else:
            yield json.dumps({"error": "Pollen AI not configured. Please set AI_MODEL_URL environment variable"})
    
    async def _stream_from_pollen_ai(self, prompt: str, context: dict) -> AsyncGenerator[str, None]:
        """
        Stream from Pollen AI model
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    self.model_url,
                    json={"prompt": prompt, "context": context, "stream": True},
                    headers={"Content-Type": "application/json"}
                ) as response:
                    async for line in response.aiter_lines():
                        if line and line.strip():
                            try:
                                parsed = json.loads(line)
                                if "text" in parsed:
                                    yield json.dumps({"text": parsed["text"]})
                                elif "content" in parsed:
                                    yield json.dumps({"text": parsed["content"]})
                                else:
                                    text_value = str(parsed) if isinstance(parsed, (dict, list)) else line
                                    yield json.dumps({"text": text_value})
                            except json.JSONDecodeError:
                                yield json.dumps({"text": line.strip()})
        except Exception as e:
            yield json.dumps({"error": f"Pollen AI error: {str(e)}"})

ai_service = AIService()
