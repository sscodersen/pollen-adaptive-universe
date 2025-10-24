import os
import httpx
from typing import AsyncGenerator
import json

class AIService:
    def __init__(self):
        self.model_url = os.getenv("AI_MODEL_URL", "")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        
    async def stream_response(self, prompt: str, context: dict = None) -> AsyncGenerator[str, None]:
        """
        Stream AI responses using SSE
        """
        ctx = context if context is not None else {}
        if self.model_url:
            async for chunk in self._stream_from_custom_model(prompt, ctx):
                yield chunk
        elif self.api_key:
            async for chunk in self._stream_from_openai(prompt, ctx):
                yield chunk
        else:
            yield json.dumps({"error": "No AI model configured. Please set AI_MODEL_URL or OPENAI_API_KEY"})
    
    async def _stream_from_custom_model(self, prompt: str, context: dict) -> AsyncGenerator[str, None]:
        """
        Stream from custom Pollen AI model
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
            yield json.dumps({"error": f"Custom model error: {str(e)}"})
    
    async def _stream_from_openai(self, prompt: str, context: dict) -> AsyncGenerator[str, None]:
        """
        Stream from OpenAI API
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line and line.startswith("data: "):
                            data = line[6:]
                            if data != "[DONE]":
                                try:
                                    parsed = json.loads(data)
                                    if "choices" in parsed and len(parsed["choices"]) > 0:
                                        delta = parsed["choices"][0].get("delta", {})
                                        text = delta.get("content", "")
                                        if text:
                                            yield json.dumps({"text": text})
                                except json.JSONDecodeError:
                                    pass
        except Exception as e:
            yield json.dumps({"error": f"OpenAI error: {str(e)}"})

ai_service = AIService()
