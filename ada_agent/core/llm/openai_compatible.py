from .base import LLMProvider
from openai import OpenAI
from typing import List, Dict, Any, Optional

class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Any = "auto") -> Any:
        # OpenAI SDK handles the format natively
        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
            
        return self.client.chat.completions.create(**kwargs)
