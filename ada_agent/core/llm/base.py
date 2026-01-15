from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Any = "auto") -> Any:
        """
        Sends a chat request to the LLM provider.
        
        Args:
            messages: List of message dictionaries (role, content).
            tools: List of tool schemas.
            tool_choice: Tool choice strategy.
            
        Returns:
            The raw response object from the provider (needs to be normalized by the caller or this method).
            Ideally, we return a normalized response object, but for now we might return the raw one and let the agent handle it 
            if we are sticking to OpenAI format as the internal standard. 
            
            Actually, to make it truly generic, we should return a normalized object. 
            However, the current agent code heavily relies on OpenAI response structure (choices[0].message.tool_calls).
            So the providers should try to return something that looks like an OpenAI response object 
            OR we return a normalized dict/object.
            
            Let's stick to returning an object that has attributes compatible with OpenAI's response structure 
            (choices[0].message.content, choices[0].message.tool_calls) to minimize Agent refactoring.
        """
        pass
