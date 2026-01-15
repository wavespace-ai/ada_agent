from .base import LLMProvider
import anthropic
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Mock objects to mimic OpenAI response structure for compatibility with existing Agent code
@dataclass
class MockFunction:
    name: str
    arguments: str

@dataclass
class MockToolCall:
    id: str
    function: MockFunction
    type: str = "function"

@dataclass
class MockMessage:
    content: Optional[str]
    tool_calls: Optional[List[MockToolCall]]
    def model_dump(self):
        # Return dict representation compatible with the agent's expectations
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in self.tool_calls
            ]
        return d

@dataclass
class MockChoice:
    message: MockMessage

@dataclass
class MockResponse:
    choices: List[MockChoice]

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Any = "auto") -> Any:
        # Convert OpenAI messages to Anthropic format
        system_prompt = ""
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            elif msg["role"] == "tool":
                # Anthropic expects tool results to be part of the user role or a specific tool_result block
                # OpenAI: role="tool", tool_call_id="..."
                # Anthropic: role="user", content=[{"type": "tool_result", "tool_use_id": ..., "content": ...}]
                filtered_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": msg["content"]
                        }
                    ]
                })
            elif msg["role"] == "assistant" and "tool_calls" in msg:
                # Convert OpenAI tool calls to Anthropic tool_use
                # OpenAI: tool_calls=[{function: {name, arguments}, id}]
                # Anthropic: content=[{type: "tool_use", id, name, input}]
                content_block = []
                if msg.get("content"):
                    content_block.append({"type": "text", "text": msg["content"]})
                
                for tc in msg["tool_calls"]:
                    # Check if tc is dict or object (Agent might store dicts in history)
                    tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                    func = tc["function"] if isinstance(tc, dict) else tc.function
                    fname = func["name"] if isinstance(func, dict) else func.name
                    import json
                    fargs = json.loads(func["arguments"]) if isinstance(func, dict) else json.loads(func.arguments)
                    
                    content_block.append({
                        "type": "tool_use",
                        "id": tc_id,
                        "name": fname,
                        "input": fargs
                    })
                filtered_messages.append({"role": "assistant", "content": content_block})
            else:
                filtered_messages.append(msg)

        # Convert Tools
        anthropic_tools = []
        if tools:
            for t in tools:
                if t["type"] == "function":
                    anthropic_tools.append({
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "input_schema": t["function"]["parameters"]
                    })
        
        # Make request
        kwargs = {
            "model": self.model_name,
            "messages": filtered_messages,
            "max_tokens": 4096,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            # Anthropic tool_choice is different, simplified here:
            if tool_choice == "required": 
                kwargs["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                pass # Default is auto
            # else Leave as auto

        response = self.client.messages.create(**kwargs)

        # Convert back to OpenAI Response format for Agent compatibility
        content_text = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                import json
                tool_calls.append(MockToolCall(
                    id=block.id,
                    function=MockFunction(
                        name=block.name,
                        arguments=json.dumps(block.input)
                    )
                ))
        
        return MockResponse(choices=[
            MockChoice(message=MockMessage(
                content=content_text if content_text else None,
                tool_calls=tool_calls if tool_calls else None
            ))
        ])
