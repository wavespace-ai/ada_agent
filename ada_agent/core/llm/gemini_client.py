from .base import LLMProvider
import google.generativeai as genai
from google.generativeai.types import content_types
from google.protobuf import struct_pb2
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

# Reuse Mock objects
@dataclass
class MockFunction:
    name: str
    arguments: str

@dataclass
class MockToolCall:
    id: str  # Gemini function calls don't strictly have IDs in the same way, but we will generate key
    function: MockFunction
    type: str = "function"

@dataclass
class MockMessage:
    content: Optional[str]
    tool_calls: Optional[List[MockToolCall]]
    def model_dump(self):
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

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _convert_tool_calls_to_parts(self, tool_calls_data):
        parts = []
        for tc in tool_calls_data:
            func = tc["function"] if isinstance(tc, dict) else tc.function
            name = func["name"] if isinstance(func, dict) else func.name
            args = json.loads(func["arguments"]) if isinstance(func, dict) else json.loads(func.arguments)
            
            fc = content_types.FunctionCall(name=name, args=args)
            parts.append(fc)
        return parts

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Any = "auto") -> Any:
        # Convert History
        # Gemini uses 'user' and 'model'. System instructions are passed at init or separate.
        # We will try to map system role to system instruction if possible, or merge into first user msg
        
        gemini_history = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content = msg.get("content")
            
            if role == "system":
                # Gemini supports system_instruction on GenerativeModel init or request
                # We can prepend to history or set as instruction. 
                # Ideally, we collect all system prompts.
                if system_instruction is None:
                    system_instruction = content
                else:
                    system_instruction += "\n" + content
            
            elif role == "user":
                gemini_history.append({"role": "user", "parts": [content]})
            
            elif role == "assistant":
                parts = []
                if content:
                    parts.append(content)
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Convert OpenAI tool calls to Gemini FunctionCalls
                    # Note: We need to reconstruct the FunctionCall objects
                     fc_parts = self._convert_tool_calls_to_parts(msg["tool_calls"])
                     parts.extend(fc_parts)
                
                gemini_history.append({"role": "model", "parts": parts})
                
            elif role == "tool":
                # Gemini expects FunctionResponse
                # We need to match the previous FunctionCall. 
                # OpenAI has tool_call_id. Gemini flow is strictly sequential in parts usually.
                # Simplification: We map the tool response to a FunctionResponse part.
                # We need the function name for the response, which OpenAI role='tool' doesn't explicitly have without lookback.
                # BUT the agent stores the history. The provider receives the FULL history.
                # We can attempt to deduce the name or just pass the ID as part of the response if Gemini allowed.
                # Gemini SDK wants: part = FunctionResponse(name=..., response={...})
                
                # Look back in messages to find the tool call with this ID
                # This is O(N) but history is short.
                func_name = "unknown_tool"
                for prev in reversed(messages):
                    if prev.get("tool_calls"):
                        for tc in prev["tool_calls"]:
                            tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                            if tc_id == msg["tool_call_id"]:
                                func = tc["function"] if isinstance(tc, dict) else tc.function
                                func_name = func["name"] if isinstance(func, dict) else func.name
                                break
                    if func_name != "unknown_tool": 
                        break
                
                # Gemini expects dict response
                try:
                    resp_dict = json.loads(msg["content"])
                except:
                    resp_dict = {"result": msg["content"]}
                    
                gemini_history.append({
                    "role": "user",
                    "parts": [
                        content_types.FunctionResponse(name=func_name, response=resp_dict)
                    ]
                })

        # Configure Tools
        gemini_tools = []
        if tools:
            # Convert OpenAI schema to Gemini
            # Luckily Gemini SDK supports passing the function declarations relatively easily
            # But we have JSON schema.
            # We can pass a list of tools.
            declarations = []
            for t in tools:
                if t["type"] == "function":
                     # We need to convert JSON Schema to Gemini's format manually or use helpers
                     # For simplicity in this Adapter, we will rely on internal helpers or
                     # just construct the Tool object if possible.
                     # Actually, `genai.configure` and passing tools arg often accepts list of functions.
                     # Since we have schema, we might need `genai.protos.Tool`.
                     
                     # Simple workaround: The `tools` arg in `model.generate_content` is robust.
                     # We can define the function declaration dictionary.
                     f = t["function"]
                     declarations.append({
                         "name": f["name"],
                         "description": f.get("description"),
                         "parameters": f.get("parameters")
                     })
            gemini_tools = [declarations] # List of lists of tools (Tool dicts)

        # Generate
        # Note: Gemini 1.5/2.0 supports system_instruction argument in generate_content (sometimes) or model init.
        # Safest is model init, but we want stateless per request if possible.
        # `model.generate_content` accepts `system_instruction`? 
        # Yes, in newer versions.
        
        request_options = {}
        if system_instruction:
            # We might need to re-instantiate model if system prompt changes dynamically and isn't supported in generate_content
            # Actually for now let's hope it is.
            # If not, we just prepend to history parts[0].
            pass

        # Since we are using chat, we should use start_chat for history management? 
        # Or just generate_content with full history.
        # generate_content with full history is effectively stateless "chat"
        
        # Adjust history for generate_content: it expects a list of Content objects or dicts
        # The first message must be user.
        if gemini_history and gemini_history[0]["role"] == "model":
             # Insert a dummy user message or merge?
             gemini_history.insert(0, {"role": "user", "parts": ["Hi"]}) 
        
        if system_instruction:
             # Prepend system instruction as a user message or system if supported
             # Current API often prefers system configs on model.
             # Let's try adding it as the very first USER part for compatibility
             if gemini_history:
                existing_parts = gemini_history[0]["parts"]
                if isinstance(existing_parts, list):
                    existing_parts.insert(0, f"System Instruction: {system_instruction}")
                else:
                    # just overwrite?
                    gemini_history[0]["parts"] = [f"System Instruction: {system_instruction}", str(existing_parts)]
        
        response = self.model.generate_content(
            contents=gemini_history,
            tools=gemini_tools or None,
            # tool_config=... # for tool_choice
        )
        
        # Convert response to OpenAI format
        try:
            p = response.parts[0]
        except:
             return MockResponse(choices=[MockChoice(message=MockMessage(content="Error: Empty response from Gemini", tool_calls=None))])
             
        content_text = None
        tool_calls = []
        
        for part in response.parts:
            if part.text:
                if content_text is None: content_text = ""
                content_text += part.text
            if part.function_call:
                import uuid
                tool_calls.append(MockToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}", # Gemini doesn't verify IDs
                    function=MockFunction(
                        name=part.function_call.name,
                        arguments=json.dumps(dict(part.function_call.args))
                    )
                ))
        
        return MockResponse(choices=[
            MockChoice(message=MockMessage(
                content=content_text,
                tool_calls=tool_calls if tool_calls else None
            ))
        ])
