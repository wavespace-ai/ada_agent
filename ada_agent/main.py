import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .core.agent import Agent
# Lazy import providers to allow running with partial dependencies
# from .core.llm.openai_compatible import OpenAICompatibleProvider
# from .core.llm.anthropic_client import AnthropicProvider
# from .core.llm.gemini_client import GeminiProvider

def main():
    load_dotenv()
    
    # Provider Selection Logic
    # Default to DeepSeek if no args or specialized config
    # We can use an env var PROVIDER or check which keys exist
    
    provider_name = os.getenv("LLM_PROVIDER", "deepseek").lower()
    
    provider = None
    
    try:
        if provider_name == "deepseek":
            from .core.llm.openai_compatible import OpenAICompatibleProvider
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key: raise ValueError("DEEPSEEK_API_KEY not found")
            provider = OpenAICompatibleProvider(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                model_name="deepseek-chat"
            )
            
        elif provider_name == "grok":
            from .core.llm.openai_compatible import OpenAICompatibleProvider
            api_key = os.getenv("GROK_API_KEY")
            if not api_key: raise ValueError("GROK_API_KEY not found")
            provider = OpenAICompatibleProvider(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                model_name="grok-beta"
            )
            
        elif provider_name == "claude" or provider_name == "anthropic":
            from .core.llm.anthropic_client import AnthropicProvider
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("ANTHROPIC_API_KEY not found")
            provider = AnthropicProvider(api_key=api_key)
            
        elif provider_name == "gemini":
            from .core.llm.gemini_client import GeminiProvider
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key: raise ValueError("GEMINI_API_KEY not found")
            provider = GeminiProvider(api_key=api_key)
            
        else:
            print(f"Unknown provider: {provider_name}")
            return

        print(f"Initializing Agent with Provider: {provider_name.upper()}...")
        agent = Agent(provider=provider, verbose=True) # Enable verbose for demo
        print(f"Loaded {len(agent.loaded_skill_names)} active skills (Discovery only initially).")
        
        print("\n--- ADA: Autonomous Digital Agent ---")
        print(f"Provider: {provider_name}")
        print("Type 'exit' to quit.")
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                response = agent.chat(user_input)
                print(f"Bot: {response}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

if __name__ == "__main__":
    main()
