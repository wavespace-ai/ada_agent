import os
import sys

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ada_agent import Agent, init, OpenAICompatibleProvider
from dotenv import load_dotenv

def main():
    load_dotenv()
    print("--- Simple RAG Demo with Context ---")
    
    # 1. Setup Provider (DeepSeek)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Please set DEEPSEEK_API_KEY in .env")
        return

    base_url = "https://api.deepseek.com/v1"
    model_name = "deepseek-chat"
    
    provider = OpenAICompatibleProvider(
        api_key=api_key, 
        base_url=base_url,
        model_name=model_name
    )
    
    # 2. Setup Context
    # We want to run this in the 'examples' folder context or similar.
    # For this demo, we assume 'examples/context' is the target.
    # Note: In a real app, user would run 'ada_agent.init()' in their root.
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    context_dir = os.path.join(base_dir, "context")
    
    # If using the init flow programmatically:
    if not os.path.exists(context_dir):
        print("Context not found. Initializing...")
        init(base_dir) # This creates base_dir/context
    else:
        print(f"Using existing context at {context_dir}")

    # Paths based on the standard init structure
    knowledge_path = os.path.join(context_dir, "knowledge")
    memory_path = os.path.join(context_dir, "memory", "memory.json")
    skills_path = os.path.join(context_dir, "skills")
    persona_path = os.path.join(context_dir, "persona")
    
    # 3. Initialize Agent
    print(f"Initializing agent with:\n- Knowledge: {knowledge_path}\n- Memory: {memory_path}\n- Skills: {skills_path}\n- Persona: {persona_path}")
    agent = Agent(
        provider=provider, 
        knowledge_path=knowledge_path, 
        memory_path=memory_path,
        skills_dirs=[skills_path],
        persona_path=persona_path,
        verbose=True
    )
    
    # 4. Ask a question that requires RAG
    # Data: "The capital of Mars is Utopia Planitia."
    question = "What is the capital of Mars in this universe?"
    print(f"\nUser: {question}")
    
    response = agent.chat(question)
    
    print(f"\nAgent: {response}")

if __name__ == "__main__":
    main()
