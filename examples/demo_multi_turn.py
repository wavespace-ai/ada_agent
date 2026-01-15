import os
import sys

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ada_agent import Agent, OpenAICompatibleProvider

def main():
    print("--- Multi-turn Conversation Demo ---")
    
    # 1. Setup Provider (Using DeepSeek by default, can be swapped)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    # if not api_key: ... (Optional warning or let it fail naturally)

    provider = OpenAICompatibleProvider(
        api_key=api_key, 
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat"
    )
    
    # 2. Initialize Agent
    print("Initialize Agent...")
    agent = Agent(provider=provider)
    
    # 3. Conversation Loop
    # We will simulate a user talking to the agent to show it remembers context.
    
    turns = [
        "Hi, I am planning a trip to Mars.",
        "What was the first destination I mentioned?",
        "Can you give me a travel itinerary for there?"
    ]

    for i, user_input in enumerate(turns):
        print(f"\n[Turn {i+1}] User: {user_input}")
        response = agent.chat(user_input)
        print(f"[Turn {i+1}] Agent: {response}")

if __name__ == "__main__":
    main()
