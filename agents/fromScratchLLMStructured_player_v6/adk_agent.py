#!/usr/bin/env python3
"""
Simple ADK agent for Catanatron player evolution.
This is a simplified single-agent version using Google ADK.
"""

import os
from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.runner import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Configuration
FOO_TARGET_FILENAME = "foo_player.py"
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME

def read_foo_player() -> str:
    """Read the current foo_player.py file."""
    try:
        return FOO_TARGET_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "File not found"

def write_foo_player(new_code: str) -> str:
    """Write new code to the foo_player.py file."""
    try:
        FOO_TARGET_FILE.write_text(new_code, encoding="utf-8")
        return f"Successfully updated {FOO_TARGET_FILENAME}"
    except Exception as e:
        return f"Error writing file: {e}"

def run_catanatron_test() -> str:
    """Run a Catanatron test game."""
    import subprocess
    import shlex
    
    try:
        result = subprocess.run(
            shlex.split("catanatron-play --players=AB,FOO_LLM_S5_M --num=1 --config-vps-to-win=5"),
            capture_output=True,
            text=True,
            timeout=300,
            check=False
        )
        return f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Game timed out (as expected for long games)"
    except Exception as e:
        return f"Error running game: {e}"

# Create the main evolution agent
def create_evolution_agent():
    """Create the main ADK agent for evolving the Catanatron player."""
    
    agent = LlmAgent(
        model="gemini-2.0-flash",
        name="catanatron_evolution_agent",
        description="An agent that evolves and improves Catanatron game players",
        instruction="""
        You are an expert AI agent that specializes in evolving and improving Catanatron (Settlers of Catan) game players.

        Your main tasks:
        1. Analyze the current foo_player.py implementation
        2. Identify areas for improvement in the game strategy
        3. Write better code for the player
        4. Test the player by running games
        5. Iterate and improve based on results

        When analyzing code, look for:
        - Syntax errors and bugs
        - Strategic improvements
        - Better decision-making logic
        - Proper use of game mechanics

        When writing code:
        - Follow Python best practices
        - Add helpful comments
        - Use proper error handling
        - Implement smart game strategies

        Always provide detailed explanations of your changes and reasoning.
        """,
        tools=[read_foo_player, write_foo_player, run_catanatron_test]
    )
    
    return agent

def main():
    """Main function to run the ADK evolution agent."""
    
    # Check for API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        print("Then run: export GOOGLE_API_KEY='your-api-key-here'")
        return 1
    
    print("Creating ADK Evolution Agent...")
    
    # Create session service
    session_service = InMemorySessionService()
    app_name = "catanatron_evolution"
    user_id = "evolution_user"
    session_id = "evolution_session"
    
    # Create session
    session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    
    # Create agent and runner
    agent = create_evolution_agent()
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service
    )
    
    print("Agent created successfully!")
    print("You can now interact with the agent to evolve your Catanatron player.")
    print("\nExample queries:")
    print("- 'Analyze the current foo_player.py and suggest improvements'")
    print("- 'Write a better strategy for the Catanatron player'")
    print("- 'Run a test game and analyze the results'")
    print("\nType 'quit' to exit")
    print("-" * 50)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input:
                continue
            
            print("Agent: Working...")
            
            # Create user message
            user_content = types.Content(
                role='user', 
                parts=[types.Part(text=user_input)]
            )
            
            # Get response from agent
            events = runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content
            )
            
            # Print the final response
            for event in events:
                if event.is_final_response() and event.content and event.content.parts:
                    response = event.content.parts[0].text
                    print(f"Agent: {response}")
                    break
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 