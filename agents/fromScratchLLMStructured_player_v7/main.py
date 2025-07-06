"""
Main entry point for the Agent Evolver system
"""

from agent_evolver import AgentEvolver
import json
import asyncio


async def main():
    """
    Example of how to use the Agent Evolver
    Note: Ollama configuration is loaded from .env file automatically
    """
    # Initialize the Agent Evolver (Ollama configuration loaded from .env file)
    evolver = AgentEvolver()
    
    # Single evolution cycle
    result = await evolver.evolve_system(
        target_system="game_playing_agent",
        improvement_goal="Improve win rate and decision-making speed"
    )
    
    print("\n🎉 Evolution Result:")
    print(json.dumps(result, indent=2))
    
    # Continuous evolution
    continuous_result = await evolver.continuous_evolution(
        target_system="game_playing_agent",
        iterations=3
    )
    
    print("\n🔄 Continuous Evolution Completed!")
    print(f"Total iterations: {continuous_result['total_iterations']}")


if __name__ == "__main__":
    asyncio.run(main())
