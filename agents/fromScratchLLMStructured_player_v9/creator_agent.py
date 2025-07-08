"""
Agent Evolver Implementation using Google ADK with LiteLLM/Ollama
This implements a self-improving agent system with multiple specialized agents.
"""

from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.genai import types
from dotenv import load_dotenv
from config import Config
from subagents import (
    StrategizerAgent, 
    CoderAgent, 
    ResearcherAgent, 
    AnalyzerAgent, 
    EvolverAgent, 
    PlayerAgent
)
from typing import Dict, List, Any, Optional
import json
import asyncio
import os
import logging

# Enable debug logging for LiteLLM
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CreatorAgent:
    """
    Main Agent Evolver system that coordinates multiple specialized agents
    to continuously improve and evolve software/agents.
    """
    
    def __init__(self, model_config: Optional[str] = None, api_config: Optional[Dict[str, str]] = None):
        # Configure for Ollama usage
        if model_config is None:
            ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
            
            print(f"🔧 Configuring Ollama with:")
            print(f"   Base URL: {ollama_base_url}")
            print(f"   Model: {ollama_model}")
            
            # Enable LiteLLM debug mode for troubleshooting
            import litellm
            litellm.set_verbose = True
            
            # Try the most compatible Ollama configuration
            try:
                self.model = LiteLlm(
                    model=f"ollama/{ollama_model}",
                    api_base=ollama_base_url
                )
                print("✅ Successfully configured Ollama model")
            except Exception as e:
                print(f"⚠️ Error configuring Ollama model: {e}")
                print("📝 Make sure Ollama is running and the model is available:")
                print(f"   curl {ollama_base_url}/api/tags")
                print(f"   ollama run {ollama_model}")
                raise
        else:
            self.model = model_config
        
        # Use provided API config or load from environment
        if api_config is None:
            api_config = {
                "base_url": Config.EXTERNAL_API_BASE_URL,
                "api_key": Config.EXTERNAL_API_KEY,
                "timeout": Config.EXTERNAL_API_TIMEOUT
            }
        
        self.api_config = api_config
        self.setup_agents()
        
    def setup_agents(self):
        """Initialize all specialized agents"""
        
        # Initialize individual agent classes
        self.strategizer_agent = StrategizerAgent(self.model)
        self.coder_agent = CoderAgent(self.model)
        self.researcher_agent = ResearcherAgent(self.model)
        self.analyzer_agent = AnalyzerAgent(self.model)
        self.evolver_agent = EvolverAgent(self.model)
        self.player_agent = PlayerAgent(self.model, self.api_config)
        
        # Get the underlying LLM agents for orchestration
        self.strategizer = self.strategizer_agent.get_agent()
        self.coder = self.coder_agent.get_agent()
        self.researcher = self.researcher_agent.get_agent()
        self.analyzer = self.analyzer_agent.get_agent()
        self.evolver = self.evolver_agent.get_agent()
        self.player = self.player_agent.get_agent()
    
    async def evolve_system(self, target_system: str, improvement_goal: str) -> Dict[str, Any]:
        """
        Main evolution process that coordinates all agents using a single main runner
        """
        print(f"🚀 Starting evolution process for: {target_system}")
        print(f"📋 Goal: {improvement_goal}")
        
        # Set up ADK session and runner infrastructure
        # Using SQLite database for persistent storage
        db_url = "sqlite:///./my_agent_data.db"
        session_service = DatabaseSessionService(db_url=db_url)
        
        app_name = "agent_evolver"
        user_id = "system_user"
        session_id = f"evolution_{hash(target_system + improvement_goal) % 10000}"
        
        # Initialize session with the evolution task context
        initial_state = {
            "target_system": target_system,
            "improvement_goal": improvement_goal,
            "researcher_task": f"Research best practices for improving {target_system}",
            "strategizer_task": f"Create improvement strategy for {target_system}"
        }
        
        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=initial_state
        )
        
        # Create fresh agent instances for this evolution cycle to avoid parent conflicts
        fresh_researcher = ResearcherAgent(self.model)
        fresh_strategizer = StrategizerAgent(self.model)
        fresh_coder = CoderAgent(self.model)
        fresh_player = PlayerAgent(self.model, self.api_config)
        fresh_analyzer = AnalyzerAgent(self.model)
        fresh_evolver = EvolverAgent(self.model)
        
        # Create the main orchestrating agent structure
        # Phase 1: Research and Strategy (Parallel execution)
        research_strategy_agent = ParallelAgent(
            name="research_strategy_phase",
            sub_agents=[
                fresh_researcher.get_agent(),
                fresh_strategizer.get_agent()
            ]
        )
        
        # Phase 2: Implementation and Analysis (Sequential execution)
        implementation_analysis_agent = SequentialAgent(
            name="implementation_analysis_phase",
            sub_agents=[
                fresh_coder.get_agent(),
                fresh_player.get_agent(),
                fresh_analyzer.get_agent()
            ]
        )
        
        # Main orchestrating agent that coordinates all phases
        main_evolution_agent = SequentialAgent(
            name="main_evolution_orchestrator",
            sub_agents=[
                research_strategy_agent,           # Phase 1: Research & Strategy (Parallel)
                fresh_evolver.get_agent(),         # Phase 2: Evolution Planning
                implementation_analysis_agent,     # Phase 3: Implementation & Analysis (Sequential)
                fresh_evolver.get_agent()          # Phase 4: Final Assessment (reuse evolver)
            ]
        )
        
        # Create single main runner
        main_runner = Runner(
            agent=main_evolution_agent,
            app_name=app_name,
            session_service=session_service
        )
        
        # Execute the entire evolution process with a single comprehensive message
        evolution_message = f"""
        Execute the complete evolution process for {target_system} with the following phases:

        Phase 1 - Research & Strategy (Parallel):
        - Research best practices for improving {target_system}
        - Create improvement strategy for {target_system}
        - Goal: {improvement_goal}

        Phase 2 - Evolution Planning:
        - Create detailed evolution plan based on research findings
        - Use Chain-of-Thought reasoning for approach selection

        Phase 3 - Implementation & Analysis (Sequential):
        - Implement the planned improvements
        - Execute and test the implementation
        - Analyze results and performance

        Phase 4 - Final Assessment:
        - Assess the evolution results
        - Plan next iteration improvements
        - Provide comprehensive evaluation
        """
        
        content = types.Content(
            role='user', 
            parts=[types.Part(text=evolution_message)]
        )
        
        # Execute the complete evolution process
        complete_result = None
        phase_results = []
        
        try:
            print("🔄 Executing complete evolution process...")
            for event in main_runner.run(user_id=user_id, session_id=session_id, new_message=content):
                if event.is_final_response() and event.content and event.content.parts:
                    complete_result = event.content.parts[0].text
                    phase_results.append(complete_result)
                    print(f"✅ Phase completed: {len(phase_results)}")
                
        except Exception as e:
            print(f"⚠️ Error in evolution process: {e}")
            complete_result = f"Evolution process encountered an error: {str(e)}"
        
        # Get final session state
        final_session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
        
        # Parse results from the complete execution
        # Since we're using a single runner, we'll structure the results based on what we expect
        results = {
            "system": target_system,
            "goal": improvement_goal,
            "complete_execution": complete_result,
            "phase_results": phase_results,
            "session_state": final_session.state if final_session else {},
            "status": "completed" if complete_result else "failed"
        }
        
        # Extract individual phase results if possible
        if isinstance(complete_result, str):
            # Try to parse structured results from the complete execution
            try:
                # Look for phase markers in the result
                if "Research" in complete_result or "Strategy" in complete_result:
                    results["research_strategy"] = complete_result
                if "Evolution Plan" in complete_result:
                    results["evolution_plan"] = complete_result
                if "Implementation" in complete_result:
                    results["implementation"] = complete_result
                if "Assessment" in complete_result:
                    results["assessment"] = complete_result
            except:
                # If parsing fails, just use the complete result
                results["research_strategy"] = complete_result
                results["evolution_plan"] = complete_result
                results["implementation"] = complete_result
                results["assessment"] = complete_result
        
        return results
    
    async def continuous_evolution(self, target_system: str, iterations: int = 5):
        """
        Run continuous evolution cycles
        """
        print(f"🔄 Starting continuous evolution for {iterations} iterations")
        
        evolution_history = []
        current_goal = "Initial system optimization"
        
        for i in range(iterations):
            print(f"\n--- Evolution Iteration {i+1}/{iterations} ---")
            
            try:
                result = await self.evolve_system(target_system, current_goal)
                evolution_history.append(result)
                
                # Extract next goal from assessment with proper error handling
                if result and isinstance(result, dict):
                    assessment = result.get("assessment", "")
                    if isinstance(assessment, dict):
                        next_goal = assessment.get("next_goal", "Continue optimization")
                    else:
                        # If assessment is a string, use a default next goal
                        next_goal = f"Continue optimization based on iteration {i+1}"
                else:
                    next_goal = f"Continue optimization (iteration {i+1} had issues)"
                
                current_goal = next_goal
                print(f"✅ Iteration {i+1} completed")
                
            except Exception as e:
                print(f"⚠️ Error in iteration {i+1}: {e}")
                # Create a fallback result
                fallback_result = {
                    "system": target_system,
                    "goal": current_goal,
                    "research_strategy": f"Iteration {i+1} failed due to error: {str(e)}",
                    "evolution_plan": None,
                    "implementation": None,
                    "assessment": None,
                    "status": "failed"
                }
                evolution_history.append(fallback_result)
                current_goal = f"Recover from iteration {i+1} failure"
            
        return {
            "total_iterations": iterations,
            "evolution_history": evolution_history,
            "final_state": evolution_history[-1] if evolution_history else None
        }

