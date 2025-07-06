# Google ADK-powered Multi-Agent System for Evolving Catan Game Players
# =======================================================================
# This file implements a sophisticated multi-agent system using Google's 
# Agent Development Kit (ADK) to evolve and improve Catan game-playing agents.
# The system consists of specialized agents that work together to analyze,
# strategize, research, and code improvements to a Catan player.

import time
import os
import sys, pathlib
# Add parent directory to path to access base_llm module
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import base_llm
from typing import List, Dict, Tuple, Any, Optional
import json
import random
from enum import Enum
from io import StringIO
from datetime import datetime
import shutil
from pathlib import Path
import subprocess, shlex

# Google ADK imports - Core components for the multi-agent system
from google.adk.agents import LlmAgent
from google.adk.runner import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ==================== CONFIGURATION CONSTANTS ====================
# These constants define the core settings for the evolution system

# Path to the local Catanatron game codebase for research and analysis
LOCAL_CATANATRON_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()

# Target filename for the player we're evolving
FOO_TARGET_FILENAME = "foo_player.py"

# Absolute path to the current foo_player.py file being evolved
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME    

# Maximum file size limit to keep within context windows
FOO_MAX_BYTES   = 64_000                                     

# Command to run Catanatron games for testing the evolved player
# AB = AlphaBeta player, FOO_LLM_V6 = our evolved player
FOO_RUN_COMMAND = "catanatron-play --players=AB,FOO_LLM_V6  --num=3  --config-vps-to-win=10"

class CreatorAgent():
    """
    ADK-powered multi-agent system for evolving Catan game players.
    
    This class orchestrates a team of specialized AI agents:
    - Meta Agent: Coordinates and decides which specialist to use
    - Analyzer Agent: Analyzes game performance and identifies issues
    - Strategizer Agent: Develops new strategies and approaches
    - Researcher Agent: Researches game mechanics and code examples
    - Coder Agent: Implements code changes to the player
    
    The system runs evolution cycles where:
    1. The current player is tested in games
    2. Results are analyzed
    3. The meta agent decides what needs improvement
    4. The appropriate specialist agent makes improvements
    5. The cycle repeats to continuously evolve the player
    """
    
    # Class-level properties shared across all instances
    run_dir = None              # Directory for storing run results
    current_evolution = 0       # Counter for evolution cycles

    def __init__(self):
        """
        Initialize the ADK-powered creator agent system.
        
        Sets up:
        - Google ADK with Gemini model
        - Run directory for storing results
        - Session management for agent communication
        - All specialized agents and their runners
        """
        # Configure the LLM model for all agents
        self.llm_name = "gemini-2.0-flash"
        
        # Create unique run directory if it doesn't exist
        # This ensures each run has its own isolated storage
        if CreatorAgentADK.run_dir is None:
            agent_dir = os.path.dirname(os.path.abspath(__file__))
            runs_dir = os.path.join(agent_dir, "runs_adk")
            os.makedirs(runs_dir, exist_ok=True)
            run_id = datetime.now().strftime("adk_creator_%Y%m%d_%H%M%S")
            CreatorAgentADK.run_dir = os.path.join(runs_dir, run_id)
            os.makedirs(CreatorAgentADK.run_dir, exist_ok=True)

        # Copy the template player file to start evolution from a clean slate
        shutil.copy2(
            (Path(__file__).parent / ("__TEMPLATE__" + FOO_TARGET_FILENAME)).resolve(),
            FOO_TARGET_FILE.resolve()
        )

        # Initialize ADK session service for managing agent conversations
        self.session_service = InMemorySessionService()
        self.app_name = "catanatron_creator_adk"
        self.user_id = "creator_user"
        self.session_id = f"session_{run_id}"
        
        # Create the main session that all agents will use
        self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id
        )

        # Create all the specialized agents and their runners
        self.create_adk_agents()

    def create_adk_agents(self):
        """
        Create all ADK agents for the multi-agent system.
        
        This method:
        1. Defines tool functions that agents can use
        2. Creates specialized agents with specific roles and instructions
        3. Sets up runners to execute each agent
        
        Each agent has access to different tools based on their role:
        - Analyzer: Can read files and view game results
        - Strategizer: Has broader access including web search
        - Researcher: Can list files and search for information
        - Coder: Can read and write the player file
        """
        
        # ==================== TOOL FUNCTION DEFINITIONS ====================
        # These functions provide capabilities that agents can use
        
        def list_catanatron_files() -> str:
            """
            Return all files beneath BASE_DIR, one per line.
            
            This tool helps agents discover what files are available
            in the Catanatron codebase for research purposes.
            
            Returns:
                str: Newline-separated list of relative file paths
            """
            return "\n".join(
                str(p.relative_to(LOCAL_CATANATRON_BASE_DIR))
                for p in LOCAL_CATANATRON_BASE_DIR.glob("**/*")
                if p.is_file() and p.suffix in {".py", ".txt", ".md"}
            )

        def read_local_file(rel_path: str) -> str:
            """
            Return the text content of rel_path if it's inside BASE_DIR.
            
            This tool allows agents to read files from:
            - The Catanatron codebase (for research)
            - The current run directory (for accessing results)
            - The current foo_player.py being evolved
            
            Args:
                rel_path: Relative path to the file to read
                
            Returns:
                str: UTF-8 content of the file
                
            Raises:
                ValueError: If file is too large or access is denied
            """
            # Special case: reading the current player file
            if rel_path == FOO_TARGET_FILENAME:
                return read_foo()
            
            # Handle Catanatron codebase files
            if rel_path.startswith("catanatron/"):
                candidate = (LOCAL_CATANATRON_BASE_DIR / rel_path.replace("catanatron/", "")).resolve()
                if not str(candidate).startswith(str(LOCAL_CATANATRON_BASE_DIR)) or not candidate.is_file():
                    raise ValueError("Access denied or not a file")
                if candidate.stat().st_size > 64_000:
                    raise ValueError("File too large")
                return candidate.read_text(encoding="utf-8", errors="ignore")
            
            # Handle run directory files (results, logs, etc.)
            run_path = Path(CreatorAgentADK.run_dir) / rel_path
            if run_path.exists() and run_path.is_file():
                if run_path.stat().st_size > 64_000:
                    raise ValueError("File too large")
                return run_path.read_text(encoding="utf-8", errors="ignore")
            
            # Fallback: try reading from Catanatron base directory
            candidate = (LOCAL_CATANATRON_BASE_DIR / rel_path).resolve()
            if not str(candidate).startswith(str(LOCAL_CATANATRON_BASE_DIR)) or not candidate.is_file():
                raise ValueError(f"Access denied or file not found: {rel_path}")
            if candidate.stat().st_size > 64_000:
                raise ValueError("File too large")
            return candidate.read_text(encoding="utf-8", errors="ignore")

        def read_foo() -> str:
            """
            Return the UTF-8 content of the current player file (≤64 kB).
            
            This is the main function for reading the player being evolved.
            
            Returns:
                str: Current content of foo_player.py
                
            Raises:
                ValueError: If file is too large for processing
            """
            if FOO_TARGET_FILE.stat().st_size > FOO_MAX_BYTES:
                raise ValueError("File too large for the agent")
            return FOO_TARGET_FILE.read_text(encoding="utf-8", errors="ignore")

        def write_foo(new_text: str) -> str:
            """
            Overwrite the player file with new_text (UTF-8).
            
            This is how the Coder agent implements changes to the player.
            
            Args:
                new_text: New Python code for the player
                
            Returns:
                str: Success message
                
            Raises:
                ValueError: If new text is too large
            """
            if len(new_text.encode()) > FOO_MAX_BYTES:
                raise ValueError("Refusing to write >64 kB")
            FOO_TARGET_FILE.write_text(new_text, encoding="utf-8")
            return f"{FOO_TARGET_FILENAME} updated successfully"

        def run_testfoo(short_game: bool = False) -> str:
            """
            Run one Catanatron match and return raw CLI output.
            
            This tool allows agents to test the current player implementation.
            
            Args:
                short_game: If True, runs a shorter test game
                
            Returns:
                str: Game output and results
            """
            return self._run_game_test(short_game)

        def web_search(query: str) -> str:
            """
            Perform a web search for research purposes.
            
            Note: This is currently a placeholder implementation.
            In a production system, this would connect to a real search API.
            
            Args:
                query: Search query string
                
            Returns:
                str: Search results (currently placeholder)
            """
            return f"Search results for: {query}\n[Web search functionality - implement with actual search API]"

        def view_last_game_llm_query(query_number: int = -1) -> str:
            """
            View the game results from a specific run.
            
            This helps agents analyze previous game performance.
            
            Args:
                query_number: Which game run to analyze (-1 for latest)
                
            Returns:
                str: Game query results
            """
            return self._get_game_query_results(query_number)

        def read_game_results_file(num: int = -1) -> str:
            """
            Return the contents of the *.json* game-results file.
            
            This provides structured data about game performance.
            
            Args:
                num: Which evolution's results to read (-1 for latest)
                
            Returns:
                str: JSON game results
            """
            return self._read_game_file("json_game_results_path", num)

        def read_older_foo_file(num: int = -1) -> str:
            """
            Return the contents of an older version foo_player.py file.
            
            This allows agents to compare current and previous implementations.
            
            Args:
                num: Which evolution's player to read (-1 for latest)
                
            Returns:
                str: Previous player implementation
            """
            return self._read_game_file("cur_foo_player_path", num)

        # ==================== AGENT CREATION ====================
        # Create the specialized agents with their specific roles
        
        # META AGENT: The coordinator that decides which specialist to use
        self.meta_agent = LlmAgent(
            model=self.llm_name,
            name="meta_supervisor",
            description="Meta supervisor that coordinates other agents in the multi-agent system",
            instruction=f"""
            You are the META SUPERVISOR in a multi-agent system working to evolve {FOO_TARGET_FILENAME} 
            to become the best Catanatron player.

            HIGH LEVEL GOAL: Learn how to create a Catanatron player that can win games against opponents.

            Your Tasks:
            1. Look at previous messages and analyze your goals and newest information
            2. Output your current MEDIUM LEVEL GOAL and LOW LEVEL GOAL
            3. Determine which sub-agent to consult and prepare an OBJECTIVE message

            AGENTS AVAILABLE:
            - ANALYZER: Has access to performance history, game outputs, and results
            - STRATEGIZER: Generates new strategies and analyzes previous attempts
            - RESEARCHER: Accesses game files and performs research
            - CODER: Writes the {FOO_TARGET_FILENAME} file

            Output Format:
            - MEDIUM LEVEL GOAL: <5 iteration objective>
            - LOW LEVEL GOAL: <next iteration objective>  
            - CHOSEN AGENT: [ANALYZER/STRATEGIZER/RESEARCHER/CODER]
            - AGENT OBJECTIVE: <specific task for the agent>
            """
        )

        # ANALYZER AGENT: Specializes in analyzing game performance and results
        self.analyzer_agent = LlmAgent(
            model=self.llm_name,
            name="analyzer",
            description="Analyzes game performance, outputs, and results",
            instruction=f"""
            You are the ANALYZER expert for evolving the {FOO_TARGET_FILENAME} player.
            
            Your Role:
            - Analyze game outputs and performance history
            - Interpret game results and identify issues
            - Provide detailed reports on player performance
            - Identify syntax errors and implementation problems

            Always start responses with 'ANALYSIS:' and end with 'END ANALYSIS'.
            Focus on concrete data from game results and outputs.
            """,
            tools=[read_local_file, view_last_game_llm_query]
        )

        # STRATEGIZER AGENT: Develops new strategies and approaches
        self.strategizer_agent = LlmAgent(
            model=self.llm_name,
            name="strategizer", 
            description="Develops and refines game strategies",
            instruction=f"""
            You are the STRATEGIZER expert for evolving the {FOO_TARGET_FILENAME} player.
            
            Your Role:
            - Generate new strategic approaches
            - Analyze previous strategy effectiveness
            - Recommend strategic improvements
            - Search for new strategy ideas when needed

            Always start responses with 'STRATEGY:' and end with 'END STRATEGY'.
            Be creative and look for breakthrough approaches.
            """,
            tools=[read_local_file, read_game_results_file, read_older_foo_file, web_search, view_last_game_llm_query]
        )

        # RESEARCHER AGENT: Researches game mechanics and implementation details
        self.researcher_agent = LlmAgent(
            model=self.llm_name,
            name="researcher",
            description="Researches game mechanics and implementation details", 
            instruction=f"""
            You are the RESEARCHER expert for evolving the {FOO_TARGET_FILENAME} player.
            
            Your Role:
            - Research Catanatron game mechanics and codebase
            - Find implementation details and syntax information
            - Provide code examples and API documentation
            - Search for relevant information online

            Always start responses with 'RESEARCH:' and end with 'END RESEARCH'.
            Provide concrete code examples and cite sources.
            """,
            tools=[read_local_file, web_search, list_catanatron_files]
        )

        # CODER AGENT: Implements code changes to the player file
        self.coder_agent = LlmAgent(
            model=self.llm_name,
            name="coder",
            description="Implements code changes to the foo_player.py file",
            instruction=f"""
            You are the CODER expert for evolving the {FOO_TARGET_FILENAME} player.
            
            Your Role:
            - Implement code changes based on META instructions
            - Write syntactically correct Python code
            - Fix bugs and errors
            - Add useful debugging and comments

            Coding Guidelines:
            - Follow Python 3.12 syntax
            - Add plenty of comments
            - Use print statements for debugging
            - Prioritize fixing bugs and errors
            - Don't make up variables or functions

            Always start responses with 'CODER:' and end with 'END CODER'.
            Report on all changes made to the code.
            """,
            tools=[write_foo, read_foo]
        )

        # ==================== RUNNER CREATION ====================
        # Create runners for each agent to handle execution and communication
        
        self.meta_runner = Runner(
            agent=self.meta_agent,
            app_name=self.app_name, 
            session_service=self.session_service
        )

        self.analyzer_runner = Runner(
            agent=self.analyzer_agent,
            app_name=self.app_name,
            session_service=self.session_service  
        )

        self.strategizer_runner = Runner(
            agent=self.strategizer_agent,
            app_name=self.app_name,
            session_service=self.session_service
        )

        self.researcher_runner = Runner(
            agent=self.researcher_agent,
            app_name=self.app_name,
            session_service=self.session_service
        )

        self.coder_runner = Runner(
            agent=self.coder_agent,
            app_name=self.app_name,
            session_service=self.session_service
        )

    def run_evolution_cycle(self):
        """
        Run the main evolution cycle using ADK agents.
        
        This is the core method that orchestrates the entire evolution process:
        1. Tests the current player in games
        2. Gets analysis of the results
        3. Uses meta agent to coordinate next steps
        4. Routes to appropriate specialist agent for improvements
        5. Repeats for multiple evolution cycles
        
        The process continues for NUM_EVOLUTIONS iterations, with each
        cycle potentially improving the player's capabilities.
        """
        NUM_EVOLUTIONS = 20  # Total number of evolution cycles to run
        
        print("Starting ADK-based evolution cycle...")
        
        # Main evolution loop
        while CreatorAgentADK.current_evolution < NUM_EVOLUTIONS:
            print(f"\n=== EVOLUTION {CreatorAgentADK.current_evolution} ===")
            
            # Step 1: Test the current player implementation
            print("Running game test...")
            game_results = self._run_game_test(short_game=False)
            
            # Step 2: Get detailed analysis from the analyzer agent
            print("Getting analysis...")
            analysis_query = f"""
            ANALYZER OBJECTIVE:
            
            Analyze the results from Evolution {CreatorAgentADK.current_evolution}.
            
            If no syntax errors:
            - Report scores of the {FOO_TARGET_FILENAME} player
            - Analyze game output for interesting findings
            - Emphasize any errors, warnings, or implementation issues
            
            If syntax error:
            - Report error message and line number
            - Identify the problematic code
            
            Keep response concise.
            Start with "After Running The New {FOO_TARGET_FILENAME} Player, Here is my analysis:"
            """
            
            analysis_response = self._call_agent(self.analyzer_runner, analysis_query)
            
            # Step 3: Get coordination decision from meta agent
            print("Getting meta coordination...")
            meta_query = f"""
            Previous Analysis: {analysis_response}
            Game Results: {game_results}
            Current Evolution: {CreatorAgentADK.current_evolution}
            Performance History: {self._get_performance_summary()}
            
            Based on this information, determine next steps.
            """
            
            meta_response = self._call_agent(self.meta_runner, meta_query)
            
            # Step 4: Route to the appropriate specialist agent based on meta decision
            self._route_to_specialist(meta_response)
            
            # Move to next evolution cycle
            CreatorAgentADK.current_evolution += 1
            
            # Save progress after each evolution
            self._save_evolution_state()

        print("\n=== EVOLUTION COMPLETE ===")
        self._save_final_results()

    def _call_agent(self, runner: Runner, query: str) -> str:
        """
        Call an ADK agent and return the response.
        
        This method handles the communication with individual agents,
        managing sessions and processing responses.
        
        Args:
            runner: The Runner instance for the specific agent
            query: The query/request to send to the agent
            
        Returns:
            str: The agent's response
            
        The method:
        1. Creates a unique session for the agent if needed
        2. Sends the query to the agent
        3. Processes the response events
        4. Returns the final response text
        """
        try:
            # Prepare the user query as ADK content
            user_content = types.Content(role='user', parts=[types.Part(text=query)])
            
            final_response = ""
            session_id = f"{self.session_id}_{runner.agent.name}"
            
            # Create session for this specific agent if it doesn't exist
            try:
                self.session_service.create_session(
                    app_name=self.app_name,
                    user_id=self.user_id,
                    session_id=session_id
                )
            except:
                pass  # Session might already exist, which is fine
            
            # Run the agent and collect response events
            events = runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=user_content
            )
            
            # Process events to extract the final response
            for event in events:
                if event.is_final_response() and event.content and event.content.parts:
                    final_response = event.content.parts[0].text
                    break
                    
            return final_response
            
        except Exception as e:
            print(f"Error calling agent {runner.agent.name}: {e}")
            return f"Error: {str(e)}"

    def _route_to_specialist(self, meta_response: str):
        """
        Route to the appropriate specialist agent based on meta response.
        
        This method implements the routing logic that decides which
        specialist agent should handle the current task. It looks for
        keywords in the meta agent's response to determine routing.
        
        Args:
            meta_response: The response from the meta agent containing
                          routing decisions and objectives
        
        Routing Logic:
        - If "ANALYZER" found → Route to analyzer agent
        - If "STRATEGIZER" found → Route to strategizer agent  
        - If "RESEARCHER" found → Route to researcher agent
        - If "CODER" found → Route to coder agent
        - Otherwise → Default to analyzer agent
        
        For each routing decision, it:
        1. Extracts the specific objective for the agent
        2. Calls the appropriate agent with that objective
        3. Prints the response for monitoring
        """
        
        if "ANALYZER" in meta_response:
            print("Routing to Analyzer...")
            objective = self._extract_objective(meta_response)
            response = self._call_agent(self.analyzer_runner, objective)
            print(f"Analyzer Response: {response}")
            
        elif "STRATEGIZER" in meta_response:
            print("Routing to Strategizer...")
            objective = self._extract_objective(meta_response)
            response = self._call_agent(self.strategizer_runner, objective)
            print(f"Strategizer Response: {response}")
            
        elif "RESEARCHER" in meta_response:
            print("Routing to Researcher...")
            objective = self._extract_objective(meta_response)
            response = self._call_agent(self.researcher_runner, objective)
            print(f"Researcher Response: {response}")
            
        elif "CODER" in meta_response:
            print("Routing to Coder...")
            objective = self._extract_objective(meta_response)
            response = self._call_agent(self.coder_runner, objective)
            print(f"Coder Response: {response}")
            
        else:
            print("No clear routing found, defaulting to Analyzer")
            default_objective = "Provide a general analysis of the current state."
            response = self._call_agent(self.analyzer_runner, default_objective)
            print(f"Default Analyzer Response: {response}")

    def _extract_objective(self, meta_response: str) -> str:
        """
        Extract the AGENT OBJECTIVE from meta response.
        
        The meta agent is instructed to include an "AGENT OBJECTIVE:" section
        in its response. This method parses that section to extract the
        specific task for the chosen specialist agent.
        
        Args:
            meta_response: Full response from meta agent
            
        Returns:
            str: Extracted objective text for the specialist agent
            
        Parsing Logic:
        1. Split response into lines
        2. Find line containing "AGENT OBJECTIVE:"
        3. Extract that line and subsequent lines until hitting another section
        4. Return combined objective text
        5. Fallback to full response if parsing fails
        """
        try:
            lines = meta_response.split('\n')
            for i, line in enumerate(lines):
                if "AGENT OBJECTIVE:" in line:
                    # Get this line and subsequent lines until we hit another section
                    objective_lines = [line.replace("AGENT OBJECTIVE:", "").strip()]
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() and not lines[j].startswith('-'):
                            objective_lines.append(lines[j].strip())
                        else:
                            break
                    return '\n'.join(objective_lines)
        except:
            pass  # If parsing fails, fall back to full response
        return meta_response  # Fallback to full response

    def _run_game_test(self, short_game: bool = False) -> str:
        """
        Run the Catanatron game test.
        
        This method executes the actual Catanatron game to test the
        current player implementation. It:
        1. Creates a game run directory
        2. Copies the current player file
        3. Runs the Catanatron command
        4. Captures and processes output
        5. Saves results and updates performance history
        
        Args:
            short_game: If True, runs a shorter test game with reduced timeout
            
        Returns:
            str: Game output and results (limited to MAX_CHARS for readability)
            
        The method handles:
        - Timeout scenarios (expected for long games)
        - Output capture and limiting
        - File organization and storage
        - Performance tracking
        """
        # Create unique run ID based on game type
        if short_game:
            run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S_vg")  # vg = very short game
        else:
            run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S_fg")  # fg = full game
            
        # Set up directory structure for this game run
        game_run_dir = Path(CreatorAgentADK.run_dir) / run_id
        game_run_dir.mkdir(exist_ok=True)
        
        # Copy current player implementation to run directory for record-keeping
        cur_foo_path = game_run_dir / FOO_TARGET_FILENAME
        shutil.copy2(FOO_TARGET_FILE.resolve(), cur_foo_path)
        
        # Limit output size to prevent overwhelming the system
        MAX_CHARS = 20_000
        
        try:
            # Set timeout based on game type (short games = 30s, full games = 4 hours)
            timeout = 30 if short_game else 14400
            
            # Execute the Catanatron game command
            result = subprocess.run(
                shlex.split(FOO_RUN_COMMAND),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False  # Don't raise exception on non-zero exit codes
            )
            
            # Capture and limit output for processing
            stdout_limited = result.stdout[-MAX_CHARS:]
            stderr_limited = result.stderr[-MAX_CHARS:]
            game_results = (stdout_limited + stderr_limited).strip()
            
        except subprocess.TimeoutExpired as e:
            # Handle timeout scenarios (common for full games)
            stdout_output = e.stdout or ""
            stderr_output = e.stderr or ""
            
            # Ensure outputs are strings (handle bytes if necessary)
            if stdout_output and not isinstance(stdout_output, str):
                stdout_output = stdout_output.decode('utf-8', errors='ignore')
            if stderr_output and not isinstance(stderr_output, str):
                stderr_output = stderr_output.decode('utf-8', errors='ignore')
                
            # Limit output and add timeout notice
            stdout_limited = stdout_output[-MAX_CHARS:]
            stderr_limited = stderr_output[-MAX_CHARS:]
            game_results = "Game Ended From Timeout (As Expected).\n\n" + (stdout_limited + stderr_limited).strip()
        
        # Save complete output to file for detailed analysis
        output_file_path = game_run_dir / "game_output.txt"
        with open(output_file_path, "w") as output_file:
            output_file.write(game_results)
        
        # Update performance tracking for full games
        if not short_game:
            self._update_performance_history(game_results, output_file_path, cur_foo_path)
        
        return game_results

    def _update_performance_history(self, game_results: str, output_file_path: Path, cur_foo_path: Path):
        """
        Update the performance history with game results.
        
        This method maintains a JSON file tracking the performance
        of each evolution cycle. It extracts key metrics from game
        results and stores them for analysis by agents.
        
        Args:
            game_results: Raw output from the game execution
            output_file_path: Path to the saved game output file
            cur_foo_path: Path to the player file used in this game
            
        The performance history includes:
        - Win/loss statistics
        - Average scores and turn counts
        - File paths for detailed analysis
        - Timestamps for tracking progress
        
        Note: The current implementation uses simplified metric extraction.
        A production version would parse game results more thoroughly.
        """
        performance_history_path = Path(CreatorAgentADK.run_dir) / "performance_history.json"
        
        # Load existing performance history or create new one
        try:
            with open(performance_history_path, 'r') as f:
                performance_history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            performance_history = {}
        
        # Extract basic statistics from game results
        # Note: This is simplified - production version would have more sophisticated parsing
        wins = 0
        avg_score = 0
        avg_turns = 0
        
        # Simple extraction logic - you might want to enhance this
        if "FooPlayer" in game_results:
            # Basic parsing logic here - could be expanded significantly
            pass
        
        # Create performance record for this evolution
        evolution_key = CreatorAgentADK.current_evolution
        rel_output_file_path = output_file_path.relative_to(Path(CreatorAgentADK.run_dir))
        rel_cur_foo_path = cur_foo_path.relative_to(Path(CreatorAgentADK.run_dir))
        
        performance_history[f"Evolution {evolution_key}"] = {
            "wins": wins,
            "avg_score": avg_score,
            "avg_turns": avg_turns,
            "full_game_log_path": str(rel_output_file_path),
            "json_game_results_path": "None",  # Simplified for now
            "cur_foo_player_path": str(rel_cur_foo_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Save updated performance history
        with open(performance_history_path, 'w') as f:
            json.dump(performance_history, f, indent=2)

    def _get_performance_summary(self) -> str:
        """
        Get a summary of performance history.
        
        This method provides agents with access to historical performance
        data so they can make informed decisions about improvements.
        
        Returns:
            str: JSON-formatted performance history or error message
            
        The performance summary helps agents understand:
        - How the player has evolved over time
        - Which changes were successful or unsuccessful
        - Trends in performance metrics
        """
        performance_history_path = Path(CreatorAgentADK.run_dir) / "performance_history.json"
        
        if not performance_history_path.exists():
            return "No performance history available."
        
        try:
            with open(performance_history_path, 'r') as f:
                performance_history = json.load(f)
            return json.dumps(performance_history, indent=2)
        except:
            return "Error reading performance history."

    def _get_game_query_results(self, query_number: int = -1) -> str:
        """
        Get game query results (placeholder implementation).
        
        This method would provide access to specific game analysis results.
        Currently implemented as a placeholder for future functionality.
        
        Args:
            query_number: Which game query to retrieve (-1 for latest)
            
        Returns:
            str: Game query results (currently placeholder)
        """
        return "Game query results would be retrieved here."

    def _read_game_file(self, path_key: str, num: int = -1) -> str:
        """
        Read a game file from performance history.
        
        This method provides access to files from previous evolution cycles,
        allowing agents to compare implementations or analyze results.
        
        Args:
            path_key: Type of file to read (e.g., "cur_foo_player_path")
            num: Which evolution's file to read (-1 for latest)
            
        Returns:
            str: File content (currently placeholder)
        """
        return f"Game file content for {path_key} (Evolution {num}) would be here."

    def _save_evolution_state(self):
        """
        Save the current evolution state.
        
        This method saves a snapshot of the evolution progress,
        allowing for resumption or analysis of the evolution process.
        
        The state includes:
        - Current evolution number
        - Timestamp
        - Model information
        """
        state_file = Path(CreatorAgentADK.run_dir) / "evolution_state.json"
        state = {
            "current_evolution": CreatorAgentADK.current_evolution,
            "timestamp": datetime.now().isoformat(),
            "llm_name": self.llm_name
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _save_final_results(self):
        """
        Save final results and copy the best player.
        
        This method is called when evolution is complete. It:
        1. Creates a final timestamped copy of the evolved player
        2. Saves it in the run directory for preservation
        3. Provides feedback on completion
        
        The final player file represents the best version evolved
        through all the cycles and can be used for further testing
        or as a starting point for future evolution runs.
        """
        # Create timestamped filename for final player
        dt = datetime.now().strftime("_%Y%m%d_%H%M%S_")
        final_file = Path(CreatorAgentADK.run_dir) / ("final_adk" + dt + FOO_TARGET_FILENAME)
        
        # Copy the evolved player to final location
        shutil.copy2(FOO_TARGET_FILE.resolve(), final_file)
        
        print(f"ADK Evolution complete! Final player saved to: {final_file}")


# ==================== HELPER FUNCTIONS ====================
# Utility functions that support the main class functionality

def read_foo(_: str = "") -> str:
    """
    Return the UTF-8 content of Agent File (≤64 kB).
    
    This is a standalone helper function that provides the same
    functionality as the read_foo method within the class.
    It's used by tool functions that need to read the current player.
    
    Args:
        _: Unused parameter (for compatibility)
        
    Returns:
        str: Current content of the player file
        
    Raises:
        ValueError: If file is too large for processing
    """
    if FOO_TARGET_FILE.stat().st_size > FOO_MAX_BYTES:
        raise ValueError("File too large for the agent")
    return FOO_TARGET_FILE.read_text(encoding="utf-8", errors="ignore") 