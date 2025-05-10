import time
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import base_llm
from base_llm import OpenAILLM, AzureOpenAILLM, MistralLLM, BaseLLM, MistralLLM, AzureOpenAILLM
from typing import List, Dict, Tuple, Any, Optional
import json
import random
from enum import Enum
from io import StringIO
from datetime import datetime
import shutil
from pathlib import Path
import subprocess, shlex


from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_aws import ChatBedrockConverse
from langgraph.graph import MessagesState, START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.tools.tavily_search import TavilySearchResults


# -------- tool call configuration ----------------------------------------------------
LOCAL_SEARCH_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()
FOO_TARGET_FILENAME = "agent_code.py"
PLAYER_CODE = "codeRefiningLLM_player.py"
PLAYER_CODE_PATH = Path(__file__).parent / PLAYER_CODE
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME    # absolute path
FOO_MAX_BYTES   = 64_000                                     # context-friendly cap, R,CR_LLM
FOO_RUN_COMMAND = "catanatron-play --players=AB,CR_LLM --num=1 --config-map=MINI --output=data/ --json"

# -------------------------------------------------------------------------------------

class CreatorAgent():
    """LLM-powered player that uses Claude API to make Catan game decisions."""
    # Class properties
    run_dir = None

    def __init__(self, opponent: str = "AB"):
        # Get API key from environment variable
        # self.llm_name = "gpt-4o"
        # self.llm = AzureChatOpenAI(
        #     model="gpt-4o",
        #     azure_endpoint="https://gpt-amayuelas.openai.azure.com/",
        #     api_version = "2024-12-01-preview"
        # )

        self.llm_name = "claude-3.7"
        self.llm = ChatBedrockConverse(
            aws_access_key_id = os.environ["AWS_ACESS_KEY"],
            aws_secret_access_key = os.environ["AWS_SECRET_KEY"],
            region_name = "us-east-2",
            provider = "anthropic",
            model_id="arn:aws:bedrock:us-east-2:288380904485:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        )

        # self.llm_name = "mistral-large-latest"
        # rate_limiter = InMemoryRateLimiter(
        #     requests_per_second=0.5,    # Adjust based on your API tier
        #     check_every_n_seconds=0.1,
        #     max_bucket_size=10        # Allows for burst handling
        # )
        # self.llm = ChatMistralAI(
        #     model="mistral-large-latest",
        #     temperature=0,
        #     max_retries=2,
        #     rate_limiter=rate_limiter,
        # )

        self.foo_run_command = f"catanatron-play --players={opponent},CR_LLM --num=5 --config-map=MINI --output=data/ --json"

        # Create run directory if it doesn't exist
        if CreatorAgent.run_dir is None:
            agent_dir = os.path.dirname(os.path.abspath(__file__))
            runs_dir = os.path.join(agent_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            CreatorAgent.run_dir = os.path.join(runs_dir, run_id)
            os.makedirs(CreatorAgent.run_dir, exist_ok=True)
            
            # Create a marker file to allow the player to detect the current run directory
            with open(os.path.join(runs_dir, "current_run.txt"), "w") as f:
                f.write(CreatorAgent.run_dir)

        # Copy the Blank FooPlayer to the run directory
        shutil.copy2(                           # ↩ copy with metadata
            (Path(__file__).parent / ("__TEMPLATE__" + FOO_TARGET_FILENAME)).resolve(),  # ../foo_player.py
            FOO_TARGET_FILE.resolve()          # ./foo_player.py
        )

        # Also save a copy of the initial template to the run directory
        shutil.copy2(
            (Path(__file__).parent / ("__TEMPLATE__" + FOO_TARGET_FILENAME)).resolve(),
            Path(CreatorAgent.run_dir) / f"initial_{FOO_TARGET_FILENAME}"
        )
       
        self.config = {
            "recursion_limit": 50, # set recursion limit for graph
            "configurable": {
                "thread_id": "1"
            }
        }
        self.num_memory_messages = 5        # Trim number of messages to keep in memory to limit API usage
        self.react_graph = self.create_langchain_react_graph()
        
    def create_langchain_react_graph(self):
        """Create a react graph for the LLM to use."""
        
        tools = [list_local_files, read_local_file, read_foo, read_player_code, write_foo, self.run_testfoo, web_search_tool_call]
        llm_with_tools = self.llm.bind_tools(tools)
        
        def assistant(state: MessagesState):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
        def trim_messages(state: MessagesState):
            # Delete all but the specified most recent messages
            delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-self.num_memory_messages]]
            return {"messages": delete_messages}


        builder = StateGraph(MessagesState)

        # Define nodes: these do the work
        builder.add_node("trim_messages", trim_messages)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(tools))

        # Define edges: these determine how the control flow moves
        builder.add_edge(START, "trim_messages")
        builder.add_edge("trim_messages", "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        
        return builder.compile(checkpointer=MemorySaver())

    def print_react_graph(self):
        """
        Print the react graph for debugging purposes.
        ONLY WORKS IN .IPYNB NOTEBOOKS
        """
        display(Image(self.react_graph.get_graph(xray=True).draw_mermaid_png()))

    def run_react_graph(self):
        # prompt = (
        #     f"""
        #     You are in charge of creating the code for a Catan Player codeRefiningLLM_player in {FOO_TARGET_FILENAME}. 
            
        #     You Have the Following Tools at Your Disposal:
        #     - list_local_files: List all files in the current directory.
        #     - read_local_file: Read the content of a file in the current directory.
        #     - read_foo: Read the content of {FOO_TARGET_FILENAME}.
        #     - read_player_code: Read the content of the player code to understand how to access game state information and how the game is played.
        #     - write_foo: Modify {FOO_TARGET_FILENAME} with the goal of improving gameplay. Write code based on your research that uses a strategy to gives advice to the LLM that is making the final decisions. Ensure that anything you want to be included in the LLM prompt is outputted as a string from get_prompt_enhancement().
        #     - run_testfoo: Test the content of {FOO_TARGET_FILENAME} in a game.
        #     - web_search_tool_call: Perform a web search to search for strategies and advice on playing and winning a Catan game.

        #     YOUR GOAL: Create a prompt for the Catan Player That Will allow us to play run_testfoo and win a majority of games without crashing. 
        #     Don't stop refining the prompt until codeRefiningLLM_player wins at least 3/5 games against its opponent when calling run_testfoo.

        #     """
        # )
        prompt = (
            f"""
            You are an expert AI developer creating a strategic advisor for a Catan game-playing agent.
            
            ## Your Task
            Enhance the {FOO_TARGET_FILENAME} file to implement strategic game analysis that will guide the LLM player (codeRefiningLLM_player) to victory. The core player code already handles game mechanics and state formatting - your job is to add strategic intelligence.
            
            ## How the System Works
            1. The LLM player (codeRefiningLLM_player.py) handles all game mechanics and formatting game state
            2. The agent_code.py file contains a get_prompt_enhancement() function that returns strategic advice
            3. The advice from get_prompt_enhancement() is added to the beginning of the LLM prompt
            4. The LLM uses this enhanced prompt to make better decisions in the game
            
            ## Your Approach Should Be
            1. First analyze what's happening - read the player code to understand how game state is accessed and how decisions are made
            2. Research effective Catan strategies through web search
            3. Implement those strategies in agent_code.py by:
            - Adding functions that analyze the game state (using the same access methods as in the player code)
            - Evaluating positions, resources, and potential moves
            - Returning strategic advice as a string from get_prompt_enhancement()
            
            ## Tools at Your Disposal
            - list_local_files: List files to understand the codebase structure
            - read_local_file: Read the content of a file in the current directory.
            - read_player_code: Read the player code to understand how game state is accessed
            - read_foo: Read the current agent_code.py implementation
            - write_foo: Update agent_code.py with your enhanced implementation
            - run_testfoo: Test your implementation in actual games
            - web_search_tool_call: Research Catan strategies and tactics (Don't call more than 3 times in a row)
            
            ## Implementation Guidelines
            1. KEEP the get_prompt_enhancement() function signature - this is required
            2. Add helper functions that analyze game state and implement strategies
            3. The advice returned should be clear, concise, and directly applicable
            4. Maintain proper Python style and ensure the code is well-documented
            5. Make sure get_prompt_enhancement() returns a string (not None or other types)
            6. DO NOT use placeholders (like "..." or "some_condition()") - write actual executable code
            
            ## Success Criteria
            Your implementation should enable the LLM player to win at least 3/5 games against its opponent.
            
            ## Process to Follow
            1. Understand: Read the player code to see how it accesses game state and makes decisions
            2. Research: Look up winning Catan strategies using web search
            3. Design: Plan what state analysis and advice would most help the player
            4. Implement: Write code that performs this analysis in agent_code.py
            5. Test: Run games and observe performance
            6. Iterate: Refine your implementation based on game results
            
            Keep improving until the agent consistently wins. Start now by examining the existing code. DO NOT stop until you win 3/5 games when calling the run_testfoo() function.
            """
        )

        try:
            # Create a specific log file for creator agent interactions
            creator_log_path = os.path.join(CreatorAgent.run_dir, f"creator_agent_log.txt")

            # Run Through The Graph
            initial_input = {"messages": prompt}
            with open(creator_log_path, "a") as log_file:
                log_file.write(f"=== Creator Agent Session Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
                
                # Run the graph until the first interruption
                for event in self.react_graph.stream(initial_input, self.config, stream_mode="values"):
                    msg = event['messages'][-1]
                    msg.pretty_print()
                    log_file.write(f"{msg.pretty_repr()}\n\n")
                    
                log_file.write(f"\n=== Creator Agent Session Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

            print("✅  graph finished")

            # Copy Result File to the new directory
            shutil.copy2(                           
                (FOO_TARGET_FILE).resolve(),
                (Path(CreatorAgent.run_dir) / FOO_TARGET_FILENAME)
            )

        except Exception as e:
            print(f"Error calling LLM: {e}")
        return None
    
    def run_testfoo(self, _: str = "") -> str:
        """Run one Catanatron match using the agent code and save logs to run folder."""
        # Create a timestamped folder for this specific game run
        run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S")
        game_run_dir = Path(CreatorAgent.run_dir) / run_id
        game_run_dir.mkdir(exist_ok=True)
        
        # Save the current code used for this game
        code_copy_path = game_run_dir / f"code_used_{FOO_TARGET_FILENAME}"
        shutil.copy2(FOO_TARGET_FILE, code_copy_path)
        
        # Create/update env variable or file to tell the player which run directory to use
        # This ensures the Player writes logs to the same top-level directory
        os.environ["CATAN_CURRENT_RUN_DIR"] = str(CreatorAgent.run_dir)
        
        # Run the game
        result = subprocess.run(
            shlex.split(self.foo_run_command),
            capture_output=True,
            text=True,
            timeout=14400,
            check=False
        )
        
        # Save the output to a log file in the game run directory
        output_log = game_run_dir / "game_output.txt"
        with open(output_log, "w", encoding="utf-8") as f:
            f.write((result.stdout + result.stderr).strip())
    
        # Return the results as before
        return (result.stdout + result.stderr).strip()

def list_local_files(_: str = "") -> str:
    """Return all files beneath BASE_DIR, one per line."""
    return "\n".join(
        str(p.relative_to(LOCAL_SEARCH_BASE_DIR))
        for p in LOCAL_SEARCH_BASE_DIR.glob("**/*")
        if p.is_file() and p.suffix in {".py", ".txt", ".md"}
    )

def read_local_file(rel_path: str) -> str:
    """
    Return the text content of rel_path if it’s inside BASE_DIR.
    Args:
        rel_path: Relative path to the file to read.
    """
    if rel_path == FOO_TARGET_FILENAME:
        return read_foo()
    candidate = (LOCAL_SEARCH_BASE_DIR / rel_path).resolve()
    if not str(candidate).startswith(str(LOCAL_SEARCH_BASE_DIR)) or not candidate.is_file():
        raise ValueError("Access denied or not a file")
    if candidate.stat().st_size > 64_000:
        raise ValueError("File too large")
    return candidate.read_text(encoding="utf-8", errors="ignore")

def read_player_code(_: str = "") -> str:
    """Return the UTF-8 content of Agent File (≤64 kB)."""
    if PLAYER_CODE_PATH.stat().st_size > FOO_MAX_BYTES:
        raise ValueError("File too large for the agent")
    return PLAYER_CODE_PATH.read_text(encoding="utf-8", errors="ignore")  # pathlib API :contentReference[oaicite:2]{index=2}

def read_foo(_: str = "") -> str:
    """Return the UTF-8 content of Agent File (≤64 kB)."""
    if FOO_TARGET_FILE.stat().st_size > FOO_MAX_BYTES:
        raise ValueError("File too large for the agent")
    return FOO_TARGET_FILE.read_text(encoding="utf-8", errors="ignore")  # pathlib API :contentReference[oaicite:2]{index=2}


def write_foo(new_text: str) -> str:
    """Overwrite Agent File with new_text (UTF-8)."""
    if len(new_text.encode()) > FOO_MAX_BYTES:
        raise ValueError("Refusing to write >64 kB")
    FOO_TARGET_FILE.write_text(new_text, encoding="utf-8")                 # pathlib write_text :contentReference[oaicite:3]{index=3}
    return f"{FOO_TARGET_FILENAME} updated successfully"

def web_search_tool_call(query: str) -> str:
    """Perform a web search using the Tavily API.

    Args:
        query: The search query string.

    Returns:
        The search result as a string.
    """
    # Simulate a web search
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return formatted_search_docs
