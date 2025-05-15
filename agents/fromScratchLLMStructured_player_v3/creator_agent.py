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
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage, ToolMessage, BaseMessage
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_aws import ChatBedrockConverse
from langgraph.errors import GraphRecursionError


from typing_extensions import TypedDict
from typing_extensions import TypedDict

# import warnings
# from langchain.warnings import LangChainDeprecationWarning   # same class they raise
# warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)



# -------- tool call configuration ----------------------------------------------------
LOCAL_CATANATRON_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()
FOO_TARGET_FILENAME = "foo_player.py"
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME    # absolute path
FOO_MAX_BYTES   = 64_000                                     # context-friendly cap
# Set winning points to 5 for quicker game
FOO_RUN_COMMAND = "catanatron-play --players=AB,FOO_LLM_S3  --num=10 --config-map=MINI  --config-vps-to-win=10"
RUN_TEST_FOO_HAPPENED = False # Used to keep track of whether the testfoo tool has been called
# -------------------------------------------------------------------------------------


LLM_NAME = "o1"
LLM = AzureChatOpenAI(
    model="o1",
    azure_endpoint="https://gpt-amayuelas.openai.azure.com/",
    api_version = "2024-12-01-preview"
)
from botocore.config import Config
# import boto3
# import os

config = Config(read_timeout=3600)
# bedrock_client = boto3.client(
#     service_name='bedrock-runtime',
#     region_name='us-east-2',
#     config=config
# )

thinking_params= {
    "thinking": {
        "type": "disabled",
        #"budget_tokens": 2000
    }
}


#bedrock_client = client(service_name='bedrock-runtime', region_name='us-east-2', config=config)
# LLM_NAME = "claude-3.7"
# LLM = ChatBedrockConverse(
#     aws_access_key_id = os.environ["AWS_ACESS_KEY"],
#     aws_secret_access_key = os.environ["AWS_SECRET_KEY"],
#     region_name = "us-east-2",
#     provider = "anthropic",
#     model_id="arn:aws:bedrock:us-east-2:288380904485:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
#     additional_model_request_fields=thinking_params,
#     config=config

# )

os.environ["LANGCHAIN_TRACING_V2"] = "false"


# LLM_NAME = "mistral-large-latest"
# rate_limiter = InMemoryRateLimiter(
#     requests_per_second=1,    # Adjust based on your API tier
#     check_every_n_seconds=0.1,
# )
# LLM = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=0,
#     max_retries=10,
#     rate_limiter=rate_limiter,
# )
class CreatorAgent():
    """LLM-powered player that uses Claude API to make Catan game decisions."""
    # Class properties
    run_dir = None
    current_evolution = 0

    def __init__(self):
        # Get API key from environment variable

        # Create run directory if it doesn't exist
        if CreatorAgent.run_dir is None:
            agent_dir = os.path.dirname(os.path.abspath(__file__))
            runs_dir = os.path.join(agent_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            run_id = datetime.now().strftime("creator_%Y%m%d_%H%M%S")
            CreatorAgent.run_dir = os.path.join(runs_dir, run_id)
            os.makedirs(CreatorAgent.run_dir, exist_ok=True)

        #Copy the Blank FooPlayer to the run directory
        shutil.copy2(                           # ↩ copy with metadata
            (Path(__file__).parent / ("__TEMPLATE__" + FOO_TARGET_FILENAME)).resolve(),  # ../foo_player.py
            FOO_TARGET_FILE.resolve()          # ./foo_player.py
        )

        self.config = {
            "recursion_limit": 150, # set recursion limit for graph
            # "configurable": {
            #     "thread_id": "1"
            # }
        }
        self.react_graph = self.create_langchain_react_graph()

    def create_langchain_react_graph(self):
        """Create a react graph for the LLM to use."""
        

        class CreatorGraphState(TypedDict):
            # full_results: HumanMessage # Last results of running the game
            messages_buffer: List[AnyMessage] # Messages to be passed to the LLM
            performance_history: HumanMessage # Performance history of the player

            full_results: HumanMessage # Last results of running the game
            analyzer_results: HumanMessage # Last results of running the game


            evolve_counter: int         # Counter for the number of evolutions

        multi_agent_prompt = f"""You are apart of a multi-agent system that is working to evolve the code in {FOO_TARGET_FILENAME} to become the best player in the Catanatron Minigame. Get the highest score for the player by class in foo_player.py. Your performance history on this trial is saved in the json\n\tYour specific role is the:"""

        #tools = [add, multiply, divide]
        DEFAULT_EVOLVE_COUNTER = 10
        DEFAULT_VALIDATOR_COUNTER = 2
        MAX_MESSAGES_TOOL_CALLING = 10

        analyzer_continue_key = "CONTINUE_EVOLVING"
        analyzer_stop_key = "STOP_EVOLVING"
        

        def init_node(state: CreatorGraphState):
            """
            Initialize the state of the graph
            """
            #print("In Init Node")
            # Create the initial state of the graph
            if "evolve_counter" not in state:
                evolve_counter = DEFAULT_EVOLVE_COUNTER
            else:
                evolve_counter = state["evolve_counter"]


            evolve_counter: int         # Counter for the number of evolutions
            return {
                "evolve_counter": evolve_counter,
                "performance_history": HumanMessage(content=""),
                "messages_buffer": [],
            }

        def run_player_node(state: CreatorGraphState):
            """
            Runs Catanatron with the current Code
            """
            #print("In Run Player Node")
            
            evolve_counter = state["evolve_counter"] - 1

            # Generate a test results (later will be running the game)
            game_results = run_testfoo(short_game=False)
            #output = llm.invoke([HumanMessage(content="Create me a complicated algebra math problem, without a solution. Return only the problem...NO COMMENTARY!!")])
            full_results = HumanMessage(content=f"GAME RESULTS:\n\n{game_results}")
            state["messages_buffer"].append(full_results)
            # Clear all past messages
            return {
                "full_results": full_results,
                "evolve_counter": evolve_counter,
            }

        def summarizer_node(state: CreatorGraphState):
    
            # print("In Summarizer Node")
            # sys_msg = SystemMessage(content=
            #     f"""You are tasked with summarizing an Multi-Agent Workflow with the steps Full_Results, Analysis, Strategy, Solution, Code Additions, and Validation
            #     This workflow will iterate until the code is correct and the game is won. 

            #     Above, you have new Full_Results, Analysis, Solution, Code Additions Messages for a new step in the Multi-Agent Workflow
            #     Your Summary should look like the following:

            #     <Short High Level Description>
            #         Game Results Summary: <summary of the game results>
            #         Analysis: <summary of the analysis>
            #         Strategy: <summary of the strategy>
            #         Solution: <summary of the solution>
            #         Code Additions: <summary of the code additions>
            #         Validation: <summary of the validation>
                

            #     Do not make anything up. Only write what you are given, or nothing at all

            #     IMPORTANT: Only include the summary in the output, no other commentary or information
            #     Make sure to keep the summary concise and to the point

            #     """
            # )
            
            # state_msgs = state["messages_buffer"] + [HumanMessage(content="Summarize the above messages")]
            # tools = []
            # output = tool_calling_state_graph(sys_msg, state_msgs, tools)
            # summary = output["messages"][-1].content
            
            performance_history = {}
            # Update the performance history JSON with the summary
            try:
                # Load the performance history
                performance_history_path = Path(CreatorAgent.run_dir) / "performance_history.json"
                if performance_history_path.exists():
                    with open(performance_history_path, 'r') as f:
                        performance_history = json.load(f)
                    
                    # Find the entry for the current evolution
                    # The current evolution key should be for the current iteration
                    evolution_key = f"Evolution {CreatorAgent.current_evolution - 1}"  # -1 because it's incremented after running
                    
                    if evolution_key in performance_history:
                        # Update the summary field
                        #performance_history[evolution_key]["summary"] = summary
                        
                        # Write the updated performance history back to the file
                        with open(performance_history_path, 'w') as f:
                            json.dump(performance_history, f, indent=2)
                            
                        print(f"Updated performance history with summary for {evolution_key}")
                    else:
                        print(f"Warning: Could not find {evolution_key} in performance history")
            except Exception as e:
                print(f"Error updating performance history with summary: {e}")
                import traceback
                traceback.print_exc()
            return {
                "performance_history": HumanMessage(content=json.dumps(performance_history)),
            }
        
        def analyzer_node(state: CreatorGraphState):
            #print("In Analyzer Node")

            
            sys_msg = SystemMessage(
                content=f"""
                    You are **Evolver**, the code-editing agent in a closed loop that trains
                    FooPlayer (`{FOO_TARGET_FILENAME}`) to dominate the Catanatron MINI match.
                    After you finish, a new game will run automatically—so your #1 goal each
                    turn is to get the facts you need, then call **write_foo(new_code)**.

                    ─────────────────  Resources  ─────────────────
                    • Tools you may invoke this turn (max 4 total):
                        - ask_question(query)      ↳ ask a helper LLM *one* concise question
                        - read_foo()               ↳ read current FooPlayer source output is string
                        - write_foo(string new_text)      ↳ replace FooPlayer (≤64 kB) input is string
                    • Performance history JSON:
                    {state['performance_history'].content}
                    ───────────────────────────────────────────────


                    ask_question(query) will be used to ask a question to helper LLM
                        - What can be answered
                            - Questions about the result of the last game, or any game in performance history
                            - Questions about the catanatron api and the player function
                            - Questions requiring a web search
                            - Questions about syntax
                            - Questions about the current player
                            - Request to view the output of the last player (see print() statemmnts)

                        - Recommendation
                            - Ask SPECIFIC questions



                    Notes on write_foo()
                        - Make sure to make iterative improvements. 
                        - It is better to be able to see results, instead of continuously causing exceptions
                        - Make sure to always keep the imports, and the decide() function  in the file. (This is how next turn is done)

                    
                    - When getting your previous game results, I recommend calling ask_question to get more information about the game. (Like Output)

                    - Try and use ask_question at least once per turn to get insightful information, instead of just updateing the code

                    - After You call write_foo(new_code), Return immediately a quick summary of what you did to initiate the test game

                    DO NOT RESPOND TO THIS PROMPT...JUST CALL THE write_foo(new_code) TOOL CALL to Write to the New Player
                    """

            )
            msgs = state["messages_buffer"]
            tools = [write_foo, read_foo, ask_question]
            output = tool_calling_state_graph(sys_msg, msgs, tools)
            #analysis = HumanMessage(content=output["messages"][-1].content)

            existing_msg_ids = {id(msg) for msg in msgs}
    
            # Add only new messages that aren't already in the buffer
            for msg in output["messages"]:
                if id(msg) not in existing_msg_ids:
                    state["messages_buffer"].append(msg)
                    existing_msg_ids.add(id(msg))
            
            return {"analyzer_results": output["messages"][-1].content}
        
        def continue_evolving(state: CreatorGraphState):
            """
            Conditional edge for Analyzer
            """
            print("In Conditional Edge Analyzer")
            
            # Get the content of the validation message
            
            # Check for the presence of our defined result strings (Because analyzer node decrements the counter)
            if state["evolve_counter"] <= -1:
                print("Evolve counter is 0 - ending workflow")
                return END
                
            return "analyzer"

        def construct_graph():
            graph = StateGraph(CreatorGraphState)
            graph.add_node("init", init_node)
            graph.add_node("run_player", run_player_node)
            graph.add_node("analyzer", analyzer_node)
            graph.add_node("summarizer",summarizer_node)
           

            graph.add_edge(START, "init")
            graph.add_edge("init", "run_player") 
            graph.add_edge("run_player", "summarizer")
            graph.add_conditional_edges(
                "summarizer",
                continue_evolving,
                {END, "analyzer"}
            )   
            graph.add_edge("analyzer", "run_player")
            

            return graph.compile()
    
        return construct_graph()

    def print_react_graph(self):
        """
        Print the react graph for debugging purposes.
        ONLY WORKS IN .IPYNB NOTEBOOKS
        """
        display(Image(self.react_graph.get_graph(xray=True).draw_mermaid_png()))

    def run_react_graph(self):
        
        try:

            log_path = os.path.join(CreatorAgent.run_dir, f"llm_log_{LLM_NAME}.txt")

            with open(log_path, "a") as log_file:                # Run the graph until the first interruption
                for step in self.react_graph.stream({}, self.config, stream_mode="updates"):
                    #print(step)
                    #log_file.write(f"Step: {step.}\n")
                    for node, update in step.items():
                        
                        print(f"In Node: {node}")
                        
                        # Simplified Messages code
                        key_types = ["analysis", "strategy", "solution", "code_additions", "validation", "performance_history", "full_results", "test_results"]
                        for key in key_types:
                            if key in update:
                                msg = update[key]
                                msg.pretty_print()
                                log_file.write(msg.pretty_repr())

                        if "tool_calling_messages" in update:
                            count = 0
                            for msg in update["tool_calling_messages"]:
                                #print(msg)
                                #msg.pretty_print()
                                if isinstance(msg, ToolMessage):
                                    print("Tool Message: ", msg.name)
                                count += 1
                                log_file.write((msg).pretty_repr())
                            print(f"Number of Tool Calling Messages: {count}")
                       
                        if "evolve_counter" in update:
                            print("ENVOLVE COUNTER: ", update["evolve_counter"])
                            log_file.write(f"Evolve Counter: {update['evolve_counter']}\n")
                        if "validator_counter" in update:
                            print("VALIDATOR COUNTER: ", update["validator_counter"])
                            log_file.write(f"Validator Counter: {update['validator_counter']}\n")


            print("✅  graph finished")

            # Copy Result File to the new directory
            dt = datetime.now().strftime("_%Y%m%d_%H%M%S_")

            shutil.copy2(                           
                (FOO_TARGET_FILE).resolve(),
                (Path(CreatorAgent.run_dir) / ("final" + dt + FOO_TARGET_FILENAME))
            )

        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import traceback
            traceback.print_exc()
        return None


def list_catanatron_files(_: str = "") -> str:
    """Return all files beneath BASE_DIR, one per line."""
    return "\n".join(
        str(p.relative_to(LOCAL_CATANATRON_BASE_DIR))
        for p in LOCAL_CATANATRON_BASE_DIR.glob("**/*")
        if p.is_file() and p.suffix in {".py", ".txt", ".md"}
    )

def read_local_file(rel_path: str) -> str:
    """
    Return the text content of rel_path if it's inside BASE_DIR.
    Args:
        rel_path: Relative path to the file to read.
    """
    # Path Requested is from Agent File
    if rel_path == FOO_TARGET_FILENAME:
        return read_foo()
    
    # Path is from Catanatron base directory
    if rel_path.startswith("catanatron/"):
        candidate = (LOCAL_CATANATRON_BASE_DIR / rel_path.replace("catanatron/", "")).resolve()
        if not str(candidate).startswith(str(LOCAL_CATANATRON_BASE_DIR)) or not candidate.is_file():
            raise ValueError("Access denied or not a file")
        if candidate.stat().st_size > 64_000:
            raise ValueError("File too large")
        return candidate.read_text(encoding="utf-8", errors="ignore")
    
    # Handle paths relative to run_dir (used in performance history)
    # This includes both paths starting with "runs/" and paths that don't start with "/"
    run_path = Path(CreatorAgent.run_dir) / rel_path
    if run_path.exists() and run_path.is_file():
        if run_path.stat().st_size > 64_000:
            raise ValueError("File too large")
        return run_path.read_text(encoding="utf-8", errors="ignore")
    
    # Check if path is relative to Catanatron directory
    candidate = (LOCAL_CATANATRON_BASE_DIR / rel_path).resolve()
    if not str(candidate).startswith(str(LOCAL_CATANATRON_BASE_DIR)) or not candidate.is_file():
        raise ValueError(f"Access denied or file not found: {rel_path}")
    if candidate.stat().st_size > 64_000:
        raise ValueError("File too large")
    return candidate.read_text(encoding="utf-8", errors="ignore")

def read_foo(_: str = "") -> str:
    """
    Return the UTF-8 content of Agent File (≤64 kB).
    """
    if FOO_TARGET_FILE.stat().st_size > FOO_MAX_BYTES:
        raise ValueError("File too large for the agent")
    return FOO_TARGET_FILE.read_text(encoding="utf-8", errors="ignore")  # pathlib API :contentReference[oaicite:2]{index=2}

def write_foo(new_text: str) -> str:
    """
    Overwrite Agent File with new_text (UTF-8).
    """
    if len(new_text.encode()) > FOO_MAX_BYTES:
        raise ValueError("Refusing to write >64 kB")
    FOO_TARGET_FILE.write_text(new_text, encoding="utf-8")                 # pathlib write_text :contentReference[oaicite:3]{index=3}
    
    # # Copy Result File to the new directory
    # dt = datetime.now().strftime("%Y%m%d_%H%M%S_")

    # shutil.copy2(                           
    #     (FOO_TARGET_FILE).resolve(),
    #     (Path(CreatorAgent.run_dir) / (dt + FOO_TARGET_FILENAME))
    # )

    return f"{FOO_TARGET_FILENAME} updated successfully"

def run_testfoo(short_game: bool = False) -> str:
    """
    Run one Catanatron match (R vs Agent File) and return raw CLI output.
    Input: short_game (bool): If True, run a short game with a 30 second timeout.
    """

    if short_game:
        run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S_vg")
    else:
        run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S_fg")
    game_run_dir = Path(CreatorAgent.run_dir) / run_id
    game_run_dir.mkdir(exist_ok=True)
    
    cur_foo_path = game_run_dir / FOO_TARGET_FILENAME
    # Save the current prompt used for this game
    shutil.copy2(
        FOO_TARGET_FILE.resolve(),
        cur_foo_path
    )
        
    MAX_CHARS = 20_000                      

    try:
        if short_game:
            result = subprocess.run(
                shlex.split(FOO_RUN_COMMAND),
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
        else:
            result = subprocess.run(
                shlex.split(FOO_RUN_COMMAND),
                capture_output=True,
                text=True,
                timeout=14400,
                check=False
            )
        stdout_limited  = result.stdout[-MAX_CHARS:]
        stderr_limited  = result.stderr[-MAX_CHARS:]
        game_results = (stdout_limited + stderr_limited).strip()
    except subprocess.TimeoutExpired as e:
        # Handle timeout case
        stdout_output = e.stdout or ""
        stderr_output = e.stderr or ""
        if stdout_output and not isinstance(stdout_output, str):
            stdout_output = stdout_output.decode('utf-8', errors='ignore')
        if stderr_output and not isinstance(stderr_output, str):
            stderr_output = stderr_output.decode('utf-8', errors='ignore')
        stdout_limited  = stdout_output[-MAX_CHARS:]
        stderr_limited  = stderr_output[-MAX_CHARS:]
        game_results = "Game Ended From Timeout (As Expected).\n\n" + (stdout_limited + stderr_limited).strip()
    

    # Save the output to a log file in the game run directory
    output_file_path = game_run_dir / "game_output.txt"
    
    with open(output_file_path, "w") as output_file:
        output_file.write(game_results)


    # Search for the JSON results file path in the output
    json_path = None
    import re
    path_match = re.search(r'results_file_path:([^\s]+)', game_results)
    if path_match:
        # Extract the complete path
        json_path = path_match.group(1).strip()

    # Load in the most recent JSON File Game Rur
    json_content = {}
    json_copy_path = "None"
    # If we found a JSON file path, copy it and load its contents
    if json_path and Path(json_path).exists():
        # Copy the JSON file to our game run directory
        json_filename = Path(json_path).name
        json_copy_path = game_run_dir / json_filename
        shutil.copy2(json_path, json_copy_path)
        
        # Load the JSON content
        try:
            with open(json_path, 'r') as f:
                json_content = json.load(f)
        except json.JSONDecodeError:
            json_content = {"error": "Failed to parse JSON file"}


    # Update performance_history.json
    performance_history_path = Path(CreatorAgent.run_dir) / "performance_history.json"
    try:
        # Load existing performance history
        with open(performance_history_path, 'r') as f:
            performance_history = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        performance_history = {}
    
    # Extract relevant data from json_content
    wins = 0
    avg_score = 0
    avg_turns = 0
    
    try:
        # Extract data from the JSON structure
        if "Player Summary" in json_content:
            our_player = "FooPlayer"
            for player, stats in json_content["Player Summary"].items():
                if player.startswith(our_player):  # Check if player key starts with our player's name
                    if "WINS" in stats:
                        wins = stats["WINS"]
                    if "AVG VP" in stats:
                        avg_score = stats["AVG VP"]
                    
        if "Game Summary" in json_content:
            if "AVG TURNS" in json_content["Game Summary"]:
                avg_turns = json_content["Game Summary"]["AVG TURNS"]
    except Exception as e:
        print(f"Error extracting stats from JSON: {e}")
        
    # Update performance history only on long game runs
    if not short_game:
        # Create or update the entry for this evolution
        evolution_key = CreatorAgent.current_evolution
        CreatorAgent.current_evolution += 1
        
        # Convert paths to be relative to run_dir
        rel_output_file_path = output_file_path.relative_to(Path(CreatorAgent.run_dir))
        rel_cur_foo_path = cur_foo_path.relative_to(Path(CreatorAgent.run_dir))
        # Handle the case where json_copy_path is a string "None"
        rel_json_copy_path = "None"
        if json_copy_path != "None" and isinstance(json_copy_path, Path):
            rel_json_copy_path = json_copy_path.relative_to(Path(CreatorAgent.run_dir))
        
        performance_history[f"Evolution {evolution_key}"] = {
            "wins": wins,
            "avg_score": avg_score,
            "avg_turns": avg_turns,
            "full_game_log_path": str(rel_output_file_path),
            "json_game_results_path": str(rel_json_copy_path),
            "cur_foo_player_path": str(rel_cur_foo_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": "Not yet summarized"
        }
        
        # Write updated performance history
        with open(performance_history_path, 'w') as f:
            json.dump(performance_history, f, indent=2)
        
    if json_content:
        return json.dumps(json_content, indent=2)
    else:
        # If we didn't find a JSON file, return a limited version of the game output
        # MAX_CHARS = 5_000
        # stdout_limited = result.stdout[-MAX_CHARS:]
        # stderr_limited = result.stderr[-MAX_CHARS:]
        return game_results
    # Extract the score from the game results

    # Create a folder in the Creator.run_dir with EvolveCounter#_FooScore#

    # Inside the folder, 
    #   place the game_results.txt file with the game results
    #   copy the FOO_TARGET_FILE as foo_player.py

    # Add a file with the stdout and stderr called catanatron_output.txt
    # output_file_path = latest_run_folder / "catanatron_output.txt"
    # with open(output_file_path, "w") as output_file:
    #     output_file.write(game_results)
        
    #print(game_results)
        # limit the output to a certain number of characters

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

def read_full_performance_history(_: str = "") -> str:
    """Return the content of performance_history.json as a string (≤16 kB)."""
    performance_history_path = Path(CreatorAgent.run_dir) / "performance_history.json"

    if not performance_history_path.exists():
        return "Performance history file does not exist."
    
    if performance_history_path.stat().st_size > 64_000:
        return "Performance history file is too large (>16 KB). Consider truncating or summarizing it."
    
    with open(performance_history_path, 'r') as f:
        performance_history = json.load(f)
        return json.dumps(performance_history, indent=2)
    
def ask_question(query: str) -> str:
    """Ask a question to the LLM and return the response."""
    # Simulate a question to the LLM
    sys_msg = SystemMessage(content = """
                            
Role: **Knowledge-Fetcher**  
Answer *only* the user's question as briefly and concretely as possible to
support code evolution for FooPlayer.

Rules:
 - Do not make up answers. 
 - When being asked about syntax errors, try and find the solution in the catanatron source code
    - If you don't know the answer, say "I don't know the answer because... Here is what I can do instead...                       

                                                
Tools available: Use As Many Tool Calls as needed
  1. list_catanatron_files: 
    - lists out all the game files in catanatron, so that they can be opened
    - Utilze when you have a syntax error and need to find the code of origin
  2. read_full_performance_history: 
    - read the performance history of the player
    - Utilze when you need to see how the player has performed in the past (contains games, scores, outputs, and logs)
  3. read_local_file(path)        
    - open and read a file in the catanatron directory, or in the current run directory
    - use this after list_catanatron_files or read_full_performance_history to find the file you want to open
  4. read_foo                     
    - read current FooPlayer code
  5. web_search_tool_call(query)  - external info (Catan rules, heuristics)
    - search the web for information (If there is questions about the API, use list_catanatron_files or read_local_file first)
    - can be used to find information about the game, documentation, or strategies


Your response format **must be**:

ANSWER:
  <succinct explanation, ≤ 150 words, bulleted if helpful>

CITED_SOURCES:
  <relative paths or “web” with very short labels>

NO extra commentary. NO chain-of-thought.  
   
"""
    )
    query_msg = HumanMessage(content=query)
    tools = [list_catanatron_files, read_full_performance_history, read_local_file, read_foo, web_search_tool_call]

    output = tool_calling_state_graph(sys_msg, [query_msg], tools, 15)

    return output["messages"][-1].content

def _is_non_empty(msg: BaseMessage) -> bool:
    """
    Return True if the message should stay in history.
    Handles str, list, None, and messages with no 'content' attr.
    """
    if not hasattr(msg, "content"):           # tool_result or custom types
        return True

    content = msg.content
    if content is None:                       # explicit null
        return False

    if isinstance(content, str):
        return bool(content.strip())          # keep non‑blank strings

    if isinstance(content, (list, tuple)):    # content blocks
        return len(content) > 0               # keep if any block exists

    # Fallback: keep anything we don't explicitly reject
    return True


def tool_calling_state_graph(sys_msg: SystemMessage, msgs: list[AnyMessage], tools, MAX_MESSAGES_TOOL_CALLING: int = 15) -> dict:

    # Filter out empty messages
    #msgs = [m for m in msgs if _is_non_empty(m)]

    # Bind Tools to the LLM
    #llm_with_tools = self.llm.bind_tools(tools, parallel_tool_calls=False)
    llm_with_tools = LLM.bind_tools(tools)

    def assistant(sub_state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + sub_state["messages"])]}
    

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    #builder.add_node("final_assistant", final_assistant)

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    #builder.add_conditional_edges("tools", check_num_messages, "assistant", "final_assistant")
    builder.add_edge("tools", "assistant")
    react_graph = builder.compile()
    
    
    #messages = react_graph.invoke({"messages": msgs})
    config = {"recursion_limit": MAX_MESSAGES_TOOL_CALLING*2}

    # last_event = None
    # try:
    #     for event in react_graph.stream({"messages": msgs}, config=config, stream_mode="values"):
    #         msg = event['messages'][-1]
    #         msg.pretty_print()
    #         print("\n")
    #         last_event = event
    #     return last_event
    # except GraphRecursionError as e:
    #     print(f"Recursion limit reached {MAX_MESSAGES_TOOL_CALLING}: {e}")

    
    # # If End Early, Must Still Get Output
    # last_message = last_event["messages"][-1]

    # # If is a last AI Message and requests tools calls, delete it, if does not request tool calls, return it
    # if isinstance(last_message, AIMessage):
    #     if not last_message.tool_calls:
    #         return last_event
    #     else:
    #         # If the last message is an AI message with a tool call, remove it and add another AI message
    #         last_event["messages"] = last_event["messages"][:-1]
    #         last_event["messages"].append(AIMessage(content="OOPS! I made a mistake, I used too many tool calls"))

    # # If last is a tool call message, add AI message for mistake        
    # elif isinstance(last_message, ToolMessage):
    #     last_event["messages"].append(AIMessage(content="OOPS! I made a mistake, I used too many tool calls"))

    # # Add Human Message with instructionrs
    # last_event["messages"].append(HumanMessage(content= """
    #     YOU CAN NO LONGER USE TOOLS! YOU MUST USE WHAT KNOWLEDGE YOU HAVE TO ANSWER THE SYSTEM PROMPT"""
    # ))

    # # Combine the system message with the existing messages
    # input_msg = [sys_msg] + last_event["messages"]

    # # Invoke the LLM with the adjusted message sequence
    # assistant_response = llm_with_tools.invoke(input_msg)

    # # Append the assistant's response to the message history
    # last_event["messages"].append(assistant_response)

    
    # for m in messages['messages']:
    #     m.pretty_print()
    for event in react_graph.stream({"messages": msgs}, stream_mode="values"):
            msg = event['messages'][-1]
            msg.pretty_print()
            print("\n")
            last_event = event

    return last_event
    