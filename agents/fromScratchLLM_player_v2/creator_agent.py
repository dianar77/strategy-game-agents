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
from langgraph.graph import MessagesState, START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.rate_limiters import InMemoryRateLimiter


# -------- tool call configuration ----------------------------------------------------
LOCAL_SEARCH_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()
FOO_TARGET_FILENAME = "foo_player.py"
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME    # absolute path
FOO_MAX_BYTES   = 64_000                                     # context-friendly cap
# Set winning points to 5 for quicker game
FOO_RUN_COMMAND = "catanatron-play --players=AB,R,FOO_LLM_V2  --num=1 --config-map=MINI  --config-vps-to-win=5"
RUN_TEST_FOO_HAPPENED = False # Used to keep track of whether the testfoo tool has been called
# -------------------------------------------------------------------------------------

class CreatorAgent():
    """LLM-powered player that uses Claude API to make Catan game decisions."""
    # Class properties
    run_dir = None

    def __init__(self):
        # Get API key from environment variable
        self.llm_name = "gpt-4o"
        self.llm = AzureChatOpenAI(
            model="gpt-4o",
            azure_endpoint="https://gpt-amayuelas.openai.azure.com/",
            api_version = "2024-12-01-preview"
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


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
        self.memory_config = {
            "recursion_limit": 100, # set recursion limit for graph
            "configurable": {
                "thread_id": "1"
            }
        }
        #self.memory_config = {"configurable": {"thread_id": "1"}}
        self.num_memory_messages = 10        # Trim number of messages to keep in memory to limit API usage
        self.react_graph = self.create_langchain_react_graph()

        
    def create_langchain_react_graph(self):
        """Create a react graph for the LLM to use."""
        
        tools = [
            list_local_files,
            read_local_file,
            read_foo,
            write_foo,
            run_testfoo,
            web_search_tool_call,
            view_last_game_llm_query,
            view_last_game_results
        ]

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
        builder.add_edge(START, "assistant")
        #builder.add_edge("trim_messages", "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        builder.add_edge("tools", "trim_messages")
        builder.add_edge("trim_messages", "assistant")
        
        return builder.compile(checkpointer=MemorySaver())

    def print_react_graph(self):
        """
        Print the react graph for debugging purposes.
        ONLY WORKS IN .IPYNB NOTEBOOKS
        """
        display(Image(self.react_graph.get_graph(xray=True).draw_mermaid_png()))

    def run_react_graph(self):
        prompt = (
            f"""
            You are in charge of creating the code for a Catan Player in {FOO_TARGET_FILENAME}. 
            
            You Have the Following Tools at Your Disposal:
            - list_local_files: List all files in the current directory.
            - read_local_file: Read the content of a file in the current directory.
            - read_foo: Read the content of {FOO_TARGET_FILENAME}.
            - write_foo: Write the content of {FOO_TARGET_FILENAME}. (Make sure to keep imports) Note: print() commands will be visible in view_last_game_results
            - run_testfoo: Test the content of {FOO_TARGET_FILENAME} in a game.
            - web_search_tool_call: Perform a web search using the Tavily API.
            - view_last_game_llm_query: View the LLM query from the last game to see performance. 
            - view_last_game_results: View the game results from the last game. (Includes stdout and stderr from the game)
            
            YOUR GOAL: Create a Catan Player That Will play run_testfoo and Win Catan against the other players
            It must Incoorporate the LLM() class and the .query_llm() method

            """
        )

        try:

            log_path = os.path.join(CreatorAgent.run_dir, f"llm_log_{self.llm_name}.txt")

            # Variables for user input
            user_advice = ""
            user_advice_prompt = ""
            user_choose_continue = "y"

            while user_choose_continue == "y":

                # Place user feedback into the prompt
                if user_advice_prompt != "":
                    cur_prompt = user_advice_prompt
                else:
                    cur_prompt = prompt
                initial_input = {"messages": cur_prompt}

                # Run Through The Graph
                final_msg = ""
                with open(log_path, "a") as log_file:                # Run the graph until the first interruption
                    for event in self.react_graph.stream(initial_input, self.memory_config, stream_mode="values"):
                        msg = event['messages'][-1]
                        msg.pretty_print()
                        log_file.write((msg).pretty_repr())
                        final_msg = msg.content


                print("✅  graph finished")

                # Copy Result File to the new directory
                shutil.copy2(                           
                    (FOO_TARGET_FILE).resolve(),
                    (Path(CreatorAgent.run_dir) / FOO_TARGET_FILENAME)
                )

                # Ask the user if they want to continue, and for advice
                user_choose_continue = input("Continue? (y/n):")
                if user_choose_continue.lower() == "y":
                    user_advice = input("What do you want the agent to do? Press Enter for default:")
                    if user_advice == "":
                        user_advice = "Do This" + final_msg
                    user_advice_prompt = "NEW USER ADVICE " + user_advice + "/n/n" + prompt
                else:
                    user_advice_prompt = ""

        
        except Exception as e:
            print(f"Error calling LLM: {e}")
        return None

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
    
    # Copy Result File to the new directory
    shutil.copy2(                           
        (FOO_TARGET_FILE).resolve(),
        (Path(CreatorAgent.run_dir) / FOO_TARGET_FILENAME)
    )

    return f"{FOO_TARGET_FILENAME} updated successfully"


def run_testfoo(_: str = "") -> str:
    """
    Run one Catanatron match (R vs Agent File) and return raw CLI output.
    """    
    result = subprocess.run(
        shlex.split(FOO_RUN_COMMAND),
        capture_output=True,          # capture stdout+stderr :contentReference[oaicite:1]{index=1}
        text=True,
        timeout=3600,                  # avoids infinite-loop hangs
        check=False                   # we’ll return non-zero output instead of raising
    )
    
    game_results = (result.stdout + result.stderr).strip()

    # Update the run_test_foo flag to read game results
    global RUN_TEST_FOO_HAPPENED
    RUN_TEST_FOO_HAPPENED = True

    # Path to the runs directory
    runs_dir = Path(__file__).parent / "runs"
    
    # Find all folders that start with game_run
    game_run_folders = [f for f in runs_dir.glob("game_run*") if f.is_dir()]
    
    if not game_run_folders:
        return "No game run folders found."
    
    # Sort folders by name (which includes datetime) to get the most recent one
    latest_run_folder = sorted(game_run_folders)[-1]

    # Add a file with the stdout and stderr called catanatron_output.txt
    output_file_path = latest_run_folder / "catanatron_output.txt"
    with open(output_file_path, "w") as output_file:
        output_file.write(game_results)
        
    print(game_results)
    return game_results


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


def view_last_game_llm_query(query_number: int = -1) -> str:
    """
    View the game results from a specific run.
    
    Args:
        query_number: The index of the file to view (0-based). 
                     If -1 (default), returns the most recent file.
    
    Returns:
        The content of the requested game results file or an error message.
    """

    if RUN_TEST_FOO_HAPPENED == False:
        return "No game run has been executed yet."
    
    # Path to the runs directory
    runs_dir = Path(__file__).parent / "runs"
    
    # Find all folders that start with game_run
    game_run_folders = [f for f in runs_dir.glob("game_run*") if f.is_dir()]
    
    if not game_run_folders:
        return "No game run folders found."
    
    # Sort folders by name (which includes datetime) to get the most recent one
    latest_run_folder = sorted(game_run_folders)[-1]
    
    # Get all files in the folder and sort them
    result_files = sorted(latest_run_folder.glob("*"))
    
    if not result_files:
        return f"No result files found in {latest_run_folder.name}."
    
    # Determine which file to read
    file_index = query_number if query_number >= 0 else len(result_files) - 1
    
    # Check if index is valid
    if file_index >= len(result_files):
        return f"Invalid file index. There are only {len(result_files)} files (0-{len(result_files)-1})."
    
    target_file = result_files[file_index]
    
    # Read and return the content of the file
    try:
        with open(target_file, "r") as file:
            return f"Content of {target_file.name}:\n\n{file.read()}"
    except Exception as e:
        return f"Error reading file {target_file.name}: {str(e)}"
    

def view_last_game_results(_: str = "") -> str:
    """
    View the game results from a specific run.
    
    Returns:
        The content of the requested game results file or an error message.
    """

    if RUN_TEST_FOO_HAPPENED == False:
        return "No game run has been executed yet."
    
    # Path to the runs directory
    runs_dir = Path(__file__).parent / "runs"
    
    # Find all folders that start with game_run
    game_run_folders = [f for f in runs_dir.glob("game_run*") if f.is_dir()]
    
    if not game_run_folders:
        return "No game run folders found."
    
    # Sort folders by name (which includes datetime) to get the most recent one
    latest_run_folder = sorted(game_run_folders)[-1]

    # Read a file with the stdout and stderr called catanatron_output.txt
    output_file_path = latest_run_folder / "catanatron_output.txt"
    
    # Read and return the content of the file
    try:
        with open(output_file_path, "w") as file:
            return f"Content of {output_file_path.name}:\n\n{file.read()}"
    except Exception as e:
        return f"Error reading file {output_file_path.name}: {str(e)}"