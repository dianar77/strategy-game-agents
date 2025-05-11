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
from langgraph.graph import MessagesState, START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from typing_extensions import TypedDict
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

# -------- tool call configuration ----------------------------------------------------
LOCAL_SEARCH_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()
FOO_TARGET_FILENAME = "foo_player.py"
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME    # absolute path
FOO_MAX_BYTES   = 64_000                                     # context-friendly cap
FOO_RUN_COMMAND = "catanatron-play --players=AB,R,FOO_S_S --num=1 --config-map=MINI"

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

        # Create run directory if it doesn't exist
        if CreatorAgent.run_dir is None:
            agent_dir = os.path.dirname(os.path.abspath(__file__))
            runs_dir = os.path.join(agent_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
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
        #self.num_memory_messages = 10        # Trim number of messages to keep in memory to limit API usage
        self.react_graph = self.create_langchain_react_graph()
        
    def create_langchain_react_graph(self):
        """Create a react graph for the LLM to use."""
        
        tools = [list_local_files, read_local_file, read_foo, write_foo, run_testfoo, web_search_tool_call]
        class CreatorGraphState(TypedDict):
            # Place Variables Here
            
            full_results: SystemMessage           # Last results of running the game
            analysis: AIMessage         # Output of Anlayzer, What Happend?
            solution: AIMessage         # Ouput of Researcher, What should be done?
            new_code: AIMessage         # Output of Coder
            test_results: SystemMessage           # Running a test on code, to ensure correctness
            validation: AIMessage        # Ouptut of Validator, Is the code correct?

            evolve_counter: int         # Counter for the number of evolutions


        #tools = [add, multiply, divide]
        DEFAULT_EVOLVE_COUNTER = 3
        #llm = ChatOpenAI(model="gpt-4o")
        llm = AzureChatOpenAI(
            model="gpt-4o",
            azure_endpoint="https://gpt-amayuelas.openai.azure.com/",
            api_version = "2024-12-01-preview"
        )

        def tool_calling_state_graph(sys_msg: SystemMessage, msgs: list[AnyMessage], tools):
            # Node

            # Bind Tools to the LLM
            llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

            def assistant(state: MessagesState):
                return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
            
            # Graph
            builder = StateGraph(MessagesState)

            # Define nodes: these do the work
            builder.add_node("assistant", assistant)
            builder.add_node("tools", ToolNode(tools))

            # Define edges: these determine how the control flow moves
            builder.add_edge(START, "assistant")
            builder.add_conditional_edges(
                "assistant",
                # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
                # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
                tools_condition,
            )
            builder.add_edge("tools", "assistant")
            react_graph = builder.compile()
            
            
            messages = react_graph.invoke({"messages": msgs})
            
            for m in messages['messages']:
                m.pretty_print()

            return messages


        def run_player_node(state: CreatorGraphState):
            """
            Runs Catanatron with the current Code
            """
            #print("In Run Player Node")

            # Generate a test results (later will be running the game)
            output = llm.invoke([HumanMessage(content="Create me a complicated algebra math problem, without a solution. Return only the problem...NO COMMENTARY!!")])
            
            # Clear all past messages
            return {
                "full_results": SystemMessage(content=output.content),
                "analysis": AIMessage(content=""),
                "solution": AIMessage(content=""),
                "new_code": AIMessage(content=""),
                "test_results": SystemMessage(content=""),
                "validation": AIMessage(content="")
            }


        def test_player_node(state: CreatorGraphState):
            """
            Tests Catanatron with the current Code
            """
            #print("In Test Player Node")

            return {"test_results": SystemMessage(content="These are the test results: YOU PASSED THE TEST!")}

        def analyzer_node(state: CreatorGraphState):
            #print("In Analyzer Node")

            # If evolve_counter isnt initialized, set it to 0. If it is, increment it
            if "evolve_counter" not in state:
                evolve_counter = DEFAULT_EVOLVE_COUNTER
            else:
                evolve_counter = state["evolve_counter"] - 1


            sys_msg = SystemMessage(
                content="""
                    You are a Math Agent. I want you to output just the analysis of the problem, not the solution "
                    Make sure to start your output with 'ANALYSIS:' and end with 'END ANALYSIS'. No Commentary, just the solution.
                """
            )
            msg = [state["full_results"]]
            tools = []
            output = tool_calling_state_graph(sys_msg, msg, tools)

            #print(output)
            return {"analysis": output["messages"][-1], "evolve_counter": evolve_counter}
        
        def researcher_node(state: CreatorGraphState):
            
            #print("In Researcher Node")
            # Add custom tools for researcher

            sys_msg = SystemMessage(
                content="""
                    You are a Math Agent. I want you to research the given analysis on the previous equation, and research a way to achieve the solution.
                    This does not need specifically be the solution. Make sure to start your output with 'SOLUTION:' and end with 'END SOLUTION'. No Commentary, just the solution.
                """
            )
            msg = [state["analysis"], state["full_results"]]
            tools = []
            output = tool_calling_state_graph(sys_msg, msg, tools)
            return {"solution": output["messages"][-1]}

        def coder_node(state: CreatorGraphState):

            #print("In Researcher Node")
            # Add custom tools for researcher

            sys_msg = SystemMessage(
                content="""
                    You are a Math Agent. I want you take in the proposed solution, and implement it to output the result
                """
            )
            msg = [state["solution"]]
            tools = []
            output = tool_calling_state_graph(sys_msg, msg, tools)
            return {"new_code": output["messages"][-1]}

        val_ok_key = "PASSSED_VALIDATION"
        val_not_ok_key = "FAILED_VALIDATION"

        def validator_node(state: CreatorGraphState):
            """
            Validates the code
            """
            #print("In Validator Node")
            # Add Custom Tools For Validator
            
            sys_msg = SystemMessage(
                content=f"""
                    You are a Math Agent. The previous system message is an external validation tool as a results of your solution, which is in the previous message.
                    I want you to look at the results of the test, and follow the following instructions:
                    If the results are correct, return the key "{val_ok_key}", if the results are correct, return the key "{val_not_ok_key}".
                    Then, respond with an analysis and explanation of what key you decided to return.
                    Make sure to start your output with 'VALIDATION:' and end with 'END VALIDATION'. No Commentary, just the solution.
                """
            )
            msg = [state["new_code"], state["test_results"], ]
            tools = []
            output = tool_calling_state_graph(sys_msg, msg, tools)

            return {"validation": output["messages"][-1]}

        def conditional_edge_validator(state: CreatorGraphState):
            """
            Conditional edge for validator
            """
            print("In Conditional Edge Validator")
            
            # Get the content of the validation message
            validation_message = state["validation"].content
            
            # Check for the presence of our defined result strings
            if state["evolve_counter"] <= 0:
                print("Evolve counter is 0 - ending workflow")
                return END

            if val_ok_key in validation_message:
                print("Validation passed - ending workflow")
                return END
            elif val_not_ok_key in validation_message:  #
                print("Validation failed - rerunning player")
                return "run_player"
            else:
                # Default case if neither string is found
                print("Warning: Could not determine validation result, defaulting to END")
                return END


        graph = StateGraph(CreatorGraphState)

        graph.add_node("run_player", run_player_node)
        graph.add_node("analyzer", analyzer_node)
        #graph.add_node("tools", ToolNode(tools))
        graph.add_node("researcher", researcher_node)
        graph.add_node("coder", coder_node)
        graph.add_node("test_player", test_player_node)
        graph.add_node("validator", validator_node)

        graph.add_edge(START, "run_player")
        graph.add_edge("run_player", "analyzer")
        graph.add_edge("analyzer", "researcher")
        graph.add_edge("researcher", "coder")
        graph.add_edge("coder", "test_player")
        graph.add_edge("test_player", "validator")
        graph.add_conditional_edges(
            "validator",
            conditional_edge_validator,
            {END, "run_player"}
        )
        #graph.add_edge("validator", END)

        return graph.compile()


    def print_react_graph(self):
        """
        Print the react graph for debugging purposes.
        ONLY WORKS IN .IPYNB NOTEBOOKS
        """
        display(Image(self.react_graph.get_graph(xray=True).draw_mermaid_png()))

    def run_react_graph(self):
        #initial_state: CreatorGraphState = {}
        for step in self.react_graph.stream({}, stream_mode="updates"):
            print(step)

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
        timeout=120,                  # avoids infinite-loop hangs
        check=False                   # we’ll return non-zero output instead of raising
    )
    return (result.stdout + result.stderr).strip()

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
