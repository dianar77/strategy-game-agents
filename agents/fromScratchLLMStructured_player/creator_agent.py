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

from typing_extensions import TypedDict
from typing_extensions import TypedDict

# import warnings
# from langchain.warnings import LangChainDeprecationWarning   # same class they raise
# warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)



# -------- tool call configuration ----------------------------------------------------
LOCAL_SEARCH_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()
FOO_TARGET_FILENAME = "foo_player.py"
FOO_TARGET_FILE = Path(__file__).parent / FOO_TARGET_FILENAME    # absolute path
FOO_MAX_BYTES   = 64_000                                     # context-friendly cap
# Set winning points to 5 for quicker game
FOO_RUN_COMMAND = "catanatron-play --players=AB,FOO_LLM_S  --num=10 --config-map=MINI  --config-vps-to-win=10"
RUN_TEST_FOO_HAPPENED = False # Used to keep track of whether the testfoo tool has been called
# -------------------------------------------------------------------------------------

class CreatorAgent():
    """LLM-powered player that uses Claude API to make Catan game decisions."""
    # Class properties
    run_dir = None

    def __init__(self):
        # Get API key from environment variable
        # self.llm_name = "gpt-4o"
        # self.llm = AzureChatOpenAI(
        #     model="gpt-4o",
        #     azure_endpoint="https://gpt-amayuelas.openai.azure.com/",
        #     api_version = "2024-12-01-preview"
        # )

        # self.llm_name = "claude-3.7"
        # self.llm = ChatBedrockConverse(
        #     aws_access_key_id = os.environ["AWS_ACESS_KEY"],
        #     aws_secret_access_key = os.environ["AWS_SECRET_KEY"],
        #     region_name = "us-east-2",
        #     provider = "anthropic",
        #     model_id="arn:aws:bedrock:us-east-2:288380904485:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        # )
        # os.environ["LANGCHAIN_TRACING_V2"] = "false"


        self.llm_name = "mistral-large-latest"
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=1,    # Adjust based on your API tier
            check_every_n_seconds=0.1,
        )
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            max_retries=10,
            rate_limiter=rate_limiter,
        )

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
            "recursion_limit": 75, # set recursion limit for graph
            # "configurable": {
            #     "thread_id": "1"
            # }
        }
        self.react_graph = self.create_langchain_react_graph()

    def create_langchain_react_graph(self):
        """Create a react graph for the LLM to use."""
        

        class CreatorGraphState(TypedDict):
            full_results: HumanMessage # Last results of running the game
            analysis: HumanMessage         # Output of Anlayzer, What Happend?
            solution: HumanMessage         # Ouput of Researcher, What should be done?
            code_additions: HumanMessage         # Output of Coder, What was added to the code?
            test_results: HumanMessage # Running a test on code, to ensure correctness
            validation: HumanMessage       # Ouptut of Validator, Is the code correct?
            tool_calling_messages: list[AnyMessage]     # Messages from the tool calling state graph (used for debugging)
            summary: HumanMessage         # Summary of the conversation

            evolve_counter: int         # Counter for the number of evolutions
            validator_counter: int

        multi_agent_prompt = f"""You are apart of a multi-agent system that is working to evolve the code in {FOO_TARGET_FILENAME} to become the best player in the Catanatron Minigame. Get the highest score for the player by class in foo_player.py.\n\tYour specific role is the:"""

        #tools = [add, multiply, divide]
        DEFAULT_EVOLVE_COUNTER = 3
        DEFAULT_VALIDATOR_COUNTER = 2

        analyzer_continue_key = "CONTINUE_EVOLVING"
        analyzer_stop_key = "STOP_EVOLVING"


        val_ok_key = "PASSSED_VALIDATION"
        val_not_ok_key = "FAILED_VALIDATION"

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

        def tool_calling_state_graph(sys_msg: SystemMessage, msgs: list[AnyMessage], tools):

            # Filter out empty messages
            msgs = [m for m in msgs if _is_non_empty(m)]

            # Bind Tools to the LLM
            #llm_with_tools = self.llm.bind_tools(tools, parallel_tool_calls=False)
            llm_with_tools = self.llm.bind_tools(tools)

            def assistant(state: MessagesState):
                return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
            
            def trim_messages(state, keep_last_n: int = 10):
                """
                Keep the last `keep_last_n` conversational turns (assistant/user/system),
                preserve assistant–tool pairs, **and guarantee at least one SystemMessage**.
                """
                messages = state["messages"]

                kept, count = [], 0
                for m in reversed(messages):
                    # --- preserve assistant/tool pairs -----------------------------
                    if isinstance(m, ToolMessage):
                        kept.append(m)                            # tool itself
                        count += 1
                    elif isinstance(m, AIMessage) and m.tool_calls:
                        kept.append(m)
                        count += 1                # paired assistant
                    elif isinstance(m, SystemMessage):                     # system message added by invoke
                        continue
                    else:                                         # human or system
                        kept.append(m)
                        count += 1

                    if count >= keep_last_n:
                        break

                kept = list(reversed(kept))  # restore chronological order

            
            # Graph
            builder = StateGraph(MessagesState)

            # Define nodes: these do the work
            builder.add_node("assistant", assistant)
            builder.add_node("tools", ToolNode(tools))
            builder.add_node("trim_messages", trim_messages)

            # Define edges: these determine how the control flow moves
            builder.add_edge(START, "assistant")
            builder.add_conditional_edges(
                "assistant",
                # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
                # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
                tools_condition,
            )
            builder.add_edge("tools", "trim_messages")
            builder.add_edge("trim_messages", "assistant")

            react_graph = builder.compile()
            
            
            #messages = react_graph.invoke({"messages": msgs})
            for event in react_graph.stream({"messages": msgs}, stream_mode="values"):
                msg = event['messages'][-1]
                msg.pretty_print()
                print("\n")
                messages = event
            
            # for m in messages['messages']:
            #     m.pretty_print()

            return messages

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


            return {"evolve_counter": evolve_counter,"summary": HumanMessage(content="No Summary Yet")}

        def run_player_node(state: CreatorGraphState):
            """
            Runs Catanatron with the current Code
            """
            #print("In Run Player Node")

            # Generate a test results (later will be running the game)
            game_results = run_testfoo(short_game=False)
            #output = llm.invoke([HumanMessage(content="Create me a complicated algebra math problem, without a solution. Return only the problem...NO COMMENTARY!!")])
            
            # Clear all past messages
            return {
                "full_results": HumanMessage(content=f"GAME RESULTS:\n\n{game_results}"),
                "analysis": HumanMessage(content=""),
                "solution": HumanMessage(content=""),
                "code_additions": HumanMessage(content=""),
                "test_results": HumanMessage(content=""),
                "validation": HumanMessage(content=""),
                "tool_calling_messages": [],
                "validator_counter": DEFAULT_VALIDATOR_COUNTER,
            }

        def test_player_node(state: CreatorGraphState):
            """
            Tests Catanatron with the current Code
            """
            #print("In Test Player Node")
            game_results = run_testfoo(short_game=True)

            return {"test_results": HumanMessage(content=f"TEST GAME RESULTS (Not a Full Game):\n\n{game_results}")}

        def analyzer_node(state: CreatorGraphState):
            #print("In Analyzer Node")

            # If evolve_counter isnt initialized, set it to 0. If it is, increment it
            
            evolve_counter = state["evolve_counter"] - 1


            sys_msg = SystemMessage(
                content=f"""
                    {multi_agent_prompt} ANALYZER
                    
                    Task: Your job is to analyze the results of how foo_player did in the game and create a report to the Researcher
                    
                    1. Analyze
                        - Analyze on any errors or issues that occurred during the game
                        - Analyze on the performance of the player, and how it did against other players
                        - Analyze on any terminal ouput from the player

                    2. Think
                        - Think about what the researcher should look into for the next iteration (Note: The Researher can search the web)
                        - Think about what the coder should look into for the next iteration
                        - Think about why the other player is winning, and how to beat it
                    
                    3. Decide
                        - Decide if the player is is good enough to stop evolving, or if it should continue evolving
                        - If the player is can beat the players consistently, just return the key "{analyzer_stop_key}" (Ignore Step 4)
                        - If the player is not good enough, return the key "{analyzer_continue_key}" (defaults to not good enough)

                    4. Report (Output)
                        - Create a concise and efficient report with summarized results, analysis, thoughts, and action items for the researcher
                        - Include anything you learned from your tools calls
                        - Make sure to include any errors or issues that occurred during the game
                        - Be sure a clear list of action items for the researcher to follow

                    
                    Utilize as many tool calls as necessary for you to get your job done. 

                    You Have the Following Tools at Your Disposal:
                        - list_local_files: List all files in the current directory.
                        - read_local_file: Read the content of a file in the current directory.(DO NOT CALL MORE THAT TWICE)
                        - read_foo: Read the content of {FOO_TARGET_FILENAME}.
                    
                    KEEP YOUR TOOL CALLS TO A MINIMUM!
                    Make sure to start your output with 'ANALYSIS:' and end with 'END ANALYSIS'.
                    Respond with No Commentary, just the analysis.

                """
            )
            msg = [state["summary"], state["code_additions"], state["full_results"]]
            tools = [list_local_files, read_local_file, read_foo]
            output = tool_calling_state_graph(sys_msg, msg, tools)
            analysis = HumanMessage(content=output["messages"][-1].content)

            #print(output)
            return {"analysis": analysis, "evolve_counter": evolve_counter, "tool_calling_messages": output["messages"]}
        
        def researcher_node(state: CreatorGraphState):
            
            #print("In Researcher Node")
            # Add custom tools for researcher

            sys_msg = SystemMessage(
                content=f"""
                    {multi_agent_prompt} RESEARCHER
                     
                    Task: Digest the analysis from the Analyzer, perform your own research, STRATEGIZE!!, and create a solution for the Coder to implement

                    
                    1. Digest
                        - Digest the analysis and game summary from the Analyzer.
                        - If needed, use the read_foo tool call to view the player to understand the code
                        - Digest the recommended questions and action items from the Analyzer

                    2. Research
                        - Perform research on the questions and action items from the Analyzer (or any other questions you have)
                        - Use the web_search_tool_call to perform a web search for any questions you have (REALLY BENEFICIAL TO USE THIS)
                        - Use the list_local_files, and read_local_file to view any game files (which are very helpful for debugging)
                        - Determine why the other player is winning, and how to beat it
                        
                    3. Strategize
                        - Think on a high level about what the coder should do to achieve the goal
                        - Create a plan with instructions for the coder to implement the solution
                        - Most Importantly: BE CREATIVE AND THINK OUTSIDE THE BOX (feel free to web search for anything you want)
                        - You must find a way to beat the other player

                
                    4. Report (Output)
                        - Create a concise and efficient report with analyzer questions and answers, your resarch, the strategy, and plan for the coder
                        - Include anything you learned from your tools calls
                        - Give clear instructions to the coder on what to implement (including any code snippets or syntax help)


                    You Have the Following Tools at Your Disposal:
                        - list_local_files: List all files in the current directory.
                        - read_local_file: Read the content of a file in the current directory.(DO NOT CALL MORE THAT TWICE)
                        - read_foo: Read the content of {FOO_TARGET_FILENAME}.
                        - web_search_tool_call: Perform a web search using the Tavily API.

                    KEEP YOUR TOOL CALLS TO A MINIMUM!
                    Make sure to start your output with 'SOLUTION:' and end with 'END SOLUTION'.
                    Respond with No Commentary, just the Research.

                """
            )

            # Choose the input based on if coming from analyzer or from validator in graph
            msg = [state["summary"], state["full_results"], state["analysis"]]

            tools = [read_foo, list_local_files, read_local_file, web_search_tool_call]
            output = tool_calling_state_graph(sys_msg, msg, tools)
            solution = HumanMessage(content=output["messages"][-1].content)
            return {"solution": solution, "tool_calling_messages": output["messages"]}

        def coder_node(state: CreatorGraphState):

            #print("In Researcher Node")
            # Add custom tools for researcher

            sys_msg = SystemMessage(
                content=f"""
                    {multi_agent_prompt} CODER
                    
                    Task: Digest at the proposed solution from the Researcher and Analyzer, and implement it into the foo_player.py file.

                    1. Digest 
                        - Digest the solution provided by the Researcher and the Analyzer, and Validator if applicable
                        - Look at the code from the foo_player.py file using the read_foo tool call
                        - Utilize the list_local_file and read_local_file to view any game files (Very helpful for debugging!)

                    2. Implement
                        - Use what you learned and digested to call write_foo tool call and write the entire new code for the foo_player.py file
                        - Focus on making sure the code implementes the solution in the most correct way possible
                        - Make Sure to not add backslashes to comments
                            WRONG:        print(\\'Choosing First Action on Default\\')
                            CORRECT:      print('Choosing First Action on Defaul')
                        - Give plenty of comments in the code to explain what you are doing, and what you have learned (along with syntax help)
                        - Use print statement to usefully debug the output of the code
                        - DO NOT MAKE UP VARIABLES OR FUNCTIONS RELATING TO THE GAME
                        - Note: You will have multiple of iterations to evolve, so make sure the syntax is correct

                    3. Review
                        - Run through the output of the write_foo tool call to make sure the code is correct, and contains now errors or bugs
                        - If there are any errors or bugs, fix them and re-run the write_foo tool call
                    
                    4. Report (Output)
                        - Create a concise and efficient report with the additions to the code you made, and why you made them for the validator
                        - Include anything you learned from your tools calls

                    You Have the Following Tools at Your Disposal:
                        - list_local_files: List all files in the current directory.
                        - read_local_file: Read the content of a file in the current directory.
                        - read_foo: Read the content of {FOO_TARGET_FILENAME}.
                        - write_foo: Write the content of {FOO_TARGET_FILENAME}. (Make sure to keep imports) Note: print() commands will be visible in view_last_game_results

                    KEEP YOUR TOOL CALLS TO A MINIMUM!
                    Make sure to start your output with 'CODER' and end with 'END CODER'.

                    
                """
            )
           
            # Choose the input based on if coming from analyzer or from validator in graph
            if state["validation"].content == "":
                # If coming from analyzer, use the full_results, analusis, and solution
                print("Coder Coming from Analyzer")
                #msg = [state["full_results"], state["analysis"], state["solution"]]
                msg = [state["solution"]]
            else:
                # If coming from validator, usee the coder, test_results, and validation messages
                print("Coder Coming from Validator")
                #msg = [state["code_additions"], state["test_results"], state["validation"]]
                msg = [state["solution"], state["validation"]]
            
            tools = [list_local_files, read_local_file, read_foo, write_foo]

            # Need to Return Anything?
            output = tool_calling_state_graph(sys_msg, msg, tools)
            code_additions = HumanMessage(content=output["messages"][-1].content)

            return {"code_additions": code_additions, "tool_calling_messages": output["messages"]}

        def validator_node(state: CreatorGraphState):
            """
            Validates the code
            """
            #print("In Validator Node")
            # Add Custom Tools For Validator
            
            sys_msg = SystemMessage(
                content=f"""
                    {multi_agent_prompt} VALIDATOR
                    
                    Task: Analyze the results of the test game and the new additions, and determine if the code is correct or not

                    1. Digest
                        - Digest the test results from the new code that was writen by the coder node
                        - Digest the code additions from the coder node, and use the read_foo tool call to view the actual player code if needed

                    2. Validate
                        - Validate the test results and determine if the code is correct or not
                        - Validate to ensure there are no errors or bugs in the code
                        - Validate to ensure the output of the test game is correct and matches the expected output
                    
                    3. Recommend
                        - You are ONLY checking to see if the code has correct execution and no errors
                        - If validation is successful, return the key "{val_ok_key}", 
                        - Otherwise, on a VERY limited time basis, return "{val_not_ok_key}".
                        - Then, commentate on why you decided to return the key
                        - Note: The ouptput will be parse for either keys, so make sure to only return one of them

                    You Have the Following Tools at Your Disposal:
                        - read_foo: Read the content of {FOO_TARGET_FILENAME}.

                    
                    Note: It is okay if the model is not perfect and is novel. 
                    Your job is to make sure the model works and is correct so it can be tested in a full game.
                    Only return "{val_not_ok_key}" if there is a trivial error that can be fixed easily by the Coder
                    
                    
                    Make sure to start your output with 'VALIDATION:' and end with 'END VALIDATION'. 
                    Respond with No Commentary, just the Validation.
                    
                """
            )
            msg = [state["summary"], state["solution"], state["code_additions"], state["test_results"]]
            tools = [read_foo]
            output = tool_calling_state_graph(sys_msg, msg, tools)
            validation = HumanMessage(content=output["messages"][-1].content)

            validator_counter = state["validator_counter"] - 1

            return {"validation": validation, "tool_calling_messages": output["messages"], "validator_counter": validator_counter}

        def summarize_conversation(state: CreatorGraphState):
    
            print("In Summarizer Node")
            # First, we get any existing summary
            if "summary" not in state:
                summary = "No summary yet"
            else:
                summary = state["summary"].content

            # Create our summarization prompt 
            if summary:
                # A summary already exists
                summary_message = (
                    f"""You are tasked with summarizing an Multi-Agent Workflow with the steps Full_Results, Analysis, Solution, Code Additions
                    This workflow will iterate until the code is correct and the game is won
                    This is your current summary of the workflow steps:
                    
                    {summary}

                    Above, you have new Full_Results, Analysis, Solution, Code Additions Messages for a new step in the Multi-Agent Workflow
                    Your Summary should look like the following:

                    Turm 0: <Short High Level Description>
                        Game Results Summary: <summary of the game results>
                        Analysis: <summary of the analysis>
                        Solution: <summary of the solution>
                        Code Additions: <summary of the code additions>
                    
                    Turn 1: Short High Level Description
                        Game Results Summary: <summary of the game results>
                        Analysis: <summary of the analysis>
                        Solution: <summary of the solution>
                        Code Additions: <summary of the code additions>

                    ...... And so on
                    

                    Please update the summary to include the new messages.
                    Make sure to include the previous summary contents, and the new summary content in your output

                    """
                )
                
            else:
                summary_message = "Create a summary of the conversation above:"

            print("Summary")
            state_msgs = [state["full_results"], state["analysis"], state["solution"], state["code_additions"], state["validation"]]
            sys_msg = SystemMessage(content=summary_message)
            tools = []
            output = tool_calling_state_graph(sys_msg, state_msgs, tools)
            summary = HumanMessage(content=output["messages"][-1].content)
            
            return {"summary": summary}
        
        def continue_evolving_analyzer(state: CreatorGraphState):
            """
            Conditional edge for Analyzer
            """
            print("In Conditional Edge Analyzer")
            
            # Get the content of the validation message
            analyzer_message = state["analysis"].content
            
            # Check for the presence of our defined result strings (Because analyzer node decrements the counter)
            if state["evolve_counter"] <= -1:
                print("Evolve counter is 0 - ending workflow")
                return END

            if analyzer_stop_key in analyzer_message:
                print("Validation passed - ending workflow")
                return END
            elif analyzer_continue_key in analyzer_message:  #
                print("Validation failed - rerunning player")
                return "researcher"
            else:
                # Default case if neither string is found
                print("Warning: Could not determine validation result, defaulting to researcher")
                return "researcher"
            
        def code_ok_validator(state: CreatorGraphState):
            """
            Conditional edge for validator
            """
            print("In Conditional Edge Validator")
            
            # Get the content of the validation message
            validation_message = state["validation"].content
            
            # Check for the presence of our defined result strings
            if state["validator_counter"] <= -1:
                print("Validator counter is 0 - rerunning player")
                return "summarizer"

            if val_ok_key in validation_message:
                print("Validation passed - rerunning player")
                return "summarizer"
            elif val_not_ok_key in validation_message:  #
                print("Validation failed - going back to coder")
                return "coder"
            else:
                # Default case if neither string is found
                print("Warning: Could not determine validation result, defaulting to running player")
                return "summarizer"

        def construct_graph():
            graph = StateGraph(CreatorGraphState)
            graph.add_node("init", init_node)
            graph.add_node("run_player", run_player_node)
            graph.add_node("analyzer", analyzer_node)
            #graph.add_node("tools", ToolNode(tools))
            graph.add_node("researcher", researcher_node)
            graph.add_node("coder", coder_node)
            graph.add_node("test_player", test_player_node)
            graph.add_node("validator", validator_node)
            graph.add_node("summarizer", summarize_conversation)

            graph.add_edge(START, "init")
            graph.add_edge("init", "run_player")
            graph.add_edge("run_player", "analyzer")
            graph.add_conditional_edges(
                "analyzer",
                continue_evolving_analyzer,
                {END, "researcher"}
            )
            #graph.add_edge("analyzer", "researcher")
            graph.add_edge("researcher", "coder")
            graph.add_edge("coder", "test_player")
            graph.add_edge("test_player", "validator")
            graph.add_conditional_edges(
                "validator",
                code_ok_validator,
                {"coder", "summarizer"}
            )
            graph.add_edge("summarizer", "run_player")

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

            log_path = os.path.join(CreatorAgent.run_dir, f"llm_log_{self.llm_name}.txt")

            with open(log_path, "a") as log_file:                # Run the graph until the first interruption
                for step in self.react_graph.stream({}, self.config, stream_mode="updates"):
                    #print(step)
                    #log_file.write(f"Step: {step.}\n")
                    for node, update in step.items():
                        if node == "run_player":
                            print("In run_player node")
                        elif node == "analyzer":
                            print("In analyzer node")
                        elif node == "researcher":
                            print("In researcher node")
                        elif node == "coder":
                            print("In coder node")
                        elif node == "test_player":
                            print("In test_player node")
                        elif node == "validator":
                            print("In validator node")
                        elif node == "summarizer":
                            print("In summarizer node")
                        

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

                        if "analysis" in update:
                            msg = update["analysis"]
                            msg.pretty_print()
                            log_file.write((msg).pretty_repr())
                        if "solution" in update:
                            msg = update["solution"]
                            msg.pretty_print()
                            log_file.write((msg).pretty_repr())
                        if "code_additions" in update:
                            msg = update["code_additions"]
                            msg.pretty_print()
                            log_file.write((msg).pretty_repr())
                        if "validation" in update:
                            msg = update["validation"]
                            msg.pretty_print()
                            log_file.write((msg).pretty_repr())
                        if "summary" in update:
                            msg = update["summary"]
                            msg.pretty_print()
                            log_file.write((msg).pretty_repr())
                        if "evolve_counter" in update:
                            print("ENVOLVE COUNTER: ", update["evolve_counter"])
                            log_file.write(f"Evolve Counter: {update['evolve_counter']}\n")
                        if "validator_counter" in update:
                            print("VALIDATOR COUNTER: ", update["validator_counter"])
                            log_file.write(f"Validator Counter: {update['validator_counter']}\n")
                        if "test_results" in update:
                            print("Test Results:", update["test_results"])
                            log_file.write(f"Test Results: {update['test_results']}\n")
                        if "full_results" in update:
                            print("Full Results:", update["full_results"])
                            log_file.write(f"Full Results: {update['full_results']}\n")


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
    dt = datetime.now().strftime("%Y%m%d_%H%M%S_")

    shutil.copy2(                           
        (FOO_TARGET_FILE).resolve(),
        (Path(CreatorAgent.run_dir) / (dt + FOO_TARGET_FILENAME))
    )

    return f"{FOO_TARGET_FILENAME} updated successfully"

def run_testfoo(short_game: bool = False) -> str:
    """
    Run one Catanatron match (R vs Agent File) and return raw CLI output.
    Input: short_game (bool): If True, run a short game with a 30 second timeout.
    """    
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
        
    #print(game_results)
        # limit the output to a certain number of characters
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
    








         
    # def create_langchain_react_graph(self):
    #     """Create a react graph for the LLM to use."""
        
    #     tools = [
    #         list_local_files,
    #         read_local_file,
    #         read_foo,
    #         write_foo,
    #         run_testfoo,
    #         web_search_tool_call,
    #         view_last_game_llm_query,
    #         view_last_game_results
    #     ]

    #     llm_with_tools = self.llm.bind_tools(tools)
        
    #     def assistant(state: MessagesState):
    #         return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
    #     def trim_messages(state: MessagesState):
    #         # Delete all but the specified most recent messages
    #         delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-self.num_memory_messages]]
    #         return {"messages": delete_messages}


    #     builder = StateGraph(MessagesState)

    #     # Define nodes: these do the work
    #     builder.add_node("trim_messages", trim_messages)
    #     builder.add_node("assistant", assistant)
    #     builder.add_node("tools", ToolNode(tools))

    #     # Define edges: these determine how the control flow moves
    #     builder.add_edge(START, "assistant")
    #     #builder.add_edge("trim_messages", "assistant")
    #     builder.add_conditional_edges(
    #         "assistant",
    #         # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    #         # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    #         tools_condition,
    #     )
    #     builder.add_edge("tools", "trim_messages")
    #     builder.add_edge("trim_messages", "assistant")
        
    #     return builder.compile(checkpointer=MemorySaver())
