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
FOO_RUN_COMMAND = "catanatron-play --players=AB,FOO_LLM_S2  --num=10 --config-map=MINI  --config-vps-to-win=10"
RUN_TEST_FOO_HAPPENED = False # Used to keep track of whether the testfoo tool has been called
# -------------------------------------------------------------------------------------

class CreatorAgent():
    """LLM-powered player that uses Claude API to make Catan game decisions."""
    # Class properties
    run_dir = None
    current_evolution = 0

    def __init__(self):
        # Get API key from environment variable
        self.llm_name = "gpt-4o"
        self.llm = AzureChatOpenAI(
            model="gpt-4o",
            azure_endpoint="https://gpt-amayuelas.openai.azure.com/",
            api_version = "2024-12-01-preview"
        )

        # config = Config(read_timeout=1000)
        # bedrock_client = client(service_name='bedrock-runtime', region_name='us-east-2', config=config)
        # self.llm_name = "claude-3.7"
        # self.llm = ChatBedrockConverse(
        #     aws_access_key_id = os.environ["AWS_ACESS_KEY"],
        #     aws_secret_access_key = os.environ["AWS_SECRET_KEY"],
        #     region_name = "us-east-2",
        #     provider = "anthropic",
        #     model_id="arn:aws:bedrock:us-east-2:288380904485:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        # )
        # os.environ["LANGCHAIN_TRACING_V2"] = "false"


        # self.llm_name = "mistral-large-latest"
        # rate_limiter = InMemoryRateLimiter(
        #     requests_per_second=1,    # Adjust based on your API tier
        #     check_every_n_seconds=0.1,
        # )
        # self.llm = ChatMistralAI(
        #     model="mistral-large-latest",
        #     temperature=0,
        #     max_retries=10,
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
            full_results: HumanMessage # Last results of running the game
            analysis: HumanMessage         # Output of Anlayzer, What Happend?
            strategy: HumanMessage         # Output of Strategizer, What should be done?
            solution: HumanMessage         # Ouput of Researcher, How should it be implemented?
            code_additions: HumanMessage         # Output of Coder, What was added to the code?
            test_results: HumanMessage # Running a test on code, to ensure correctness
            validation: HumanMessage       # Ouptut of Validator, Is the code correct?
            tool_calling_messages: list[AnyMessage]     # Messages from the tool calling state graph (used for debugging)
            performance_history: HumanMessage

            evolve_counter: int         # Counter for the number of evolutions
            validator_counter: int

        multi_agent_prompt = f"""You are apart of a multi-agent system that is working to evolve the code in {FOO_TARGET_FILENAME} to become the best player in the Catanatron Minigame. Get the highest score for the player by class in foo_player.py. Your performance history on this trial is saved in the json\n\tYour specific role is the:"""

        #tools = [add, multiply, divide]
        DEFAULT_EVOLVE_COUNTER = 10
        DEFAULT_VALIDATOR_COUNTER = 2
        MAX_MESSAGES_TOOL_CALLING = 5

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

            last_event = None
            try:
                for event in react_graph.stream({"messages": msgs}, config=config, stream_mode="values"):
                    msg = event['messages'][-1]
                    msg.pretty_print()
                    print("\n")
                    last_event = event
                return last_event
            except GraphRecursionError as e:
                print(f"Recursion limit reached {MAX_MESSAGES_TOOL_CALLING}: {e}")

            
            # If End Early, Must Still Get Output
            last_message = last_event["messages"][-1]

            # If is a last AI Message and requests tools calls, delete it, if does not request tool calls, return it
            if isinstance(last_message, AIMessage):
                if not last_message.tool_calls:
                    return last_event
                else:
                    # If the last message is an AI message with a tool call, remove it and add another AI message
                    last_event["messages"] = last_event["messages"][:-1]
                    last_event["messages"].append(AIMessage(content="OOPS! I made a mistake, I used too many tool calls"))

            # If last is a tool call message, add AI message for mistake        
            elif isinstance(last_message, ToolMessage):
                last_event["messages"].append(AIMessage(content="OOPS! I made a mistake, I used too many tool calls"))

            # Add Human Message with instructionrs
            last_event["messages"].append(HumanMessage(content= """
                YOU CAN NO LONGER USE TOOLS! YOU MUST USE WHAT KNOWLEDGE YOU HAVE TO ANSWER THE SYSTEM PROMPT"""
            ))

            # Combine the system message with the existing messages
            input_msg = [sys_msg] + last_event["messages"]

            # Invoke the LLM with the adjusted message sequence
            assistant_response = llm_with_tools.invoke(input_msg)

            # Append the assistant's response to the message history
            last_event["messages"].append(assistant_response)

            
            # for m in messages['messages']:
            #     m.pretty_print()

            return last_event

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


            return {
                "evolve_counter": evolve_counter,
                "performance_history": HumanMessage(content="No Performance History Yet"),
                "analysis": HumanMessage(content=""),
                "strategy": HumanMessage(content=""),
                "solution": HumanMessage(content=""),
                "validation": HumanMessage(content=""),
                "code_additions": HumanMessage(content=""),
            }

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
                "validator_counter": DEFAULT_VALIDATOR_COUNTER,
            }

        def test_player_node(state: CreatorGraphState):
            """
            Tests Catanatron with the current Code
            """
            #print("In Test Player Node")
            game_results = run_testfoo(short_game=True)

            return {"test_results": HumanMessage(content=f"TEST GAME RESULTS (Not a Full Game):\n\n{game_results}")}
        
        def summarizer_node(state: CreatorGraphState):
    
            print("In Summarizer Node")
            sys_msg = SystemMessage(content=
                f"""You are tasked with summarizing an Multi-Agent Workflow with the steps Full_Results, Analysis, Strategy, Solution, Code Additions, and Validation
                This workflow will iterate until the code is correct and the game is won. 

                Above, you have new Full_Results, Analysis, Solution, Code Additions Messages for a new step in the Multi-Agent Workflow
                Your Summary should look like the following:

                <Short High Level Description>
                    Game Results Summary: <summary of the game results>
                    Analysis: <summary of the analysis>
                    Strategy: <summary of the strategy>
                    Solution: <summary of the solution>
                    Code Additions: <summary of the code additions>
                    Validation: <summary of the validation>
                

                Do not make anything up. Only write what you are given, or nothing at all

                IMPORTANT: Only include the summary in the output, no other commentary or information
                Make sure to keep the summary concise and to the point

                """
            )
            
            print("Summary")
            state_msgs = [state["full_results"], state["analysis"], state["strategy"], state["solution"], state["code_additions"], state["validation"]]
            tools = []
            output = tool_calling_state_graph(sys_msg, state_msgs, tools)
            summary = output["messages"][-1].content
            
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
                        performance_history[evolution_key]["summary"] = summary
                        
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
                "analysis": HumanMessage(content=""),
                "strategy": HumanMessage(content=""),
                "solution": HumanMessage(content=""),
                "code_additions": HumanMessage(content=""),
                "test_results": HumanMessage(content=""),
                "validation": HumanMessage(content=""),
                "tool_calling_messages": [],
            }
        
        def analyzer_node(state: CreatorGraphState):
            #print("In Analyzer Node")

            # If evolve_counter isnt initialized, set it to 0. If it is, increment it
            
            evolve_counter = state["evolve_counter"] - 1


            sys_msg = SystemMessage(
                content=f"""
                    {multi_agent_prompt} ANALYZER
                    
                    Task: Your job is to analyze the results of how {FOO_TARGET_FILENAME} did in the game and create a report to the Strategizer
                    
                    1. Analyze
                        - 1st Message Given to You:  Analyze your current performance history (1st message) with how you are evolving the player
                            If you need to view the game output, call read_local_file with the filepath in "full_game_log_path" variable
                        - 2nd Message Given to You:  Analyze the CODER additions to the code
                        - 3rd Message Given to You: Analyze the results of the new player
                            - Analyze on any errors or issues that occurred during the game
                            - Analyze the results of the player, and how it did against other players
                        - If needed, analyze on any ouput from the player
                        - Look for evidence of any flaws in the code or the player

                    
                    2. Decide
                        - Decide if the player is is good enough to stop evolving, or if it should continue evolving
                        - If the player is can beat the players consistently, just return the key "{analyzer_stop_key}" (Ignore Step 4)
                        - If the player is not good enough, return the key "{analyzer_continue_key}" (defaults to not good enough)

                    3. Report (Output)
                        - Create a concise and efficient report with your summary and analysis of the previous game
                        - PRIORITIZE FIXING BUGS AND ERRORS THAT ARISE
                        - Include anything you learned from your tools calls
                        - Include questions for the strategizer to research
                    
                    You Have the Following Tools at Your Disposal:
                        - list_catanatron_files: List all files beneath the Catanatron base directory.
                        - read_full_performance_history: Return the entire performance history as a JSON string.
                        - read_local_file: Read the content of a catanatron file, or a performance history file in the run directory. (DO NOT CALL MORE THAN TWICE)
                        - read_foo: Read the content of {FOO_TARGET_FILENAME}.
                    
                    KEEP YOUR TOOL CALLS TO A MINIMUM!
                    YOU ARE LIMITED TO {MAX_MESSAGES_TOOL_CALLING} TOOL CALLS

                    Make sure to start your output with 'ANALYSIS:' and end with 'END ANALYSIS'.
                    Respond with No Commentary, just the analysis.

                """
            )
            msg = [state["performance_history"], state["code_additions"], state["full_results"]]
            tools = [list_catanatron_files, read_full_performance_history, read_local_file, read_foo]
            output = tool_calling_state_graph(sys_msg, msg, tools)
            analysis = HumanMessage(content=output["messages"][-1].content)

            #print(output)
            return {"analysis": analysis, "evolve_counter": evolve_counter, "tool_calling_messages": output["messages"]}
        
        def strategizer_node(state: CreatorGraphState):
            
            #print("In Strategizer Node")
            # Add custom tools for strategizer

            sys_msg = SystemMessage(
                content=f"""
                    {multi_agent_prompt} STRATEGIZER
                     
                    Task: Digest the analysis from the Analyzer, and devise a STRATEGY that the Researcher can research, and the coder can implement

                    
                    1. Digest
                        - 1st Message: Digest the performance history with how you are evolving the player 
                        - 2nd Message: Digest the analysis and game summary from the Analyzer.

                    3. Strategize
                        - Think on a high level about what the coder should do to achieve the goal of becoming the Master of Catan
                        - Most Importantly: BE CREATIVE AND THINK OUTSIDE THE BOX (feel free to web search for anything you want)
                        - Formulate your thoughts into a plan that the researcher can research, and the coder can implement
                        - Focus on small iterative changes that can be made to the code
                        - PRIORITIZE FIXING BUGS AND ERRORS THAT ARISE
                        - If needed, revert to a previous version of the player that is avilable in the performance history

                    3. Report (Output)
                        - Create a concise and efficient report with a strategy for the Researcher and the coder
                        - Include One Sentence that goes like "The current strategic advice is to ___, and we will achieve this by ___".
                        - Stick to one or two main points that are the most important
                        - Give Very SPECIFIC Advice...with clear instructinos
                        - Include anything you learned from your tools calls
                        - Give clear instructions


                    You Have the Following Tools at Your Disposal:
                        - read_local_file: Read the content of a catanatron file, or a performance history file in the run directory. (DO NOT CALL MORE THAN TWICE)
                        - read_full_performance_history: Return the entire performance history as a JSON string.
                        - read_foo: Read the content of {FOO_TARGET_FILENAME}.
                        - web_search_tool_call: Perform a web search using the Tavily API.

                    KEEP YOUR TOOL CALLS TO A MINIMUM!
                    YOU ARE LIMITED TO {MAX_MESSAGES_TOOL_CALLING} TOOL CALLS
                    Make sure to start your output with 'STRATEGY:' and end with 'END STRATEGY'.
                    Respond with No Commentary, just the Strategy.

                """
            )

            # Choose the input based on if coming from analyzer or from validator in graph
            msg = [state["performance_history"], state["analysis"]]

            tools = [read_local_file, read_full_performance_history, read_foo, web_search_tool_call]
            output = tool_calling_state_graph(sys_msg, msg, tools)
            strategy = HumanMessage(content=output["messages"][-1].content)
            return {"strategy": strategy, "tool_calling_messages": output["messages"]}

        def researcher_node(state: CreatorGraphState):
            
            #print("In Researcher Node")
            # Add custom tools for researcher

            sys_msg = SystemMessage(
                content=f"""
                    {multi_agent_prompt} RESEARCHER
                     
                    Task: Digest the analysis from the Strategizer, perform research, and create a detailed solution for the Coder to follow and implement

                    1. Digest
                        - 1nd Message you Receive: Digest the analysis and game summary from the Analyzer.
                        - 2nd Message you Receive: Digest the strategy from the Strategizer
                        - If needed, use the read_foo tool call to view the player to understand the code
                        - If needed, use the read_local_file tool call to view any game files in the performance history (1st message)

                    2. Research
                        - Perform research on the questions from the and action items from the Strategizer
                        - If needed, Use the web_search_tool_call to perform a web search for any questions you have (REALLY BENEFICIAL TO USE THIS)
                        - For fixing references to Catanatron API/Objects, view local game files, for syntax bugs, query the web
                        - For implementing new code, se the list_local_files, and read_local_file to view any game files (which are very helpful for debugging)
                        - Determine why the other player is winning, and how to beat it
                        - PRIORITIZE FIXING BUGS AND ERRORS THAT ARISE


                    3. Report (Output)
                        - Create a concise and efficient report with strategizer questions and answers, your resarch, and plan for the coder
                        - Use FACTS you found from your research to back up your claims
                        - If you are fixing syntax errors, provide the correct syntax
                        - Include anything you learned from your tools calls
                        - Give clear instructions to the coder on what to implement
                        - Include any code snippets that are needed or you discovered to be helpful
                        - Note: The Coder Can Only Read and Write Foo....so give it all the information it needs


                    You Have the Following Tools at Your Disposal:
                        - list_catanatron_files: List all files beneath the Catanatron base directory. (Includes Player, Game, State classes etc.)
                        - read_local_file: Read the content of a catanatron file, or a performance history file in the run directory. (DO NOT CALL MORE THAN TWICE)
                        - read_full_performance_history: Return the entire performance history as a JSON string.
                        - read_foo: Read the content of {FOO_TARGET_FILENAME}.
                        - web_search_tool_call: Perform a web search using the Tavily API.

                    KEEP YOUR TOOL CALLS TO A MINIMUM!
                    YOU ARE LIMITED TO {MAX_MESSAGES_TOOL_CALLING} TOOL CALLS

                    Make sure to start your output with 'SOLUTION:' and end with 'END SOLUTION'.
                    Respond with No Commentary, just the Research.

                """
            )

            # Choose the input based on if coming from analyzer or from validator in graph
            msg = [state["analysis"], state["strategy"]]

            if state["validation"].content == "":
                # If coming from analyzer, use the full_results, analusis, and solution
                print("Researcher Coming from Researcher")
                #msg = [state["full_results"], state["analysis"], state["solution"]]
                msg = [state["analysis"], state["strategy"]]
            else:
                # If coming from validator, usee the coder, test_results, and validation messages
                print("Researcher Coming from Validator")
                #msg = [state["code_additions"], state["test_results"], state["validation"]]
                msg = [state["test_results"], state["validation"]]

            tools = [list_catanatron_files, read_local_file, read_full_performance_history, read_foo, web_search_tool_call]
            output = tool_calling_state_graph(sys_msg, msg, tools)
            solution = HumanMessage(content=output["messages"][-1].content)
            return {"solution": solution, "tool_calling_messages": output["messages"]}

        def coder_node(state: CreatorGraphState):

            #print("In Researcher Node")
            # Add custom tools for researcher

            sys_msg = SystemMessage(
                content=f"""
                    {multi_agent_prompt} CODER
                    
                    Task: Digest at the proposed solution from the Strategizer and Researcher, and implement it into the foo_player.py file.

                    1. Digest 
                        - Digest the solution provided by the Researcher and the Analyzer
                        - OR Digest the Code Problems from the Test Results and the Validator
                        - Digest the current player code given for {FOO_TARGET_FILENAME}
                        - If needed: Utilize the list_local_file and read_local_file to view any game files (Very helpful for debugging!)

                    2. Implement
                        - CALL THE **write_foo** tool call to write the new code to the foo_player.py file with the new code
                        - Use what you learned and write the entire new code for the foo_player.py file
                        - Focus on making sure the code implementes the solution in the most correct way possible
                        - Make Sure to not add backslashes to comments, ONLY OUTPUT VALID PYTHON CODE
                            WRONG:        print(\\'Choosing First Action on Default\\')
                            CORRECT:      print('Choosing First Action on Default')
                        - Give plenty of comments in the code to explain what you are doing, and what you have learned (along with syntax help)
                        - Use print statement to usefully debug the output of the code
                        - DO NOT MAKE UP VARIABLES OR FUNCTIONS RELATING TO THE GAME
                        - Note: You will have multiple of iterations to evolve, so make sure the syntax is correct
                        - PRIORITIZE FIXING BUGS AND ERRORS THAT ARISE
                        - Make sure to follow **python 3.12** syntax!!

                    
                    3. Report (Output)
                        - After you do the write_foo tool call, create a report
                        - Take concise and efficient notes with the additions to the code you made, and why you made them for the validator
                        - Include anything you learned from your tools calls

                    You Have the Following Tools at Your Disposal:
                        - read_foo: Read the content of {FOO_TARGET_FILENAME}. MUST BE CALLED BEFORE the {MAX_MESSAGES_TOOL_CALLING}th tool call
                        - write_foo: Write the content of {FOO_TARGET_FILENAME}. (Make sure to keep imports) Note: print() commands will be visible in view_last_game_results

                    KEEP YOUR TOOL CALLS TO A MINIMUM!
                    YOU ARE LIMITED TO {MAX_MESSAGES_TOOL_CALLING} TOOL CALLS
                    Make sure to start your output with 'CODER' and end with 'END CODER'.                    
                """
            )
           
            # p1_sys_msg = SystemMessage(
            #     content =  
            #     f"""
            #     {multi_agent_prompt} CODER PLANNER
                    
            #         Task: Digest at the proposed solution from the Strategizer and Researcher, and create and outline with #TODO statements in the code

            #         1. Digest 
            #             - Digest the solution provided by the Researcher and the Analyzer
            #             - OR Digest the Code Problems from the Test Results and the Validator
            #             - Digest the current player code given for {FOO_TARGET_FILENAME}
            #             - If needed: Utilize the list_local_file and read_local_file to view any game files (Very helpful for debugging!)

            #         2. Implement
            #             - CALL THE **write_foo** tool call to write the new code to the foo_player.py file with the new code
            #              -ONLY WRITE PLACE HOLDERS AND COMMENTS FOR THE CODE. DO NOT ACTUAL IMPLEMENT ANYTHING
            #             - Focus on making sure the code implementes the solution in the most correct way possible
            #             - Make Sure to not add backslashes to comments, ONLY OUTPUT VALID PYTHON CODE
            #                 WRONG:        print(\\'Choosing First Action on Default\\')
            #                 CORRECT:      print('Choosing First Action on Default')
            #             - Give plenty of comments in the code to explain what you are doing, and what you have learned (along with syntax help)
            #             - Use print statement to usefully debug the output of the code
            #             - DO NOT MAKE UP VARIABLES OR FUNCTIONS RELATING TO THE GAME
            #             - Note: You will have multiple of iterations to evolve, so make sure the syntax is correct
            #             - PRIORITIZE FIXING BUGS AND ERRORS THAT ARISE
            #             - Make sure to follow **python 3.12** syntax!!

                    
            #         3. Report (Output)
            #             - After you do the write_foo tool call, create a report
            #             - Take concise and efficient notes with the additions to the code you made, and why you made them for the validator
            #             - Include anything you learned from your tools calls

            #         You Have the Following Tools at Your Disposal:
            #             - read_foo: Read the content of {FOO_TARGET_FILENAME}. MUST BE CALLED BEFORE the {MAX_MESSAGES_TOOL_CALLING}th tool call
            #             - write_foo: Write the content of {FOO_TARGET_FILENAME}. (Make sure to keep imports) Note: print() commands will be visible in view_last_game_results

            #         KEEP YOUR TOOL CALLS TO A MINIMUM!
            #         YOU ARE LIMITED TO {MAX_MESSAGES_TOOL_CALLING} TOOL CALLS
            #         Make sure to start your output with 'CODER' and end with 'END CODER'.                    
            #     """
            # )
            
            #Choose the input based on if coming from analyzer or from validator in graph
            if state["validation"].content == "":
                # If coming from analyzer, use the full_results, analusis, and solution
                print("Coder Coming from Researcher")
                #msg = [state["full_results"], state["analysis"], state["solution"]]
                msg = [state["full_results"], state["strategy"], state["solution"]]
            else:
                # If coming from validator, usee the coder, test_results, and validation messages
                print("Coder Coming from Validator")
                #msg = [state["code_additions"], state["test_results"], state["validation"]]
                msg = [state["test_results"], state["validation"], state["solution"]]
        
            
            tools = [read_foo, write_foo]

            # Update the Code the first time
            output = tool_calling_state_graph(sys_msg, msg, tools)
            code_additions = HumanMessage(content=output["messages"][-1].content)

            # p2_sys_msg = SystemMessage(
            #     content =  
            #     f"""
            #         {multi_agent_prompt} IMPLEMENTER CODER
                    
            #         Task: Digest at the proposed solution PLANNER CODER, and implement it into the foo_player.py file.

            #         1. Digest 
            #             - Digest the solution from the PLANNER CODER and what has been outlined for you
            #             - OR Digest the Code Problems from the Test Results and the Validator
            #             - Digest the current player code given for {FOO_TARGET_FILENAME}
            #             - If needed: Utilize the list_local_file and read_local_file to view any game files (Very helpful for debugging!)

            #         2. Implement
            #             - CALL THE **write_foo** tool call to write the new code to the foo_player.py file with the new code
            #             - YOUR JOB IS TO IMPLEMENT ALL THE COMMENTS AND PLACEHOLDERS FROM THE PLANNER CODER
            #             - Use what you learned and write the entire new code for the foo_player.py file
            #             - Focus on making sure the code implementes the solution in the most correct way possible
            #             - Make Sure to not add backslashes to comments, ONLY OUTPUT VALID PYTHON CODE
            #                 WRONG:        print(\\'Choosing First Action on Default\\')
            #                 CORRECT:      print('Choosing First Action on Default')
            #             - Give plenty of comments in the code to explain what you are doing, and what you have learned (along with syntax help)
            #             - Use print statement to usefully debug the output of the code
            #             - DO NOT MAKE UP VARIABLES OR FUNCTIONS RELATING TO THE GAME
            #             - Note: You will have multiple of iterations to evolve, so make sure the syntax is correct
            #             - PRIORITIZE FIXING BUGS AND ERRORS THAT ARISE
            #             - Make sure to follow **python 3.12** syntax!!

                    
            #         3. Report (Output)
            #             - After you do the write_foo tool call, create a report
            #             - Take concise and efficient notes with the additions to the code you made, and why you made them for the validator
            #             - Include anything you learned from your tools calls

            #         You Have the Following Tools at Your Disposal:
            #             - read_foo: Read the content of {FOO_TARGET_FILENAME}. MUST BE CALLED BEFORE the {MAX_MESSAGES_TOOL_CALLING}th tool call
            #             - write_foo: Write the content of {FOO_TARGET_FILENAME}. (Make sure to keep imports) Note: print() commands will be visible in view_last_game_results

            #         KEEP YOUR TOOL CALLS TO A MINIMUM!
            #         YOU ARE LIMITED TO {MAX_MESSAGES_TOOL_CALLING} TOOL CALLS
            #         Make sure to start your output with 'CODER' and end with 'END CODER'.                    
            #     """
            # )

            # # update the Code the second time
            # msg.append(code_additions)
            # output = tool_calling_state_graph(p2_sys_msg, msg, tools)
            # code_additions = HumanMessage(content=code_additions.content + output["messages"][-1].content)

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
                        - read_full_performance_history: Return the entire performance history as a JSON string.
                        - read_local_file: Read the content of a catanatron file, or a performance history file in the run directory. (DO NOT CALL MORE THAN TWICE)
                    
                    Note: It is okay if the model is not perfect and is novel. 

                    Your job is to make sure the model works and is correct so it can be tested in a full game.
                    Only return "{val_not_ok_key}" if there is a trivial error that can be fixed easily by the Coder
                    
                    
                    Make sure to start your output with 'VALIDATION:' and end with 'END VALIDATION'. 
                    Respond with No Commentary, just the Validation.
                    
                """
            )
            msg_code = HumanMessage(content=read_foo())
            msg = [state["solution"], state["code_additions"], msg_code, state["test_results"]]
            tools = [read_local_file, read_full_performance_history]
            output = tool_calling_state_graph(sys_msg, msg, tools)
            validation = HumanMessage(content=output["messages"][-1].content)

            validator_counter = state["validator_counter"] - 1

            return {"validation": validation, "tool_calling_messages": output["messages"], "validator_counter": validator_counter}
        
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
                return "strategizer"
            else:
                # Default case if neither string is found
                print("Warning: Could not determine validation result, defaulting to strategizer")
                return "strategizer"
            
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
                return "run_player"

            if val_ok_key in validation_message:
                print("Validation passed - rerunning player")
                return "run_player"
            elif val_not_ok_key in validation_message:  #
                print("Validation failed - going back to researcher")
                return "researcher"
            else:
                # Default case if neither string is found
                print("Warning: Could not determine validation result, defaulting to running player")
                return "run_player"

        def construct_graph():
            graph = StateGraph(CreatorGraphState)
            graph.add_node("init", init_node)
            graph.add_node("run_player", run_player_node)
            graph.add_node("analyzer", analyzer_node)
            graph.add_node("strategizer", strategizer_node)
            graph.add_node("researcher", researcher_node)
            graph.add_node("coder", coder_node)
            graph.add_node("test_player", test_player_node)
            graph.add_node("validator", validator_node)
            graph.add_node("summarizer", summarizer_node)

            graph.add_edge(START, "init")
            graph.add_edge("init", "run_player")
            graph.add_edge("run_player", "summarizer")
            graph.add_edge("summarizer", "analyzer")
            graph.add_conditional_edges(
                "analyzer",
                continue_evolving_analyzer,
                {END, "strategizer"}
            )
            #graph.add_edge("analyzer", "researcher")
            graph.add_edge("strategizer", "researcher")
            graph.add_edge("researcher", "coder")
            graph.add_edge("coder", "test_player")
            graph.add_edge("test_player", "validator")
            graph.add_conditional_edges(
                "validator",
                code_ok_validator,
                {"researcher", "run_player"}
            )

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
    



# def view_last_game_llm_query(query_number: int = -1) -> str:
#     """
#     View the game results from a specific run.
    
#     Args:
#         query_number: The index of the file to view (0-based). 
#                      If -1 (default), returns the most recent file.
    
#     Returns:
#         The content of the requested game results file or an error message.
#     """

#     if RUN_TEST_FOO_HAPPENED == False:
#         return "No game run has been executed yet."
    
#     # Path to the runs directory
#     runs_dir = Path(__file__).parent / "runs"
    
#     # Find all folders that start with game_run
#     game_run_folders = [f for f in runs_dir.glob("game_run*") if f.is_dir()]
    
#     if not game_run_folders:
#         return "No game run folders found."
    
#     # Sort folders by name (which includes datetime) to get the most recent one
#     latest_run_folder = sorted(game_run_folders)[-1]
    
#     # Get all files in the folder and sort them
#     result_files = sorted(latest_run_folder.glob("*"))
    
#     if not result_files:
#         return f"No result files found in {latest_run_folder.name}."
    
#     # Determine which file to read
#     file_index = query_number if query_number >= 0 else len(result_files) - 1
    
#     # Check if index is valid
#     if file_index >= len(result_files):
#         return f"Invalid file index. There are only {len(result_files)} files (0-{len(result_files)-1})."
    
#     target_file = result_files[file_index]
    
#     # Read and return the content of the file
#     try:
#         with open(target_file, "r") as file:
#             return f"Content of {target_file.name}:\n\n{file.read()}"
#     except Exception as e:
#         return f"Error reading file {target_file.name}: {str(e)}"
    
# # def view_last_game_results(_: str = "") -> str:
#     """
#     View the game results from a specific run.
    
#     Returns:
#         The content of the requested game results file or an error message.
#     """

#     if RUN_TEST_FOO_HAPPENED == False:
#         return "No game run has been executed yet."
    
#     # Path to the runs directory
#     runs_dir = Path(__file__).parent / "runs"
    
#     # Find all folders that start with game_run
#     game_run_folders = [f for f in runs_dir.glob("game_run*") if f.is_dir()]
    
#     if not game_run_folders:
#         return "No game run folders found."
    
#     # Sort folders by name (which includes datetime) to get the most recent one
#     latest_run_folder = sorted(game_run_folders)[-1]

#     # Read a file with the stdout and stderr called catanatron_output.txt
#     output_file_path = latest_run_folder / "catanatron_output.txt"
    
#     # Read and return the content of the file
#     try:
#         with open(output_file_path, "w") as file:
#             return f"Content of {output_file_path.name}:\n\n{file.read()}"
#     except Exception as e:
#         return f"Error reading file {output_file_path.name}: {str(e)}"
    








         
#     # def create_langchain_react_graph(self):
#     #     """Create a react graph for the LLM to use."""
        
#     #     tools = [
#     #         list_local_files,
#     #         read_local_file,
#     #         read_foo,
#     #         write_foo,
#     #         run_testfoo,
#     #         web_search_tool_call,
#     #         view_last_game_llm_query,
#     #         view_last_game_results
#     #     ]

#     #     llm_with_tools = self.llm.bind_tools(tools)
        
#     #     def assistant(state: MessagesState):
#     #         return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
#     #     def trim_messages(state: MessagesState):
#     #         # Delete all but the specified most recent messages
#     #         delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-self.num_memory_messages]]
#     #         return {"messages": delete_messages}


#     #     builder = StateGraph(MessagesState)

#     #     # Define nodes: these do the work
#     #     builder.add_node("trim_messages", trim_messages)
#     #     builder.add_node("assistant", assistant)
#     #     builder.add_node("tools", ToolNode(tools))

#     #     # Define edges: these determine how the control flow moves
#     #     builder.add_edge(START, "assistant")
#     #     #builder.add_edge("trim_messages", "assistant")
#     #     builder.add_conditional_edges(
#     #         "assistant",
#     #         # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
#     #         # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
#     #         tools_condition,
#     #     )
#     #     builder.add_edge("tools", "trim_messages")
#     #     builder.add_edge("trim_messages", "assistant")
        
#     #     return builder.compile(checkpointer=MemorySaver())


# def run_testfoo(short_game: bool = False) -> str:
#     """
#     Run one Catanatron match (R vs Agent File) and return raw CLI output.
#     Input: short_game (bool): If True, run a short game with a 30 second timeout.
#     """    
#     MAX_CHARS = 20_000                      

#     try:
#         if short_game:
#             result = subprocess.run(
#                 shlex.split(FOO_RUN_COMMAND),
#                 capture_output=True,
#                 text=True,
#                 timeout=30,
#                 check=False
#             )
#         else:
#             result = subprocess.run(
#                 shlex.split(FOO_RUN_COMMAND),
#                 capture_output=True,
#                 text=True,
#                 timeout=14400,
#                 check=False
#             )
#         stdout_limited  = result.stdout[-MAX_CHARS:]
#         stderr_limited  = result.stderr[-MAX_CHARS:]
#         game_results = (stdout_limited + stderr_limited).strip()
#     except subprocess.TimeoutExpired as e:
#         # Handle timeout case
#         stdout_output = e.stdout or ""
#         stderr_output = e.stderr or ""
#         if stdout_output and not isinstance(stdout_output, str):
#             stdout_output = stdout_output.decode('utf-8', errors='ignore')
#         if stderr_output and not isinstance(stderr_output, str):
#             stderr_output = stderr_output.decode('utf-8', errors='ignore')
#         stdout_limited  = stdout_output[-MAX_CHARS:]
#         stderr_limited  = stderr_output[-MAX_CHARS:]
#         game_results = "Game Ended From Timeout (As Expected).\n\n" + (stdout_limited + stderr_limited).strip()
    
#     # Extract the score from the game results

#     # Create a folder in the Creator.run_dir with EvolveCounter#_FooScore#

#     # Inside the folder, 
#     #   place the game_results.txt file with the game results
#     #   copy the FOO_TARGET_FILE as foo_player.py

#         # Extract the score from the game results
#     def extract_game_stats(results: str):
#         """
#         Extract average VP for the FOO player from game results.
#         Returns the avg_vp as a float.
#         If stats can't be found, returns 0.0
#         """
#         import re
        
#         # Look for the FOO_PLAYER_STATS format we're printing in play.py
#         stats_pattern = r"===== FOO_PLAYER_STATS: wins=\d+, avg_vp=(\d+\.\d+) ====="
#         stats_match = re.search(stats_pattern, results)
        
#         if stats_match:
#             return float(stats_match.group(1))
        
#         # Fallback to looking for stats in the regular table output
#         foo_pattern = r"FooPlayer:BLUE\s*[│|]\s*\d+\s*[│|]\s*(\d+\.\d+)"
#         foo_match = re.search(foo_pattern, results, re.DOTALL)
        
#         if foo_match:
#             return float(foo_match.group(1))
                
#         return 0.0  # Default if no pattern matched
#     # Extract game stats
#     foo_avg_vp = extract_game_stats(game_results)

#     # Create a folder in the Creator.run_dir with EvolveCounter#_FooScore#
#     run_folder_name = f"EvolveN{CreatorAgent.evolve_counter}_AvgVP{foo_avg_vp:.1f}"
#     run_folder_path = Path(CreatorAgent.run_dir) / run_folder_name
#     run_folder_path.mkdir(exist_ok=True)

#     # Inside the folder, 
#     # place the game_results.txt file with the game results
#     game_results_path = run_folder_path / "game_results.txt"
#     with open(game_results_path, "w") as f:
#         f.write(game_results)

#     # copy the FOO_TARGET_FILE as foo_player.py
#     shutil.copy2(
#         FOO_TARGET_FILE.resolve(),
#         run_folder_path / FOO_TARGET_FILENAME
#     )
    
#     # Increment the evolve counter
#     CreatorAgent.evolve_counter += 1

#     # Add a file with the stdout and stderr called catanatron_output.txt
#     # output_file_path = latest_run_folder / "catanatron_output.txt"
#     # with open(output_file_path, "w") as output_file:
#     #     output_file.write(game_results)
        
#     #print(game_results)
#         # limit the output to a certain number of characters
#     return game_results
