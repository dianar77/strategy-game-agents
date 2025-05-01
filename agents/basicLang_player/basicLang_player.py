import os
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

from catanatron import Player
from catanatron_experimental.cli.cli_players import register_player


from langchain_openai import AzureChatOpenAI
from langchain_mistral import ChatMistralAI
from langgraph.graph import MessagesState, START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver

@register_player("bLang")
class BasicLang(Player):
  def __init__(self, color):
    super().__init__(color)


    # Create LLM Client
    self.client = AzureChatOpenAI(
        model="gpt-4o-mini",
        azure_endpoint="https://gpt-amayuelas.openai.azure.com/",
        api_version = "2024-12-01-preview"
    )
    self.llm_name = "gpt-4o-mini"

    # Create LLM Run Director
    if BasicLang.run_dir is None:
      agent_dir = os.path.dirname(os.path.abspath(__file__))
      runs_dir = os.path.join(agent_dir, "runs")
      os.makedirs(runs_dir, exist_ok=True)
      run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
      BasicLang.run_dir = os.path.join(runs_dir, run_id)
      os.makedirs(BasicLang.run_dir, exist_ok=True)


    self.debug_mode = True





  def decide(self, game, playable_actions):
    """Should return one of the playable_actions.

    Args:
        game (Game): complete game state. read-only.
        playable_actions (Iterable[Action]): options to choose from
    Return:
        action (Action): Chosen element of playable_actions
    """
    # ===== YOUR CODE HERE =====

    if self.debug_mode:
        print(f"Deciding for {self.color} with {len(playable_actions)} actions available")

    # Use LLM to choose an action
    try:
        chosen_action_idx = self._get_llm_decision(game, playable_actions)
        if chosen_action_idx is not None and 0 <= chosen_action_idx < len(playable_actions):
            action = playable_actions[chosen_action_idx]
            if self.debug_mode:
                print(f"LLM chose action {chosen_action_idx}: {self._get_action_description(action)}")

            # Record decision time
            #decision_time = time.time() - start_time
            #self.decision_times.append(decision_time)

            return action
    except Exception as e:
        if self.debug_mode:
            print(f"Error getting LLM decision: {e}")
            print("Falling back to rule-based strategy")

    # Fallback to rule-based selection if API call fails or no API key
    return self._select_action(playable_actions, state)
    
    # As an example we simply return the first action:
    #return playable_actions[0]
    # ===== END YOUR CODE =====


  def _get_llm_decision(self, game, playable_actions) -> Optional[int]:
    """Send game state to OpenAI LLM and get the selected action index.

    Args:
        game_state_text: Formatted game state text to send to the LLM
        num_actions: Number of available actions (for validation)

    Returns:
        int: Index of the selected action, or None if API call fails
    """

    num_actions = len(playable_actions)
    # Compose the prompt (system + user, as a single string)
    prompt = (
        "You are playing Settlers of Catan. Your task is to analyze the game state and choose the best action from the available options.\n\n"
        "Rules:\n"
        "1. Think through your decision step by step, analyzing the game state, resources, and available actions\n"
        "2. Your aim is to WIN. That means 10 victory points.\n"
        "3. Put your final chosen action inside a box like \\boxed{5}\n"
        "4. Your final answer must be a single integer corresponding to the action number\n"
        "5. If you want to create or update your strategic plan, put it in <plan> tags like:\n"
        "   <plan>Build roads toward port, then build settlement at node 13, then focus on city upgrades</plan>\n"
        "6. Analyze the recent resource changes to understand what resources you're collecting effectively\n"
        "7. Think about the next 2-3 turns, not just the immediate action\n\n"
        "Board Understanding Guide:\n"
        "- The RESOURCE & NODE GRID shows hexagonal tiles with their coordinates, resources, and dice numbers\n"
        "- The nodes connected to each tile are listed below each tile\n"
        "- 🔍 marks the robber's location, blocking resource production on that hex\n"
        "- Settlements/cities and their production are listed in the BUILDINGS section\n"
        "- Understanding the connectivity between nodes is crucial for road building strategy\n"
        "- Ports allow trading resources at better rates (2:1 or 3:1)\n\n"
        "Here is the current game state:\n\n"
        f"{game}\n\n"
        "Here are your available actions:\n"
        f"{playable_actions}\n\n"
        f"Based on this information, which action number do you choose? Think step by step about your options, then put the final action number in a box like \\boxed{{1}}."
    )

    try:
      #response = self.llm.query(prompt)

      msg = HumanMessage(content=prompt)
      response = self.client.invoke([msg]).content

      # # Get the root directory (project root)
      # agent_dir = os.path.dirname(os.path.abspath(__file__))
      # runs_dir = os.path.join(agent_dir, "runs")

      # # Create runs directory if it doesn't exist
      # os.makedirs(runs_dir, exist_ok=True)

      # # Create a unique subdirectory for this run
      # run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
      # run_dir = os.path.join(runs_dir, run_id)
      # os.makedirs(run_dir, exist_ok=True)

      # # Use the model name for the log file
      # log_path = os.path.join(run_dir, f"llm_log_{self.llm_name}.txt")

      # # Now write your log as before
      # with open(log_path, "a") as log_file:
      #     log_file.write("PROMPT:\n")
      #     log_file.write(prompt + "\n")
      #     log_file.write("RESPONSE:\n")
      #     log_file.write(str(response) + "\n")
      #     log_file.write("="*40 + "\n")

      log_path = os.path.join(BasicLang.run_dir, f"llm_log_{self.llm_name}.txt")
      with open(log_path, "a") as log_file:
          log_file.write("PROMPT:\n")
          log_file.write(prompt + "\n")
          log_file.write("RESPONSE:\n")
          log_file.write(str(response) + "\n")
          log_file.write("="*40 + "\n")

      # Extract the first integer from a boxed answer or any number
      import re
      boxed_match = re.search(r'\\boxed\{(\d+)\}', str(response))
      if boxed_match:
          idx = int(boxed_match.group(1))
          if 0 <= idx < num_actions:
              return idx
      # Fallback: look for any number
      numbers = re.findall(r'\d+', str(response))
      if numbers:
          idx = int(numbers[0])
          if 0 <= idx < num_actions:
              return idx
    except Exception as e:
        if self.debug_mode:
            print(f"Error calling LLM: {e}")
    return None