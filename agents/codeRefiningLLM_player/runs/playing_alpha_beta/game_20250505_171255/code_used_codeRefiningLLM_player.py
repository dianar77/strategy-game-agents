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

from catanatron.models.player import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.board import Board
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES, SETTLEMENT, CITY
from catanatron.models.map import CatanMap, LandTile, Port
from catanatron.models.decks import freqdeck_count
from catanatron.state_functions import (
    get_player_buildings,
    player_key,
    player_num_resource_cards,
    player_num_dev_cards,
    get_player_freqdeck,
    get_longest_road_length,
    get_largest_army,
)
from catanatron.state import State

# Constants for pretty printing
RESOURCE_EMOJI = {
    "WOOD": "🌲",
    "BRICK": "🧱",
    "SHEEP": "🐑",
    "WHEAT": "🌾",
    "ORE": "⛏️",
    None: "🏜️",
}

BUILDING_EMOJI = {
    "SETTLEMENT": "🏠",
    "CITY": "🏙️",
    "ROAD": "🛣️",
}

COSTS = {
    "ROAD": {"WOOD": 1, "BRICK": 1},
    "SETTLEMENT": {"WOOD": 1, "BRICK": 1, "WHEAT": 1, "SHEEP": 1},
    "CITY": {"WHEAT": 2, "ORE": 3},
    "DEVELOPMENT_CARD": {"SHEEP": 1, "WHEAT": 1, "ORE": 1},
}

DEV_CARD_DESCRIPTIONS = {
    "KNIGHT": "Move the robber and steal a card from a player adjacent to the new location",
    "YEAR_OF_PLENTY": "Take any 2 resources from the bank",
    "MONOPOLY": "Take all resources of one type from all other players",
    "ROAD_BUILDING": "Build 2 roads for free",
    "VICTORY_POINT": "Worth 1 victory point",
}

class CodeRefiningLLMPlayer(Player):
    """LLM-powered player that uses GPT-based API to make Catan game decisions."""
    debug_mode = True
    run_dir = None

    def __init__(self, color, name=None, llm=None):
        super().__init__(color, name)
        if llm is None:
            self.llm = AzureOpenAILLM(model_name="gpt-4o")
        else:
            self.llm = llm
        self.is_bot = True
        self.llm_name = self.llm.model

        # Setup run directory
        if CodeRefiningLLMPlayer.run_dir is None:
            if "CATAN_CURRENT_RUN_DIR" in os.environ:
                CodeRefiningLLMPlayer.run_dir = os.environ["CATAN_CURRENT_RUN_DIR"]
            else:
                agent_dir = os.path.dirname(os.path.abspath(__file__))
                runs_dir = os.path.join(agent_dir, "runs")
                os.makedirs(runs_dir, exist_ok=True)
                run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
                CodeRefiningLLMPlayer.run_dir = os.path.join(runs_dir, run_id)
        self.api_calls = 0
        self.decision_times = []

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """Use the LLM to analyze the game state and choose an action."""
        state = game.state

        if len(playable_actions) == 1:
            return playable_actions[0]

        game_state_text = self._format_game_state_for_llm(game, state, playable_actions)

        try:
            chosen_action_idx = self._get_llm_decision(game_state_text, len(playable_actions))
            if chosen_action_idx is not None and 0 <= chosen_action_idx < len(playable_actions):
                return playable_actions[chosen_action_idx]
        except Exception:
            return random.choice(playable_actions)

    def _format_game_state_for_llm(self, game: Game, state: State, playable_actions: List[Action]) -> str:
        """Generate a concise textual representation of the current game state."""
        output = StringIO()
        output.write(f"TURN {state.num_turns} | Action: {state.current_prompt.name}\n")
        output.write("AVAILABLE ACTIONS:\n")
        for i, action in enumerate(playable_actions):
            desc = self._get_action_description(action)
            output.write(f"[{i}] {desc}\n")
        return output.getvalue()

    def _get_llm_decision(self, game_state_text: str, num_actions: int) -> Optional[int]:
        """Pass the game state to the LLM and get the selected action index."""
        prompt = "Analyze the game and choose the best action:\n" + game_state_text + "\nResponse should be: \boxed{action_index}"
        response = self.llm.query(prompt)
        import re
        match = re.search(r"\\boxed\{(\d+)\}", response)
        return int(match.group(1)) if match else None

    def _get_action_description(self, action: Action) -> str:
        """Generate a human-readable description for an action."""
        if action.action_type == ActionType.BUILD_ROAD:
            return f"Build a road at {action.value}"
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            return f"Build a settlement at {action.value}"
        elif action.action_type == ActionType.BUILD_CITY:
            return f"Upgrade settlement at {action.value}"
        # Add additional descriptions as needed
        return action.action_type.name
