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

class CodeRefiningLLMPlayer(Player):
    """Optimized player for Catan using a resource-focused strategy."""

    def __init__(self, color, name=None):
        super().__init__(color, name)
        self.current_plan = "Ore-Wheat strategy for cities and development cards"

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        state = game.state

        # Prioritize actions based on strategy
        priorities = {
            ActionType.BUILD_CITY: 1,
            ActionType.BUILD_SETTLEMENT: 2,
            ActionType.BUY_DEVELOPMENT_CARD: 3,
            ActionType.BUILD_ROAD: 4,
        }

        for action_type in sorted(priorities.keys(), key=lambda x: priorities[x]):
            for action in playable_actions:
                if action.action_type == action_type:
                    return action

        # Log and fallback in case of no valid action
        if not playable_actions:
            print("No available actions. Ending turn.")
            return Action(ActionType.END_TURN, value=None)

        return playable_actions[0]

    def _format_game_state_for_llm(self, game: Game, state: State, playable_actions: List[Action]) -> str:
        # Simplified game state representation for intuitive decision-making
        resources = get_player_freqdeck(state, self.color)
        return f"Resources: {resources}, Actions: {[action.action_type.name for action in playable_actions]}"

    def _format_cost(self, cost_type):
        costs = {
            "ROAD": {"WOOD": 1, "BRICK": 1},
            "SETTLEMENT": {"WOOD": 1, "BRICK": 1, "WHEAT": 1, "SHEEP": 1},
            "CITY": {"ORE": 3, "WHEAT": 2},
        }
        return costs.get(cost_type, {})