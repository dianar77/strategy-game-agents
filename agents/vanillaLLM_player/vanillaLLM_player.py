import time
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from base_llm import AzureOpenAILLM
from typing import List, Optional
import random

from catanatron.models.player import Player
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES, SETTLEMENT, CITY
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

class VanillaLLMPlayer(Player):
    """Simplified LLM-powered player for Catan game decisions."""
    
    def __init__(self, color, name=None, llm=None):
        super().__init__(color, name)
        if llm is None:
            self.llm = AzureOpenAILLM(model_name="gpt-4o")
        else:
            self.llm = llm
        self.is_bot = True
        self.llm_name = self.llm.model

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """Use LLM to analyze game state and choose an action."""
        
        if len(playable_actions) == 1:
            return playable_actions[0]
            
        # Create a string representation of the game state
        game_state_text = self._format_game_state_for_llm(game, game.state, playable_actions)
        
        # Use LLM to choose an action
        try:
            chosen_action_idx = self._get_llm_decision(game_state_text, len(playable_actions))
            if chosen_action_idx is not None and 0 <= chosen_action_idx < len(playable_actions):
                return playable_actions[chosen_action_idx]
        except Exception as e:
            print(f"Error getting LLM decision: {e}")
            
        # Fallback to simple selection if API call fails
        return random.choice(playable_actions)

    def _format_game_state_for_llm(self, game: Game, state: State, playable_actions: List[Action]) -> str:
        """Basic game state formatting without pretty printing."""
        from io import StringIO
        output = StringIO()
        
        # Game basic info
        output.write(f"TURN {state.num_turns} | Player: {state.current_color().name} | Action: {state.current_prompt.name}\n\n")
        
        # Board state
        board = state.board
        output.write("BOARD STATE:\n")
        output.write(f"- Robber at: {board.robber_coordinate}\n")
        output.write(f"- Longest Road: {board.road_color.name if board.road_color else 'None'} ({board.road_length} segments)\n")
        
        # Simple list of tiles with resources and numbers
        output.write("- Tiles:\n")
        for coord, tile in board.map.land_tiles.items():
            resource = tile.resource if tile.resource else "DESERT"
            number = tile.number if tile.number is not None else "-"
            output.write(f"  {coord}: {resource} ({number})\n")
        
        # Buildings (settlements and cities)
        output.write("- Buildings:\n")
        for node_id, (color, building_type) in board.buildings.items():
            output.write(f"  {node_id}: {color.name} {building_type}\n")
        
        # Roads
        output.write("- Roads:\n")
        for edge, color in board.roads.items():
            output.write(f"  {edge}: {color.name}\n")
        
        # Players
        output.write("\nPLAYERS:\n")
        for color in state.colors:
            hand = get_player_freqdeck(state, color)
            key = player_key(state, color)
            vp = state.player_state.get(f"{key}_VICTORY_POINTS", 0)
            is_current = color == state.current_color()
            
            output.write(f"- {color.name} Player:\n")
            output.write(f"  Victory Points: {vp}/10\n")
            
            # Resources - only show for current player
            if is_current:
                output.write("  Resources:\n")
                for i, resource in enumerate(RESOURCES):
                    count = hand[i] if i < len(hand) else 0
                    if count > 0:
                        output.write(f"    {resource}: {count}\n")
            else:
                output.write(f"  Resources: {sum(hand)} cards (hidden)\n")
            
            # Development cards - only show details for current player
            if is_current:
                dev_cards = state.player_state.get(f"{key}_DEVELOPMENT_CARDS", {})
                output.write("  Development Cards:\n")
                for card_type, count in dev_cards.items():
                    if count > 0:
                        output.write(f"    {card_type}: {count}\n")
            
            # Building counts
            settlements = get_player_buildings(state, color, SETTLEMENT)
            cities = get_player_buildings(state, color, CITY)
            roads_available = state.player_state.get(f'{key}_ROADS_AVAILABLE', 0)
            roads_built = 15 - roads_available
            
            output.write(f"  Buildings:\n")
            output.write(f"    Settlements: {len(settlements)}\n")
            output.write(f"    Cities: {len(cities)}\n")
            output.write(f"    Roads: {roads_built}\n")
        
        # Available actions
        output.write("\nAVAILABLE ACTIONS:\n")
        for i, action in enumerate(playable_actions):
            value_str = str(action.value) if action.value is not None else ""
            output.write(f"[{i}] {action.action_type.name}: {value_str}\n")
        
        return output.getvalue()

    def _get_llm_decision(self, game_state_text: str, num_actions: int) -> Optional[int]:
        """Get action choice from LLM."""
        prompt = (
            "You are playing Settlers of Catan.\n\n"
            f"{game_state_text}\n\n"
            f"Choose the best action number from the options. Respond with just the number in a box like \\boxed{{1}}."
        )

        try:
            response = self.llm.query(prompt)
            
            # Log request/response
            model_name = self.llm_name
            safe_model_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
            log_path = os.path.join(os.path.dirname(__file__), f"llm_log_{safe_model_name}.txt")
            with open(log_path, "a") as log_file:
                log_file.write("PROMPT:\n")
                log_file.write(prompt + "\n")
                log_file.write("RESPONSE:\n")
                log_file.write(str(response) + "\n")
                log_file.write("="*40 + "\n")
            
            # Extract action number
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
            print(f"Error calling LLM: {e}")
            
        return None