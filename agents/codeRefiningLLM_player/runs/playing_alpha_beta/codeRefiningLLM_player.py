import time
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import base_llm
from base_llm import OpenAILLM, AzureOpenAILLM
from typing import List, Any, Optional
import random
from io import StringIO
from datetime import datetime
from catanatron.models.player import Player
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES

class CodeRefiningLLMPlayer(Player):
    """Strategic GPT-based player designed for Settlers of Catan."""
    def __init__(self, color, llm=None):
        super().__init__(color)
        self.llm = llm if llm else AzureOpenAILLM(model_name="gpt-4o")
        self.is_bot = True
        self.debug_mode = True  # Enable debug mode for tracing decisions

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """Analyze the game state and choose an optimal action."""
        if len(playable_actions) == 1:
            return playable_actions[0]

        state = game.state
        game_state_text = self._format_game_state_for_llm(game, state, playable_actions)

        # Attempt decision-making using LLM
        try:
            chosen_index = self._get_llm_decision(game_state_text, len(playable_actions))
            if chosen_index is not None and 0 <= chosen_index < len(playable_actions):
                if self.debug_mode:
                    print(f"LLM chose action #{chosen_index}: {playable_actions[chosen_index].action_type}")
                return playable_actions[chosen_index]
        except Exception as e:
            if self.debug_mode:
                print(f"Error during LLM decision-making: {e}")

        # Fallback: Rule-based simple strategy
        action = self._rule_based_strategy(game, playable_actions)
        if self.debug_mode:
            print(f"Fallback rule-based choice: {action.action_type}")
        return action

    def _format_game_state_for_llm(self, game: Game, state, playable_actions: List[Action]) -> str:
        """Prepare a compact, LLM-understandable description of the game state."""
        output = StringIO()
        output.write(f"TURN {state.num_turns} | Player: {self.color}\n")
        output.write("AVAILABLE ACTIONS:\n")
        for i, action in enumerate(playable_actions):
            output.write(f"[{i}] {self._get_action_description(action)}\n")
        return output.getvalue()

    def _get_llm_decision(self, game_state_text: str, num_actions: int) -> Optional[int]:
        """Query the LLM to choose an action index."""
        prompt = (
            "Analyze the game scenario and choose the most strategic action index."
            f"\n{game_state_text}\n\nRespond with a valid number in \boxed{{x}}."
        )
        response = self.llm.query(prompt)
        import re
        match = re.search(r"\\boxed\\{(\\d+)\\}", response)
        return int(match.group(1)) if match else None

    def _rule_based_strategy(self, game: Game, playable_actions: List[Action]) -> Action:
        """Simple rule-based fallback strategy."""
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                return action
            if action.action_type == ActionType.BUILD_CITY:
                return action
            if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                return action
            if action.action_type == ActionType.PLAY_KNIGHT_CARD:
                return action
            if action.action_type == ActionType.BUILD_ROAD:
                return action
        return random.choice(playable_actions)  # Default fallback

    def _get_action_description(self, action: Action) -> str:
        """Generate a clear description of an action."""
        action_type = action.action_type.name
        return f"{action_type} at {action.value}" if action.value else action_type