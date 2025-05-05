import os
from catanatron import Player
from datetime import datetime
from agents.fromScratchLLM_player.llm_tools import LLM

class FooPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name)
        self.llm = LLM()  # Includes LLM class with llm.llm_query(prompt: str) -> str method


    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # ===== Corrected Code =====
        # Extracting game state information from the `state` object
        state = game.state
        game_summary = {
            "num_turns": state.num_turns,
            "current_player": str(state.current_color()),
            "resources": {color: state.player_state[f"{color}_RESOURCES"] for color in state.colors},
            "victory_points": {color: state.player_state[f"{color}_ACTUAL_VICTORY_POINTS"] for color in state.colors},
        }
        
        prompt = (
            "You are an expert Catan player. Analyze the current game state and suggest the top action that contributes to victory. "
            "Consider maximizing resource accumulation, strategic positioning, blocking opponents, and scoring victory points. "
            f"\nGame summary: {game_summary}"
            f"\nPlayable actions: {[str(action) for action in playable_actions]}"
            "\nRespond with one recommended action in this format: 'Recommended action: <action_string>'."
        )

        llm_response = self.llm.query_llm(prompt)
        
        # Extract the recommended action from the LLM response
        recommended_action = None
        if "Recommended action:" in llm_response:
            recommended_action = llm_response.split("Recommended action:")[1].strip()

        # Validate and return the recommended action if it's in the list of playable actions
        for action in playable_actions:
            if str(action) == recommended_action:
                return action

        # Default fallback (should not be reached unless LLM fails)
        return playable_actions[0]
        # ===== END Corrected Code =====