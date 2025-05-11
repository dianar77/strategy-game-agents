import os
from catanatron import Player
from datetime import datetime
from agents.fromScratchLLM_player_v2.llm_tools import LLM

class FooPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name)
        self.llm = LLM()  # Includes LLM class with llm.query_llm(prompt: str) -> str method

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): Complete game state. read-only.
                Defined in "catanatron/catanatron_core/catanatron/game.py"
            playable_actions (Iterable[Action]): Options to choose from.
        Return:
            action (Action): Chosen element of playable_actions
        """
        # Structure the prompt for the LLM
        state_summary = self._summarize_game_state(game)
        actions_summary = self._summarize_actions(playable_actions)

        prompt = f"""
        You are playing Settlers of Catan. Based on the following game state and available actions, choose the best action:

        Game State Summary:
        {state_summary}

        Playable Actions:
        {actions_summary}

        Respond with the index of the best action (starting from 0).
        """

        # Query the LLM with the structured prompt
        try:
            response = self.llm.query_llm(prompt)
            print(f"LLM Response: {response}")
            # Extract the index (fallback for verbose responses)
            chosen_index = self._extract_index_from_response(response)
            return playable_actions[chosen_index]
        except Exception as e:
            print(f"Error querying LLM or processing response: {e}")
            print("Defaulting to the first action.")
            return playable_actions[0]

    def _summarize_game_state(self, game):
        """Summarizes the current game state for the LLM prompt."""
        player_state = game.state.player_state
        print("Debug: player_state keys ->", player_state.keys())  # Debugging output

        # Identify this player
        player_index = game.state.colors.index(self.color)

        summary = {
            "current_player": self.color.name,
            "victory_points_to_win": game.vps_to_win,
            "turn_number": game.state.num_turns,
            "players": {
                f"P{index}": {
                    "victory_points": player_state.get(f"P{index}_ACTUAL_VICTORY_POINTS", "N/A"),
                    "resources": {
                        "WOOD": player_state.get(f"P{index}_WOOD_IN_HAND", 0),
                        "BRICK": player_state.get(f"P{index}_BRICK_IN_HAND", 0),
                        "SHEEP": player_state.get(f"P{index}_SHEEP_IN_HAND", 0),
                        "WHEAT": player_state.get(f"P{index}_WHEAT_IN_HAND", 0),
                        "ORE": player_state.get(f"P{index}_ORE_IN_HAND", 0),
                    },
                    "is_current_player": index == player_index
                }
                for index in range(len(game.state.colors))
            }
        }
        return str(summary)

    def _summarize_actions(self, playable_actions):
        """Summarizes the list of playable actions for the LLM prompt."""
        return [str(action) for action in playable_actions]

    def _extract_index_from_response(self, response):
        """Extracts an index from the LLM response, with robust fallback."""
        try:
            # Attempt to parse as an integer directly
            return int(response.strip())
        except ValueError:
            # Fallback: search for patterns like 'Index X' or 'Best Action Index: X'
            import re
            match = re.search(r'\b(\d+)\b', response)
            if match:
                return int(match.group(1))
            # Default to the first action if parsing fails
            print("Could not parse LLM response as action index. Defaulting to 0.")
            return 0