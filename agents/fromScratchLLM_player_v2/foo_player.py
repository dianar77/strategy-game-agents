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
            game (Game): complete game state. read-only. 
                Defined in in "catanatron/catanatron_core/catanatron/game.py"
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # Simplify the prompt to avoid unknown attributes in the game object.
        action_descriptions = "\n".join([f"Action {i}: {action}" for i, action in enumerate(playable_actions)])

        prompt = (
            f"You are an AI agent playing as the {self.color} player in Catan. "
            f"Here are your possible actions:\n{action_descriptions}\n\n"
            f"Select ONE action you will take. Respond with only the action number in the format 'Action X' (e.g., 'Action 2')."
        )

        # Query the LLM for a decision
        llm_response = self.llm.query_llm(prompt)

        # Parse the response to select the action
        try:
            selected_action_index = int(llm_response.strip().split()[1])  # Extract the number (e.g., 'Action 2')
            if 0 <= selected_action_index < len(playable_actions):
                return playable_actions[selected_action_index]
            else:
                raise ValueError("Selected index out of range.")
        except Exception as e:
            print(f"Error interpreting LLM response: {llm_response}. Defaulting to first action. Error: {e}")
            return playable_actions[0] # Default to first action in case of an error