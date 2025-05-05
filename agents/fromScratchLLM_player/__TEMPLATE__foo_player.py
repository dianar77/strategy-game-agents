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
        # ===== YOUR CODE HERE =====
        # As an example we simply return the first action:
        return playable_actions[0]
        # ===== END YOUR CODE =====

