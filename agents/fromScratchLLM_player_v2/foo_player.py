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
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # Prepare game details for the LLM
        try:
            game_description = f"Turn: {game.turn}, Phase: {game.phase}, Dice: {game.dice}, Resources: {game.resources}, Roads: {game.roads}, Settlements: {game.settlements}, Cities: {game.cities}"
        except AttributeError:
            game_description = "Some game details could not be retrieved."

        # Construct the prompt
        prompt = ("You are a Catan player. The current game state is as follows: \n"
                  f"{game_description}\n"
                  "The available actions are listed below. Each action is indexed numerically:\n"
                  f"{[(i, str(action)) for i, action in enumerate(playable_actions)]}\n"
                  "Choose the index of the best action to play, considering the goal to win Catan.")

        # Query the LLM for decision-making
        llm_response = self.llm.query_llm(prompt)

        # Try to parse the LLM response and choose the appropriate action
        try:
            print("LLM response:", llm_response)
            action_index = int(llm_response.strip())  # Expecting an index from the LLM
            if 0 <= action_index < len(playable_actions):
                return playable_actions[action_index]
        except (ValueError, IndexError):
            print("Invalid LLM response. Choosing the first action as a fallback.")

        # Fallback to the first action
        return playable_actions[0]