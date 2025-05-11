import os
import json
from catanatron import Player
from catanatron.game import Game
from agents.fromScratchLLMStructured_player.llm_tools import LLM


class FooPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name)
        self.llm = LLM()  # Integrates the LLM for strategic decision-making.

    def serialize_game_state(self, game_state):
        """
        Extract and serialize key aspects of the game state into a simplified format
        that the LLM can understand.

        Args:
            game_state (Game): The complete game state (read-only).

        Returns:
            dict: A dictionary containing simplified game state information.
        """
        # Example serialization of game state (customize as needed for Catanatron):
        return {
            "turn": game_state.turn,
            "current_player": game_state.current_player.color,
            "player_states": {
                player.color: {
                    "resources": player.resources,
                    "victory_points": player.victory_points,
                    "development_cards": player.development_cards,
                }
                for player in game_state.players
            },
            "board": {  # Represent board state minimally (customize as needed)
                "roads": list(game_state.board.roads),
                "settlements": list(game_state.board.settlements),
                "cities": list(game_state.board.cities),
            },
        }

    def evaluate_action(self, action, game_state):
        """
        Uses the LLM to evaluate an action based on the current game state.
        
        Args:
            action (Action): An individual action from playable_actions.
            game_state (Game): The complete game state (read-only).
            
        Returns:
            score (float): The evaluated score of the action.
        """
        # Preprocess the game state
        simplified_state = self.serialize_game_state(game_state)
        
        prompt = (
            f"Evaluate the following action in the context of the current simplified game "
            f"state for maximizing victory points:\n"
            f"Action: {action}\nGame State: {json.dumps(simplified_state)}\n"
            f"Give the action a score between 0 and 10 based on its potential impact."
        )
        response = self.llm.query_llm(prompt)
        try:
            # Safely parse the response as JSON
            parsed_response = json.loads(response)
            score = float(parsed_response.get("score", 0))  # Extract the "score" if available
        except (json.JSONDecodeError, ValueError, TypeError):
            # Handle malformed or unexpected responses
            score = 0  # Default to 0 if the response is invalid or unparseable
        return score

    def decide(self, game, playable_actions):
        """
        Determines the best action to take based on the current game state
        and available actions.

        Args:
            game (Game): The current game state (read-only).
            playable_actions (Iterable[Action]): List of possible actions to choose from.

        Returns:
            Action: The highest-ranked action from the evaluation.
        """
        if not playable_actions:
            raise ValueError("No actions are available to play.")

        # Evaluate all playable actions and select the best one based on scores.
        print("Evaluating available actions...")
        best_action = None
        best_score = float('-inf')  # Start with a very low score.
        
        for action in playable_actions:
            score = self.evaluate_action(action, game)
            print(f"Action: {action}, Score: {score}")
            if score > best_score:
                best_action = action
                best_score = score

        print(f"Chosen Action: {best_action} with score: {best_score}")
        return best_action