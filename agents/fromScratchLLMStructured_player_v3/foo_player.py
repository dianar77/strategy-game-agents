import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color

class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
                Defined in "catanatron/catanatron_core/catanatron/game.py"
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # ===== YOUR CODE HERE =====

        def evaluate_action(action):
            """Evaluate the score of an action based on its type and feasibility."""
            action_str = str(action)
            # Assign scores based on action type
            if 'build_settlement' in action_str:
                return 100  # Highly prioritize settlements
            elif 'build_road' in action_str:
                return 50  # Prioritize roads for expansion
            elif 'build_city' in action_str:
                return 75  # Cities provide additional resources
            elif 'buy_development_card' in action_str:
                return 25  # Development cards can provide Victory Points
            else:
                return 10  # Default low priority for other actions

        # Add logic to choose actions based on scores and resources
        best_action = max(playable_actions, key=evaluate_action, default=None)

        if best_action:
            print(f"Choosing strategic action: {best_action}")
            return best_action

        # Default to the first action if no meaningful choice is available
        print("No strategic action identified, defaulting.")
        return playable_actions[0]

        # ===== END YOUR CODE =====