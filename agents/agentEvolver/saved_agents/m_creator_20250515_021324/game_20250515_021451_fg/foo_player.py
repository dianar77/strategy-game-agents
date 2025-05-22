import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)

    def decide(self, game, playable_actions):
        # Should return one of the playable_actions.

        # Args:
        #     game (Game): complete game state. read-only.
        #         Defined in in \"catanatron/catanatron_core/catanatron/game.py\"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions

        # ===== YOUR CODE HERE =====
        # As an example we simply return the first action:
        # print("Choosing First Action on Default")
        # return playable_actions[0]
        # ===== END YOUR CODE =====

        # Iterate through the playable actions to find the best one
        for action in playable_actions:
            # Check if the action is to build a city
            if action.action_type == ActionType.BUILD_CITY:
                print('Building a city')
                return action
            # Check if the action is to build a settlement
            elif action.action_type == ActionType.BUILD_SETTLEMENT:
                print('Building a settlement')
                return action
        # If no preferred action is found, return the first action
        print('Choosing First Action on Default')
        return playable_actions[0]