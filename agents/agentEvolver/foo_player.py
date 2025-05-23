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
        #                  Defined in "catanatron/catanatron_core/catanatron/game.py"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions
        
        # 1) If BUILD_CITY is an option, choose that
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_CITY:
                print('Choosing to build a city first')
                return action

        # 2) Otherwise, if BUILD_SETTLEMENT is available, choose that
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                print('Choosing to build a settlement')
                return action

        # 3) Otherwise, choose the first available action as fallback
        print('No city or settlement available, fallback to first action')
        return playable_actions[0]
