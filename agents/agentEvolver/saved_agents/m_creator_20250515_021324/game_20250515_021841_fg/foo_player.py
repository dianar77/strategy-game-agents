import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.enums import ActionType
from catanatron.state_functions import player_resource_freqdeck_contains
from catanatron.models.decks import SETTLEMENT_COST_FREQDECK, CITY_COST_FREQDECK


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

        state = game.state
        color = self.color
        player_index = state.color_to_index[color]

        # Iterate through the playable actions to find the best one
        for action in playable_actions:
            # Check if the action is to build a city
            if action.action_type == ActionType.BUILD_CITY:
                # Check if the player has enough resources to build a city
                if player_resource_freqdeck_contains(state, color, CITY_COST_FREQDECK) and state.player_state[f'P{player_index}_CITIES_AVAILABLE'] > 0:
                    print('Building a city')
                    return action
            # Check if the action is to build a settlement
            elif action.action_type == ActionType.BUILD_SETTLEMENT:
                # Check if the player has enough resources to build a settlement
                if player_resource_freqdeck_contains(state, color, SETTLEMENT_COST_FREQDECK) and state.player_state[f'P{player_index}_SETTLEMENTS_AVAILABLE'] > 0:
                    print('Building a settlement')
                    return action
        # If no preferred action is found, return the first action
        print('Choosing First Action on Default')
        return playable_actions[0]