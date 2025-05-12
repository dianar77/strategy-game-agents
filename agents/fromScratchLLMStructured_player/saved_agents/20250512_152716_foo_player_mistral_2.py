import os
from catanatron.models.player import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.board import Board
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES, SETTLEMENT, CITY
from catanatron.models.map import CatanMap, LandTile, Port
from catanatron.models.decks import freqdeck_count
from catanatron.state_functions import (
    get_player_buildings,
    player_key,
    player_num_resource_cards,
    player_num_dev_cards,
    get_player_freqdeck,
    get_longest_road_length,
    get_largest_army,
)
from catanatron.state import State

class FooPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name)

    def decide(self, game, playable_actions):
        print('Deciding action...')
        # Prioritize building settlements and cities
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                print('Choosing to build settlement')
                return action
            if action.action_type == ActionType.BUILD_CITY:
                print('Choosing to build city')
                return action

        # Implement trading logic to exchange excess resources
        for action in playable_actions:
            if action.action_type in [ActionType.MARITIME_TRADE, ActionType.OFFER_TRADE, ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE, ActionType.CONFIRM_TRADE]:
                # Trade excess resources for needed ones
                print('Choosing to trade')
                return action

        # Default to the first action if no better option is found
        print('Choosing default action')
        return playable_actions[0]