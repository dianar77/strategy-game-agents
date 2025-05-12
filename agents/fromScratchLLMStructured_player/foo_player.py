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
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only. 
                Defined in in "catanatron/catanatron_core/catanatron/game.py"
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # ===== YOUR CODE HERE =====
        # As an example we simply return the first action:
        print("Choosing First Action on Default")
        return playable_actions[0]
        # ===== END YOUR CODE =====

