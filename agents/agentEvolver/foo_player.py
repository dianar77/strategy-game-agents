import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.state import player_key  # Import helper function to access player key


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only. 
                Defined in in "catanatron/catanatron_core/catanatron/game.py"
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # Debug: Log available actions for analysis
        print("Deciding among playable actions:", playable_actions)

        # Resource costs for building a settlement and a city
        settlement_cost = {'wood': 1, 'brick': 1, 'wheat': 1, 'sheep': 1}
        city_cost = {'wheat': 2, 'ore': 3}

        # Fetch the player's resources using the player_key helper
        player_key_value = player_key(game.state, self.color)
        player_resources = {
            "wood": game.state.player_state[f"{player_key_value}_WOOD_IN_HAND"],
            "brick": game.state.player_state[f"{player_key_value}_BRICK_IN_HAND"],
            "sheep": game.state.player_state[f"{player_key_value}_SHEEP_IN_HAND"],
            "wheat": game.state.player_state[f"{player_key_value}_WHEAT_IN_HAND"],
            "ore": game.state.player_state[f"{player_key_value}_ORE_IN_HAND"],
        }
        print("Current player resources:", player_resources)

        # Helper function to calculate missing resources
        def calculate_missing_resources(cost):
            missing = {r: max(0, cost[r] - player_resources.get(r, 0)) for r in cost}
            return {r: v for r, v in missing.items() if v > 0}  # Filter only missing resources

        # Step 1: Determine if a settlement can be built or resources are missing
        missing_settlement_resources = calculate_missing_resources(settlement_cost)
        can_build_settlement = len(missing_settlement_resources) == 0

        if can_build_settlement:
            print("Sufficient resources to build a settlement. Searching for settlement actions.")
            for action in playable_actions:
                if action.action_type.name == 'BUILD_SETTLEMENT':
                    print("Chosen action: Building a settlement.")
                    return action

        # Step 2: Evaluate missing resources and prioritize actions to address deficits
        if missing_settlement_resources:
            print("Missing resources for a settlement:", missing_settlement_resources)

            for action in playable_actions:
                if action.action_type.name == 'TRADE':
                    print("Chosen action: TRADE to address missing resources.")
                    return action
                elif action.action_type.name == 'USE_DEVELOPMENT_CARD':
                    print("Chosen action: USE_DEVELOPMENT_CARD to address missing resources.")
                    return action

        # Step 3: Default fallback action for non-impactful decisions
        print("No prioritized action found. Defaulting to first available action.")
        return playable_actions[0]