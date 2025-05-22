from catanatron.state_functions import player_resource_freqdeck_contains
from catanatron.models.decks import SETTLEMENT_COST_FREQDECK, CITY_COST_FREQDECK
from catanatron.models.enums import ActionType

class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)

    def decide(self, game, playable_actions):
        state = game.state
        color = self.color
        player_index = state.color_to_index[color]

        # Iterate through the playable actions to find the best one
        for action in playable_actions:
            # Check if the action is to build a settlement
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                # Check if the player has enough resources to build a settlement
                if player_resource_freqdeck_contains(state, color, SETTLEMENT_COST_FREQDECK) and state.player_state[f'P{player_index}_SETTLEMENTS_AVAILABLE'] > 0:
                    print('Building a settlement')
                    return action
            # Check if the action is to build a city
            elif action.action_type == ActionType.BUILD_CITY:
                # Check if the player has enough resources to build a city
                if player_resource_freqdeck_contains(state, color, CITY_COST_FREQDECK) and state.player_state[f'P{player_index}_CITIES_AVAILABLE'] > 0:
                    print('Building a city')
                    return action
            # Check if the action is to trade
            elif action.action_type in [ActionType.MARITIME_TRADE, ActionType.OFFER_TRADE, ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE, ActionType.CONFIRM_TRADE, ActionType.CANCEL_TRADE]:
                # Implement advanced trading logic to acquire needed resources
                if self.should_trade(state, color):
                    print('Trading resources')
                    return action
        # If no preferred action is found, return the first action
        print('Choosing First Action on Default')
        return playable_actions[0]

    def should_trade(self, state, color):
        # Implement advanced logic to determine if trading is beneficial
        # For example, check if the player has excess resources and needs specific resources to build
        player_resources = state.player_state[f'P{state.color_to_index[color]}_RESOURCES']
        needed_resources = {
            'brick': 2,
            'wood': 2,
            'sheep': 2,
            'grain': 2,
            'ore': 2
        }
        # Check if the player has excess resources and needs specific resources
        for resource, needed_amount in needed_resources.items():
            if player_resources.get(resource, 0) < needed_amount:
                return True
        return False