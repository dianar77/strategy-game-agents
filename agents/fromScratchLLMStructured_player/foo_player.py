import os
from catanatron.models.player import Player
from catanatron.models.enums import ActionType
from catanatron.models.decks import freqdeck_count

class FooPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name)

    def get_opponent_color(self, state):
        """
        Infer opponent color from the known player info in the game state.
        """
        for player in state.players:
            if player.color != self.color:
                return player.color
        return None

    def score_action(self, action, game):
        state = game.state
        try:
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                settlement_bonus = sum(getattr(tile, 'resource_probability', 0) for tile in state.board.map.get_adjacent_land_tiles(action.value)) if action.value else 0
                return 10 + settlement_bonus

            elif action.action_type == ActionType.BUILD_CITY:
                return 12  # Cities provide advanced resource generation potential

            elif action.action_type == ActionType.BUILD_ROAD:
                opponent_color = self.get_opponent_color(state)
                road_bonus = 5 + (5 if state.board.map.is_adjacent_to_player(action.value, opponent_color) else 0)
                return road_bonus if action.value else 0

            elif action.action_type == ActionType.MARITIME_TRADE:
                if action.value and len(action.value) == 2:
                    give_resources, receive_resource = action.value
                    return freqdeck_count(state.resource_freqdeck.get(receive_resource, 0)) + (15 - 4 * len(give_resources))
                return 0

            elif action.action_type == ActionType.MOVE_ROBBER:
                if action.value and len(action.value) >= 2:
                    target_hex, target_player = action.value[:2]
                    opponent_color = self.get_opponent_color(state)
                    retaliation_penalty = -5 if target_player == opponent_color else 0
                    return freqdeck_count(state.resource_freqdeck.get(target_hex.resource, 0)) + getattr(target_hex, 'probability', 0) + retaliation_penalty
                return 0

            elif action.action_type == ActionType.END_TURN:
                return 0  # Lowest priority action

            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                return 3  # Moderate scoring for potential long-term victory points

            print(f"Unknown ActionType encountered: {action.action_type}")
            return -1

        except Exception as e:
            print(f"Error scoring action {action}: {e}")
            return -1

    def decide(self, game, playable_actions):
        scored_actions = [(action, self.score_action(action, game)) for action in playable_actions]
        print(f"Evaluating actions: {scored_actions}")
        max_score = max(scored_actions, key=lambda x: x[1])[1]
        tied_actions = [action for action, score in scored_actions if score == max_score]

        best_action = tied_actions[0] if len(tied_actions) == 1 else self.tie_break(game, tied_actions)

        print(f"Chosen action: {best_action}")
        return best_action

    def tie_break(self, game, tied_actions):
        def secondary_score(action):
            try:
                state = game.state
                opponent_color = self.get_opponent_color(state)

                if action.action_type == ActionType.BUILD_ROAD:
                    return 10 if state.board.map.is_adjacent_to_player(action.value, opponent_color) else 0

                elif action.action_type == ActionType.MARITIME_TRADE:
                    if action.value and len(action.value) == 2:
                        _, receive_resource = action.value
                        return freqdeck_count(state.resource_freqdeck.get(receive_resource, 0))
                    return 0

                elif action.action_type == ActionType.MOVE_ROBBER:
                    if action.value and len(action.value) >= 1:
                        target_hex = action.value[0]
                        return getattr(target_hex, 'probability', 0)
                    return 0

                elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                    return 3  # Opportunity to gain special cards or victory points

                print(f"Undefined ActionType during tie-breaking: {action.action_type}")
                return -1
            except Exception as e:
                print(f"Error during secondary scoring: {e}")
                return -1

        return max(tied_actions, key=secondary_score)