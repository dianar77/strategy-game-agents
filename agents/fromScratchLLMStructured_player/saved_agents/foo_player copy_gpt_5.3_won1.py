from catanatron.models.player import Player
from catanatron.models.enums import ActionType, ActionPrompt, SETTLEMENT, CITY
from catanatron.state_functions import (
    get_player_buildings,
    player_num_resource_cards,
    get_longest_road_length,
    get_largest_army,
)
import math

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
        def evaluate_action(action, player_state, game_phase):
            """Score the given action dynamically based on type, resource needs, and game phase."""
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                base_score = 15 if game_phase == "early" else 10
                # Boost if player has sufficient resources
                resource_weight = 5 if player_state["resources"] >= 4 else -3
                return base_score + resource_weight

            elif action.action_type == ActionType.BUILD_CITY:
                priority = 12 if game_phase in ["mid", "late"] else 8
                # Resource availability impacts value
                return priority + (5 if player_state["resources"] >= 8 else -3)

            elif action.action_type == ActionType.BUILD_ROAD:
                score = 10 if game_phase == "early" else 6
                if player_state["longest_road"] >= 4:
                    score += 4  # Incentivize Longest Road pursuit
                return score

            elif action.action_type in [ActionType.MOVE_ROBBER]:
                # Use robber to target opponent strategically
                target_player_color = action.value[1]
                if target_player_color is not None and target_player_color != self.color:
                    target_player = next(
                        (p for p in game.state.players if p.color == target_player_color), None
                    )
                    if target_player:
                        # Estimate opponent priority based on settlements, cities, and resources
                        target_settlements = len(get_player_buildings(game.state, target_player_color, SETTLEMENT))
                        target_cities = len(get_player_buildings(game.state, target_player_color, CITY))
                        target_resources = player_num_resource_cards(game.state, target_player_color)
                        estimated_value = target_settlements + (target_cities * 2) + (target_resources / 2)
                        return 9 if estimated_value >= 8 else 6
                return 2  # Low default score for no target

            elif action.action_type in [
                    ActionType.PLAY_KNIGHT_CARD, ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_MONOPOLY]:
                return 12 if (game_phase == "mid" and player_state["largest_army"][0] == self.color) else 8

            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                return 9 if game_phase == "late" else 6

            elif action.action_type == ActionType.MARITIME_TRADE:
                # Favor Maritime Trade when surplus resources exist and it leads to critical actions
                needed_resources = player_state["needed_for_build"]
                trade_value = 0

                # Extract offered resources and desired resource
                offered_resources = action.value[:4]
                desired_resource = action.value[4]

                # Evaluate trade impact
                for resource in needed_resources:
                    trade_value += 3 if resource == desired_resource else 0

                return 7 + trade_value if trade_value > 0 else 2

            elif action.action_type == ActionType.END_TURN:
                # Minimum fallback, penalize when resources are available for action
                return 1 if player_state["resources"] < 4 else -2

            elif action.action_type == ActionType.ROLL:
                # Default roll action
                return 0  

            return 0

        def analyze_player_state():
            """Analyze the current player state for decision-making."""
            player_resources = player_num_resource_cards(game.state, self.color)
            settlements = get_player_buildings(game.state, self.color, SETTLEMENT)
            cities = get_player_buildings(game.state, self.color, CITY)

            # Calculate needed resources for builds
            needed_for_build = []
            if player_resources < 4:
                needed_for_build.append("settlement")
            if player_resources < 8:
                needed_for_build.append("city")

            return {
                "resources": player_resources,
                "settlements": settlements,
                "cities": cities,
                "longest_road": get_longest_road_length(game.state, self.color),
                "largest_army": get_largest_army(game.state),
                "needed_for_build": needed_for_build
            }

        def get_game_phase(player_state):
            """Determine the current game phase (early, mid, late)."""
            total_victory_points = len(player_state["settlements"]) + len(player_state["cities"]) * 2
            if total_victory_points < 4:
                return "early"
            elif total_victory_points < 7:
                return "mid"
            else:
                return "late"

        # Analyze the game state and the player state
        player_state = analyze_player_state()
        game_phase = get_game_phase(player_state)

        # Score all actions
        scored_actions = [
            (action, evaluate_action(action, player_state, game_phase)) for action in playable_actions
        ]
        scored_actions.sort(key=lambda x: x[1], reverse=True)

        # Debugging print to trace decision-making
        print("Game Phase:", game_phase)
        print("Player State:", player_state)
        print("Playable Actions:", [str(action.action_type) for action in playable_actions])
        print("Scored Actions:", [(str(action.action_type), score) for action, score in scored_actions])

        # Return the best scored action or fallback if no actions scored positively
        return scored_actions[0][0] if scored_actions and scored_actions[0][1] > 0 else playable_actions[0]