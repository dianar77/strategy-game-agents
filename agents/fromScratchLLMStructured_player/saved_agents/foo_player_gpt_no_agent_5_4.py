import os
from catanatron import Player
from catanatron.game import Game
from catanatron.state_functions import player_key
from catanatron.models.enums import ActionType, WOOD, BRICK, SHEEP, WHEAT, ORE
from agents.fromScratchLLMStructured_player.llm_tools import LLM

class FooPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name)
        self.llm = LLM()  # Includes LLM class with llm.query_llm(prompt: str) -> str method
        self.repetitive_actions = {ActionType.END_TURN: 0, ActionType.ROLL: 0}

    def heuristic_score_action(self, action, game):
        """
        Assign a heuristic score to an action based on its potential impact.

        Args:
            action (Action): The action to score.
            game (Game): The current game state.

        Returns:
            float: The heuristic score for the action.
        """
        priorities = {
            ActionType.BUILD_CITY: 15,
            ActionType.BUILD_SETTLEMENT: 12,
            ActionType.BUILD_ROAD: 10,
            ActionType.BUY_DEVELOPMENT_CARD: 7,
            ActionType.MARITIME_TRADE: 5,
            ActionType.MOVE_ROBBER: 4,
            ActionType.END_TURN: -5,  # Penalized further for passivity
            ActionType.ROLL: -3,  # Penalized further for lack of impact
        }

        action_type = action.action_type
        base_score = priorities.get(action_type, 0)

        # Enhance road-building scoring
        if action_type == ActionType.BUILD_ROAD:
            if action.value and self._road_leads_to_settlement_spot(action, game):
                base_score += 5

        # Enhance robber logic to target opponents with higher VP
        if action_type == ActionType.MOVE_ROBBER:
            base_score += self._evaluate_robber_impact(action, game)

        # Adjust maritime trade logic to prioritize critical resources dynamically
        if action_type == ActionType.MARITIME_TRADE and action.value:
            base_score += self._evaluate_trade_value(action.value, game)

        return base_score

    def _road_leads_to_settlement_spot(self, action, game):
        """
        Analyze if the road being built leads to a potential new settlement spot.
        """
        road_endpoints = action.value
        for endpoint in road_endpoints:
            # Check if this vertex is available for settlement
            if endpoint not in game.state.board.buildings:
                return True
        return False

    def _evaluate_robber_impact(self, action, game):
        """
        Evaluate and score the impact of moving the robber.
        """
        robber_position, target_player, _ = action.value
        score = 0

        if target_player:
            player_vp_key = player_key(game.state, target_player)
            target_player_vp = game.state.player_state.get(f"{player_vp_key}_ACTUAL_VICTORY_POINTS", 0)
            # Scale aggressiveness based on opponent VP difference
            score += max(0, (target_player_vp - game.state.player_state.get(player_key(game.state, self.color) + '_ACTUAL_VICTORY_POINTS', 0)))

        # Additional logic: Evaluate blocked tiles and resources
        board = game.state.board
        if robber_position in board.map.land_tiles:
            blocked_tile = board.map.land_tiles[robber_position]
            blocked_resources = blocked_tile.resource  # A single resource per tile

            for node_id in blocked_tile.nodes.values():
                building = board.buildings.get(node_id)
                if building:
                    owner, building_type = building
                    if owner != self.color:
                        resource_penalty = 1 if building_type == 'SETTLEMENT' else 2
                        score += resource_penalty

        return score

    def _evaluate_trade_value(self, trade, game):
        """
        Evaluate the value of a maritime trade based on resource needs.
        """
        giving = trade[:-1]  # Resources being given up
        receiving = trade[-1]  # Resource being received

        # Dynamic weighting based on current shortages
        shortages = self._current_resource_shortages(game)
        resource_weights = {
            WOOD: 1 + shortages[WOOD],
            BRICK: 1 + shortages[BRICK],
            WHEAT: 4 + shortages[WHEAT],
            ORE: 4 + shortages[ORE],
            SHEEP: 2 + shortages[SHEEP],
        }

        giving_value = sum(resource_weights[res] for res in giving)
        receiving_value = resource_weights[receiving]

        # Trade is good if we're receiving a more valuable resource
        return receiving_value - giving_value

    def _current_resource_shortages(self, game):
        """
        Analyze resource shortages based on the player's current hand and targets.

        Returns:
            dict: A dictionary with resource type as key and shortage amount as value.
        """
        shortages = {WOOD: 0, BRICK: 0, WHEAT: 0, ORE: 0, SHEEP: 0}
        player_state_key = player_key(game.state, self.color)

        # Summarize player's resources in hand
        player_hand = {resource: game.state.player_state.get(f"{player_state_key}_{resource}_IN_HAND", 0) for resource in [WOOD, BRICK, SHEEP, WHEAT, ORE]}

        # Desired amounts for structures
        desired_resources = {
            WOOD: 2, BRICK: 2, WHEAT: 2, ORE: 3, SHEEP: 1
        }

        for resource, desired in desired_resources.items():
            shortages[resource] = max(0, desired - player_hand.get(resource, 0))

        return shortages

    def decide(self, game, playable_actions):
        """
        Choose the best action based on a heuristic scoring system.

        Args:
            game (Game): The complete game state (read-only).
            playable_actions (Iterable[Action]): Available actions to choose from.

        Returns:
            Action: Chosen element of playable_actions.
        """
        player_state_key = player_key(game.state, self.color)
        current_vp = game.state.player_state.get(f"{player_state_key}_ACTUAL_VICTORY_POINTS", 0)
        print(f"FooPlayer ({self.color}): Current VP: {current_vp}")

        scored_actions = [
            (action, self.heuristic_score_action(action, game)) for action in playable_actions
        ]

        # Find the best action
        best_action, best_action_score = max(scored_actions, key=lambda x: x[1])

        # Prevent repetitive passive actions with stricter penalties
        if best_action.action_type in self.repetitive_actions:
            self.repetitive_actions[best_action.action_type] += 1
            if self.repetitive_actions[best_action.action_type] > 2:
                print("Avoiding repetitive passive actions. Re-assessing other options.")
                fallback_actions = [action for action in playable_actions if action.action_type not in self.repetitive_actions]
                if fallback_actions:
                    fallback_action = max(fallback_actions, key=lambda a: self.heuristic_score_action(a, game))
                    print(f"Fallback Action: {fallback_action}, Score: {self.heuristic_score_action(fallback_action, game)}")
                    return fallback_action
        else:
            # Reset the counter for other actions
            self.repetitive_actions = {ActionType.END_TURN: 0, ActionType.ROLL: 0}

        # Log chosen action
        print(f"Chosen Action: {best_action}, Score: {best_action_score}")

        return best_action