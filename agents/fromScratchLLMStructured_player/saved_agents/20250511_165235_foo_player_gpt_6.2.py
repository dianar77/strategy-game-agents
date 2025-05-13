import os
from catanatron import Player
from catanatron.game import Game
from catanatron.state_functions import player_key
from catanatron.models.enums import ActionType, WOOD, BRICK, SHEEP, WHEAT, ORE
from agents.fromScratchLLMStructured_player.llm_tools import LLM

class FooPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name)
        self.llm = LLM()
        self.repetitive_actions = {ActionType.END_TURN: 0, ActionType.ROLL: 0}
        self.previous_action = None

    def heuristic_score_action(self, action, game):
        priorities = {
            ActionType.BUILD_CITY: 50,
            ActionType.BUILD_SETTLEMENT: 40,
            ActionType.BUILD_ROAD: 30,
            ActionType.BUY_DEVELOPMENT_CARD: 25,
            ActionType.MARITIME_TRADE: 20,
            ActionType.MOVE_ROBBER: 15,
            ActionType.END_TURN: -50,  # Stronger penalty for passive gameplay
            ActionType.ROLL: -40,     # Stronger penalty for passive gameplay
        }
 
        action_type = action.action_type
        base_score = priorities.get(action_type, 0)

        if action_type == ActionType.BUILD_CITY:
            base_score += self._dynamic_city_weight(game)

        if action_type == ActionType.BUILD_ROAD:
            if self._road_leads_to_settlement_spot(action, game):
                base_score += 15

        if action_type == ActionType.MOVE_ROBBER:
            base_score += self._evaluate_robber_impact(action, game)

        if action_type == ActionType.MARITIME_TRADE and action.value:
            base_score += self._evaluate_trade_value(action.value, game)

        if action_type == ActionType.BUY_DEVELOPMENT_CARD:
            base_score += self._evaluate_development_card_weight(game)

        return base_score

    def _dynamic_city_weight(self, game):
        player_state_key = player_key(game.state, self.color)
        current_vp = game.state.player_state.get(f"{player_state_key}_ACTUAL_VICTORY_POINTS", 0)
        return current_vp * 10  # Increased weight for proximity to victory

    def _road_leads_to_settlement_spot(self, action, game):
        road_endpoints = action.value
        for endpoint in road_endpoints:
            if endpoint not in game.state.board.buildings:
                return True
        return False

    def _evaluate_robber_impact(self, action, game):
        robber_position, target_player, _ = action.value
        score = 0

        if target_player:
            player_vp_key = player_key(game.state, target_player)
            target_player_vp = game.state.player_state.get(f"{player_vp_key}_ACTUAL_VICTORY_POINTS", 0)
            score += max(0, target_player_vp - game.state.player_state.get(player_key(game.state, self.color) + '_ACTUAL_VICTORY_POINTS', 0))

        board = game.state.board
        if robber_position in board.map.land_tiles:
            blocked_tile = board.map.land_tiles[robber_position]
            for node_id in blocked_tile.nodes.values():
                building = board.buildings.get(node_id)
                if building:
                    owner, building_type = building
                    if owner != self.color:
                        resource_penalty = 5 if building_type == 'SETTLEMENT' else 7
                        score += resource_penalty

        return score

    def _evaluate_trade_value(self, trade, game):
        giving = trade[:-1]
        receiving = trade[-1]

        shortages = self._current_resource_shortages(game)
        resource_weights = {
            WOOD: 4 + shortages[WOOD],
            BRICK: 4 + shortages[BRICK],
            WHEAT: 10 + shortages[WHEAT],
            ORE: 10 + shortages[ORE],
            SHEEP: 6 + shortages[SHEEP],
        }

        giving_value = sum(resource_weights[res] for res in giving)
        receiving_value = resource_weights[receiving]

        trade_margin = receiving_value - giving_value

        # Reward trades involving critical shortages
        if shortages[receiving] > 2:
            trade_margin += 5  # Bonus for addressing shortages

        return trade_margin

    def _evaluate_development_card_weight(self, game):
        player_state_key = player_key(game.state, self.color)
        knights_played = game.state.player_state.get(f"{player_state_key}_USED_KNIGHTS", 0)
        vp_from_cards = game.state.player_state.get(f"{player_state_key}_DEVELOPMENT_VICTORY_POINTS", 0)

        if knights_played >= 3 or vp_from_cards > 0:
            return 25
        return 15

    def _current_resource_shortages(self, game):
        shortages = {WOOD: 0, BRICK: 0, WHEAT: 0, ORE: 0, SHEEP: 0}
        player_state_key = player_key(game.state, self.color)

        player_hand = {resource: game.state.player_state.get(f"{player_state_key}_{resource}_IN_HAND", 0) for resource in [WOOD, BRICK, SHEEP, WHEAT, ORE]}
        desired_resources = {
            WOOD: 3, BRICK: 3, WHEAT: 5, ORE: 4, SHEEP: 2
        }

        for resource, desired in desired_resources.items():
            shortages[resource] = max(0, desired - player_hand.get(resource, 0))

        return shortages

    def decide(self, game, playable_actions):
        player_state_key = player_key(game.state, self.color)
        current_vp = game.state.player_state.get(f"{player_state_key}_ACTUAL_VICTORY_POINTS", 0)
        print(f"FooPlayer ({self.color}): Current VP: {current_vp}")

        scored_actions = [
            (action, self.heuristic_score_action(action, game)) for action in playable_actions
        ]

        # Debug: Print all scored actions
        for a, score in scored_actions:
            print(f"Action: {a}, Score: {score}")

        best_action, best_action_score = max(scored_actions, key=lambda x: x[1])

        if best_action.action_type in self.repetitive_actions:
            self.repetitive_actions[best_action.action_type] += 1
            if self.repetitive_actions[best_action.action_type] > 1:
                print("Avoiding repetitive passive actions. Re-assessing other options.")
                filtered_actions = [
                    x for x in scored_actions
                    if x[1] > -float('inf') and x[0].action_type not in [ActionType.END_TURN, ActionType.ROLL]
                ]

                if filtered_actions:
                    best_action, best_action_score = max(filtered_actions, key=lambda x: x[1])
                    print(f"Fallback Action: {best_action}, Score: {best_action_score}")
                    self.repetitive_actions = {ActionType.END_TURN: 0, ActionType.ROLL: 0}
                else:
                    print("No valid fallback action found, enforcing aggressive strategy.")
                    aggressive_actions = [
                        x for x in scored_actions
                        if x[0].action_type in [ActionType.BUY_DEVELOPMENT_CARD, ActionType.MARITIME_TRADE]
                    ]

                    if aggressive_actions:
                        best_action, best_action_score = max(aggressive_actions, key=lambda x: x[1])
                    else:
                        best_action, best_action_score = max(scored_actions, key=lambda x: x[1])

        else:
            self.repetitive_actions = {ActionType.END_TURN: 0, ActionType.ROLL: 0}

        self.previous_action = best_action
        print(f"Chosen Action: {best_action}, Score: {best_action_score}")

        return best_action