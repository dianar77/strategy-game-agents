import random
from typing import List, Dict, Tuple
from catanatron import Player
from catanatron.game import Game
from catanatron.models.enums import Action, RESOURCES, SETTLEMENT, CITY, ROAD, ActionType, ActionPrompt
from catanatron.state_functions import (
    get_player_buildings, 
    player_key,
    player_num_resource_cards,
    get_longest_road_length,
    get_actual_victory_points,
)

# Constants for evaluating board positions
SETTLEMENT_SCORE = 100
CITY_SCORE = 300
ROAD_SCORE = 20
RESOURCE_SCORE = 10
VP_SCORE = 1000
ROAD_LENGTH_SCORE = 20

# Number probability weights (6 and 8 are most likely to be rolled)
NUMBER_WEIGHTS = {
    2: 10,
    3: 20,
    4: 30,
    5: 40,
    6: 50,
    8: 50,
    9: 40,
    10: 30,
    11: 20,
    12: 10,
}

class FooPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name or "FooPlayer")
        
    def decide(self, game, playable_actions):
        """Choose best action using a simple evaluation function"""
        if len(playable_actions) == 1:
            return playable_actions[0]
            
        # Use a simple evaluation function to rank actions
        best_action = None
        best_value = float('-inf')
        
        # Check if we're in initial placement phase (different strategy needed)
        is_initial_placement = game.state.current_prompt in [
            ActionPrompt.BUILD_INITIAL_SETTLEMENT, 
            ActionPrompt.BUILD_INITIAL_ROAD
        ]
        
        for action in playable_actions:
            # Create a copy of the game to simulate the action
            game_copy = game.copy()
            
            # Execute action and evaluate resulting state
            game_copy.execute(action)
            value = self.evaluate_state(game_copy, is_initial_placement, action)
            
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action
    
    def evaluate_state(self, game, is_initial_placement=False, action=None):
        """Evaluate the game state from this player's perspective"""
        try:
            # Get player key for accessing state
            key = player_key(game.state, self.color)
            score = 0
            
            # 1. Victory points (most important metric)
            vp_score = get_actual_victory_points(game.state, self.color) * VP_SCORE
            score += vp_score
            
            # 2. Buildings value
            settlements = get_player_buildings(game.state, self.color, SETTLEMENT)
            cities = get_player_buildings(game.state, self.color, CITY)
            building_score = len(settlements) * SETTLEMENT_SCORE + len(cities) * CITY_SCORE
            score += building_score
            
            # 3. Resources in hand
            resources_score = sum([
                player_num_resource_cards(game.state, self.color, resource) 
                for resource in RESOURCES
            ]) * RESOURCE_SCORE
            score += resources_score
            
            # 4. Longest road bonus
            road_length = get_longest_road_length(game.state, self.color)
            road_score = road_length * ROAD_LENGTH_SCORE
            score += road_score
            
            # 5. Production potential
            production_score = self._calculate_production_score(game)
            score += production_score
            
            # Special handling for different action types
            if action is not None:
                if action.action_type == ActionType.BUILD_SETTLEMENT:
                    # Extra points for settlements on good production spots
                    node_id = action.value
                    score += self._evaluate_settlement_position(game, node_id) * 2
                    
                elif action.action_type == ActionType.BUILD_CITY:
                    # Extra points for cities on good production spots
                    node_id = action.value
                    score += self._evaluate_settlement_position(game, node_id) * 3
                    
                elif action.action_type == ActionType.BUILD_ROAD:
                    # Bonus for roads that lead to good future settlement spots
                    edge = action.value
                    score += self._evaluate_road_potential(game, edge)
            
            return score
        except Exception as e:
            # If any error occurs, return a basic score to avoid crashing
            # print(f"Error in evaluation: {e}")
            return 0
        
    def _calculate_production_score(self, game):
        """Calculate the production potential score based on building placements"""
        production_score = 0
        
        # Get all settlements and cities
        settlements = get_player_buildings(game.state, self.color, SETTLEMENT)
        cities = get_player_buildings(game.state, self.color, CITY)
        
        # For each building, add score based on the probability of adjacent tiles
        for node_id in settlements:
            production_score += self._evaluate_settlement_position(game, node_id)
                
        # Cities produce double
        for node_id in cities:
            production_score += self._evaluate_settlement_position(game, node_id) * 2
                
        return production_score
    
    def _evaluate_settlement_position(self, game, node_id):
        """Evaluate a settlement position based on surrounding tiles"""
        score = 0
        resources_seen = set()
        
        for tile_coord in game.state.board.map.adjacent_tiles[node_id]:
            tile = game.state.board.map.tiles.get(tile_coord)
            if tile is not None and hasattr(tile, 'number') and tile.number is not None:
                if tile.number == 7:  # Robber
                    continue
                # Score based on probability
                score += NUMBER_WEIGHTS.get(tile.number, 0)
                
                # Add resource diversity bonus
                if hasattr(tile, 'resource') and tile.resource is not None:
                    resources_seen.add(tile.resource)
                    score += 5  # Bonus for having access to a resource
        
        # Add diversity bonus
        score += len(resources_seen) * 10  # Additional bonus for resource diversity
        
        return score
    
    def _evaluate_road_potential(self, game, edge):
        """Evaluate a road's potential for future settlements"""
        try:
            score = 0
            
            # In Catanatron, edges are already tuples of (node_id1, node_id2)
            # So we can access the nodes directly from the edge
            for node_id in edge:
                # Check if we can build a settlement here in the future
                if self._is_potential_settlement_spot(game, node_id):
                    # Evaluate the potential settlement position
                    potential_value = self._evaluate_settlement_position(game, node_id)
                    score += potential_value * 0.5
            
            return score
        except Exception as e:
            # Fallback to simple evaluation if an error occurs
            return 0
    
    def _is_potential_settlement_spot(self, game, node_id):
        """Check if this is a potential future settlement spot"""
        # Check if node is vacant and respects distance rule
        board = game.state.board
        is_vacant = board.get_node_building(node_id) is None
        respects_distance = board.is_distance_compliant(node_id)
        
        # Also check if we have a nearby road or can build a road here
        has_or_can_build_road = False
        for edge in board.map.adjacent_edges[node_id]:
            building = board.get_edge_building(edge)
            if building is not None and building[0] == self.color:
                has_or_can_build_road = True
                break
        
        return is_vacant and respects_distance and has_or_can_build_road