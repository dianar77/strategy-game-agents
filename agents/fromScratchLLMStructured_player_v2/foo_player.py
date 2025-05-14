import random
from catanatron import Player
from catanatron.models.player import Color
from catanatron.models.enums import ActionType, RESOURCES, SETTLEMENT, CITY, ROAD
from catanatron.state_functions import (
    player_key, 
    get_player_buildings, 
    player_num_resource_cards
)

class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
    
    def decide(self, game, playable_actions):
        """Evaluates each action and selects the best one based on strategic value."""
        if len(playable_actions) == 1:
            return playable_actions[0]
        
        # Score and select the best action
        scored_actions = [(self.score_action(game, action), action) for action in playable_actions]
        best_score, best_action = max(scored_actions, key=lambda x: x[0])
        print(f"Best action: {best_action.action_type} with score {best_score}")
        return best_action
    
    def score_action(self, game, action):
        """Score an action based on its strategic value."""
        action_type = action.action_type
        
        # Base scores for different action types
        if action_type == ActionType.BUILD_CITY:
            return 1000  # Cities are highest priority for VP efficiency
        
        elif action_type == ActionType.BUILD_SETTLEMENT:
            # Evaluate settlement location based on production value
            node_id = action.value
            production_value = self.evaluate_node_production(game, node_id)
            return 800 + production_value
        
        elif action_type == ActionType.BUILD_ROAD:
            # Roads that enable new settlement spots are valuable
            return 500 + self.evaluate_road_value(game, action.value)
        
        elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return 600  # Development cards can provide VP and other benefits
        
        elif action_type == ActionType.PLAY_KNIGHT_CARD:
            return 700  # Knights help with largest army and resource control
        
        elif action_type == ActionType.MOVE_ROBBER:
            return 550 + self.evaluate_robber_move(game, action.value)
        
        elif action_type == ActionType.ROLL:
            return 2000  # Always roll when possible
        
        elif action_type == ActionType.END_TURN:
            return -100  # Only end turn when no better options
            
        elif action_type == ActionType.PLAY_MONOPOLY:
            return 650  # Monopoly can be powerful for resource collection
            
        elif action_type == ActionType.PLAY_ROAD_BUILDING:
            return 600  # Road building can be strategic for expansion
            
        elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            return 650  # Year of plenty helps gather needed resources
            
        elif action_type == ActionType.MARITIME_TRADE:
            # Evaluate trades based on what resources we need most
            return 300 + self.evaluate_trade_value(game, action.value)
            
        else:
            # Default score for other actions
            return 100
    
    def evaluate_node_production(self, game, node_id):
        """Evaluate a node's production value based on adjacent tiles."""
        value = 0
        # Get adjacent tiles
        for tile_id in game.state.board.map.adjacent_tiles.get(node_id, []):
            if tile_id in game.state.board.map.land_tiles:
                tile = game.state.board.map.land_tiles[tile_id]
                # Calculate value based on resource type and number
                if tile.resource is not None:  # Skip desert
                    # Value based on probability (6 and 8 are best)
                    probability_value = {
                        2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1
                    }.get(tile.number, 0)
                    
                    # Weight by resource value (ore and wheat for cities are valuable)
                    resource_value = {
                        "ORE": 5, "WHEAT": 5, "SHEEP": 3, "WOOD": 4, "BRICK": 4
                    }.get(str(tile.resource), 0)
                    
                    value += probability_value * resource_value
                    
                    # Bonus for resource diversity
                    value += 10
        return value
    
    def evaluate_road_value(self, game, edge):
        """Evaluate road's strategic value."""
        # Value roads that connect to potential settlement spots
        value = 0
        
        # Check if this road connects to nodes that could be settlements
        for node_id in edge:
            # Check if this node is buildable (not owned or adjacent to a settlement)
            if node_id in game.state.board.buildable_node_ids(self.color):
                # Road connects to a buildable settlement spot
                value += 50
                
                # Extra value if the node has good production
                node_production_value = self.evaluate_node_production(game, node_id)
                value += node_production_value / 5
                
        return value
    
    def evaluate_robber_move(self, game, robber_move):
        """Evaluate the value of a robber move."""
        if robber_move is None:
            return 0
            
        coordinate, target_color, _ = robber_move
        
        # Value blocking high-production tiles
        value = 0
        if coordinate in game.state.board.map.land_tiles:
            tile = game.state.board.map.land_tiles[coordinate]
            if tile.resource is not None:
                # 6 and 8 are most valuable to block
                value += {
                    2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1
                }.get(tile.number, 0) * 10
        
        # Value stealing from players with many cards
        if target_color is not None:
            value += player_num_resource_cards(game.state, target_color) * 5
            
        return value
        
    def evaluate_trade_value(self, game, trade_info):
        """Evaluate the value of a maritime trade."""
        # For maritime trade, the 5th resource is what we're getting
        if trade_info is None or len(trade_info) < 5:
            return 0
            
        # Resources we're giving up
        giving_resources = trade_info[:4]
        # Resource we're getting
        getting_resource = trade_info[4]
        
        # Value based on what we're getting
        resource_value = {
            "ORE": 5, "WHEAT": 5, "SHEEP": 3, "WOOD": 4, "BRICK": 4
        }.get(str(getting_resource), 0)
        
        # Count how many resources we're giving up
        num_giving = sum(1 for r in giving_resources if r is not None)
        
        # Better ratio = better trade
        trade_ratio_value = 40 - (num_giving * 10)
        
        # Check if the resource we're getting helps with our immediate building plans
        needed_for_city = self.resource_needed_for_city(game)
        needed_for_settlement = self.resource_needed_for_settlement(game)
        
        if getting_resource == needed_for_city or getting_resource == needed_for_settlement:
            resource_value += 20
            
        return resource_value + trade_ratio_value
        
    def resource_needed_for_city(self, game):
        """Determine which resource we most need to build a city."""
        # For a city, we need 2 wheat and 3 ore
        my_resources = self.get_my_resources(game)
        
        wheat_needed = max(0, 2 - my_resources.get("WHEAT", 0))
        ore_needed = max(0, 3 - my_resources.get("ORE", 0))
        
        if wheat_needed > ore_needed:
            return "WHEAT"
        else:
            return "ORE"
            
    def resource_needed_for_settlement(self, game):
        """Determine which resource we most need to build a settlement."""
        # For a settlement, we need 1 each of wood, brick, sheep, and wheat
        my_resources = self.get_my_resources(game)
        
        needed = {}
        for resource in ["WOOD", "BRICK", "SHEEP", "WHEAT"]:
            needed[resource] = max(0, 1 - my_resources.get(resource, 0))
            
        # Return the resource we need most
        max_needed = max(needed.values())
        if max_needed == 0:
            return None
            
        for resource, need in needed.items():
            if need == max_needed:
                return resource
                
        return None
        
    def get_my_resources(self, game):
        """Get a dictionary of my resources."""
        resources = {}
        key = player_key(game.state, self.color)
        
        for resource in RESOURCES:
            resources[resource] = game.state.player_state.get(f"{key}_{resource}_IN_HAND", 0)
            
        return resources