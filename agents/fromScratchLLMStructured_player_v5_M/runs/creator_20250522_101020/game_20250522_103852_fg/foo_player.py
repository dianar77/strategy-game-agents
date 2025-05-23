import random
import logging

from catanatron.players.player import Player
from catanatron.models.enums import ActionType, WOOD, BRICK, SHEEP, WHEAT, ORE, RESOURCES
from catanatron.state_functions import (
    get_player_buildings, 
    get_player_freqdeck, 
    get_actual_victory_points, 
    player_num_resource_cards,
    player_resource_freqdeck_contains,
)

class FooPlayer(Player):
    def __init__(self, color, llm=None):
        super().__init__(color)
        self.color = color
        self.llm = llm
        self.initial_placements_done = False
        logging.info(f"Initialized FooPlayer with color {color}")
    
    def reset_state(self):
        """Reset player state between games"""
        self.initial_placements_done = False
        logging.info(f"Reset FooPlayer state")
    
    def decide(self, game, playable_actions):
        """Make a decision based on available actions"""
        if not playable_actions:
            return None
            
        # Simple ranking of actions
        actions = self.rank_actions(game, playable_actions)
        logging.info(f"FooPlayer deciding from {len(playable_actions)} actions")
        
        chosen_action = actions[0]
        logging.info(f"FooPlayer chose action: {chosen_action.action_type}")
        return chosen_action
    
    def rank_actions(self, game, playable_actions):
        """Rank actions by priority"""
        scored_actions = []
        
        for action in playable_actions:
            score = self.evaluate_action(game, action)
            scored_actions.append((score, action))
            
        scored_actions.sort(reverse=True)
        return [action for score, action in scored_actions]
    
    def evaluate_action(self, game, action):
        """Simple evaluation of actions with reliable scoring"""
        score = 0
        state = game.state
        
        # Highest priority - must complete actions
        if action.action_type == ActionType.ROLL:
            return 100
        
        # HIGH PRIORITY: Play development cards when advantageous 
        if action.action_type in [ActionType.PLAY_KNIGHT_CARD, ActionType.PLAY_MONOPOLY, 
                                 ActionType.PLAY_ROAD_BUILDING, ActionType.PLAY_YEAR_OF_PLENTY]:
            return 90
        
        # Evaluate settlement placement - high priority for good locations
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            node_id = action.value
            try:
                # Get production potential of this node
                node_production = game.state.board.map.node_production.get(node_id, {})
                
                # Value based on production probability
                production_value = sum(node_production.values()) if node_production else 0
                
                # Value resource diversity
                diversity_value = len(node_production) * 3 if node_production else 0
                
                # Value brick and wood more in early game
                early_game_bonus = 0
                my_vp = get_actual_victory_points(state, self.color)
                if my_vp < 4:  # Early game
                    if BRICK in node_production:
                        early_game_bonus += node_production[BRICK] * 1.5
                    if WOOD in node_production:
                        early_game_bonus += node_production[WOOD] * 1.5
                    if WHEAT in node_production:
                        early_game_bonus += node_production[WHEAT] * 1.2
                
                # Check if near a port (safely)
                port_value = 0
                try:
                    for resource, port_nodes in game.state.board.map.port_nodes.items():
                        if node_id in port_nodes:
                            # Value 2:1 ports higher than 3:1 ports
                            port_value = 5 if resource is not None else 3
                except Exception:
                    pass  # Ignore port errors
                
                # Combine scores with appropriate weights
                score = 50 + (production_value * 8) + diversity_value + port_value + early_game_bonus
                
                # Log detailed settlement evaluation for important decisions
                if len(game.state.buildings) < 8:  # Initial placement phase
                    logging.info(f"Settlement at {node_id} scored: {score:.1f} (prod:{production_value:.1f}, div:{diversity_value})")
            
            except Exception as e:
                logging.warning(f"Error evaluating settlement: {e}")
                score = 50  # Default settlement score
        
        # Evaluate city building - high priority for resource production
        elif action.action_type == ActionType.BUILD_CITY:
            node_id = action.value
            try:
                # Get production potential of this node
                node_production = game.state.board.map.node_production.get(node_id, {})
                production_value = sum(node_production.values()) if node_production else 0
                
                # Cities are more valuable on high production nodes
                score = 45 + (production_value * 2)
            except Exception:
                score = 45  # Default city score
        
        # Evaluate road building - moderate priority for expansion
        elif action.action_type == ActionType.BUILD_ROAD:
            score = 30  # Default road score
            
            # Simple road evaluation - no complex node checking to avoid errors
            my_vp = get_actual_victory_points(state, self.color)
            if my_vp < 4:  # Early game - roads are more important
                score = 35
        
        # Evaluate development card purchase
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            # More valuable if we need VP or have few options
            my_vp = get_actual_victory_points(state, self.color)
            if my_vp >= 7:  # Close to winning
                score = 40  # Higher priority for potential victory points
            else:
                score = 25
            
        # Evaluate robber actions - simple scoring to avoid errors
        elif action.action_type == ActionType.MOVE_ROBBER:
            score = 20  # Default robber score
            
            try:
                # Safe unpacking with proper error handling
                if isinstance(action.value, tuple):
                    if len(action.value) >= 3:
                        # Standard format: (coord, player, _)
                        target_coord, target_color, _ = action.value
                        
                        # Bonus for targeting a player (stealing)
                        if target_color is not None:
                            score += 5
                    elif len(action.value) == 2:
                        # Alternative format: (coord, player)
                        target_coord, target_color = action.value
                        
                        # Bonus for targeting a player (stealing)
                        if target_color is not None:
                            score += 5
            except Exception as e:
                logging.warning(f"Safe robber evaluation error: {e}")
        
        # For trading and other actions        
        elif action.action_type in [ActionType.MARITIME_TRADE, ActionType.OFFER_TRADE]:
            score = 15
            
        # For end turn
        elif action.action_type == ActionType.END_TURN:
            score = 10
            
        # For other action types
        else:
            score = 5
            
        # Add small randomness to avoid ties
        score += random.uniform(0, 0.1)
        
        return score
    
    def has_resources_for_settlement(self, state):
        """Check if player has resources to build a settlement"""
        return player_resource_freqdeck_contains(
            state, 
            self.color, 
            {BRICK: 1, WOOD: 1, WHEAT: 1, SHEEP: 1}
        )
        
    def has_resources_for_city(self, state):
        """Check if player has resources to build a city"""
        return player_resource_freqdeck_contains(
            state, 
            self.color, 
            {ORE: 3, WHEAT: 2}
        )
        
    def has_resources_for_road(self, state):
        """Check if player has resources to build a road"""
        return player_resource_freqdeck_contains(
            state, 
            self.color, 
            {BRICK: 1, WOOD: 1}
        )