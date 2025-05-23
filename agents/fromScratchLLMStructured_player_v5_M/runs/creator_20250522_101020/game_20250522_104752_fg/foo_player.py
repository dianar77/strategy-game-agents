import random
import logging

from catanatron.models.player import Player
from catanatron.models.actions import Action
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
        self.strategy_analyzed = False
        self.game_strategy = None
        logging.info(f"Initialized FooPlayer with color {color}")
    
    def reset_state(self):
        """Reset player state between games"""
        self.initial_placements_done = False
        self.strategy_analyzed = False
        self.game_strategy = None
        logging.info(f"Reset FooPlayer state")
    
    def decide(self, game, playable_actions):
        """Make a decision based on available actions"""
        if not playable_actions:
            return None
        
        # Early game - request LLM guidance for overall strategy
        if not self.strategy_analyzed and self.llm:
            self.analyze_game_strategy(game)
            self.strategy_analyzed = True
        
        # Make important decisions using LLM
        if len(playable_actions) > 3 and self.llm:
            # For initial settlement placement, use LLM
            if game.state.is_initial_build_phase and any(a.action_type == ActionType.BUILD_SETTLEMENT for a in playable_actions):
                return self.llm_decide_initial_settlement(game, playable_actions)
            
            # For robber placement, use LLM
            if any(a.action_type == ActionType.MOVE_ROBBER for a in playable_actions):
                return self.llm_decide_robber_placement(game, playable_actions)
            
        # Rank actions based on their strategic value
        ranked_actions = self.rank_actions(game, playable_actions)
        
        chosen_action = ranked_actions[0]
        if len(playable_actions) > 1:
            logging.info(f"FooPlayer choosing from {len(playable_actions)} actions, selected: {chosen_action.action_type}")
        return chosen_action
    
    def analyze_game_strategy(self, game):
        """Use LLM to analyze the game and determine overall strategy"""
        if not self.llm:
            return
            
        prompt = f"""
        You are an expert Catan strategy advisor. Analyze this game board state and provide a concise strategy for the {self.color} player.
        What resources should be prioritized? What building types should be focused on first?
        Keep your response under 100 words and focus only on actionable strategy advice.
        """
        
        try:
            response = self.llm.complete(prompt)
            self.game_strategy = response
            logging.info(f"Strategy: {self.game_strategy}")
        except Exception as e:
            logging.warning(f"Error getting LLM strategy: {e}")
    
    def llm_decide_initial_settlement(self, game, playable_actions):
        """Use LLM to decide on initial settlement placement"""
        if not self.llm:
            return self.rank_actions(game, playable_actions)[0]
            
        settlement_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        if not settlement_actions:
            return self.rank_actions(game, playable_actions)[0]
            
        # Get board representation and available nodes
        board = game.state.board
        settlement_options = []
        
        for action in settlement_actions:
            node_id = action.value
            node_production = board.map.node_production.get(node_id, {})
            
            # Format the node production info for LLM
            resources_info = []
            total_prob = 0
            for resource, prob in node_production.items():
                resources_info.append(f"{resource}: {prob:.1f}%")
                total_prob += prob
                
            option_info = {
                "node_id": node_id,
                "resources": ", ".join(resources_info),
                "total_probability": f"{total_prob:.1f}%",
                "resource_count": len(node_production)
            }
            settlement_options.append(option_info)
            
        # Request LLM guidance
        prompt = f"""
        You are an expert Catan player helping choose the best initial settlement location.
        These are the available settlement locations with their potential resource production:
        
        {settlement_options}
        
        Select the best location by considering:
        1. Overall resource production probability
        2. Resource diversity (having access to many different resources)
        3. Access to scarce resources (brick and wood are important early)
        
        Return only the node_id of the best settlement location.
        """
        
        try:
            response = self.llm.complete(prompt)
            # Extract node_id from response
            for option in settlement_options:
                if str(option["node_id"]) in response:
                    for action in settlement_actions:
                        if action.value == option["node_id"]:
                            logging.info(f"LLM selected settlement at node {option['node_id']}")
                            return action
        except Exception as e:
            logging.warning(f"Error in LLM settlement decision: {e}")
        
        # Fallback to ranking algorithm
        logging.info("Falling back to algorithm for settlement placement")
        return self.rank_actions(game, playable_actions)[0]
        
    def llm_decide_robber_placement(self, game, playable_actions):
        """Use LLM to decide on robber placement"""
        if not self.llm:
            return self.rank_actions(game, playable_actions)[0]
            
        robber_actions = [a for a in playable_actions if a.action_type == ActionType.MOVE_ROBBER]
        if not robber_actions:
            return self.rank_actions(game, playable_actions)[0]
            
        # Get information about other players
        player_info = []
        state = game.state
        for player in state.players:
            if player.color != self.color:
                player_info.append({
                    "color": player.color,
                    "vp": get_actual_victory_points(state, player.color),
                    "cards": player_num_resource_cards(state, player.color)
                })
                
        # Create options for robber placements with descriptions
        robber_options = []
        for action in robber_actions:
            try:
                # Handle different robber action value formats safely
                target_coord = action.value[0] if isinstance(action.value, tuple) else action.value
                target_player = None
                if isinstance(action.value, tuple) and len(action.value) >= 2:
                    target_player = action.value[1]
                
                option_info = {
                    "action_id": id(action),  # Unique identifier for this action
                    "tile_coord": str(target_coord),
                    "targets_player": target_player.color if target_player else "None"
                }
                robber_options.append(option_info)
            except Exception as e:
                logging.warning(f"Error processing robber option: {e}")
                
        # Request LLM guidance
        prompt = f"""
        You are an expert Catan player helping choose where to place the robber.
        Information about opponents:
        {player_info}
        
        Available robber placements:
        {robber_options}
        
        Choose the best robber placement by:
        1. Targeting the player with the most victory points
        2. Targeting a player with many resource cards
        3. Blocking an important resource (brick or wood are usually valuable early)
        
        Return only the action_id of the best robber placement.
        """
        
        try:
            response = self.llm.complete(prompt)
            # Extract action_id from response or target player
            for i, option in enumerate(robber_options):
                if str(option["action_id"]) in response:
                    logging.info(f"LLM selected robber option {i}")
                    return robber_actions[i]
                
            # Try to extract player color from response as fallback
            for player in player_info:
                if str(player["color"]).lower() in response.lower():
                    target_player = player["color"]
                    for i, option in enumerate(robber_options):
                        if option["targets_player"] == target_player:
                            logging.info(f"LLM targeting {target_player} with robber")
                            return robber_actions[i]
        except Exception as e:
            logging.warning(f"Error in LLM robber decision: {e}")
            
        # Fallback to ranking algorithm
        return self.rank_actions(game, playable_actions)[0]
    
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
                
                # Log detailed settlement evaluation for initial placement
                if game.state.is_initial_build_phase:  # Correct way to check for initial placement
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
            edge = action.value
            score = 30  # Default road score
            
            # Try to evaluate if this road leads to good settlement locations
            try:
                board = game.state.board
                potential_settlements = 0
                
                # Try different ways to get nodes connected by this edge
                nodes = None
                if hasattr(board.map, "edge_nodes"):
                    nodes = board.map.edge_nodes.get(edge)
                elif hasattr(board.map, "edge_to_nodes"):
                    nodes = board.map.edge_to_nodes.get(edge)
                else:
                    # Assume edge itself contains the nodes
                    nodes = edge if isinstance(edge, tuple) and len(edge) == 2 else None
                
                if nodes:
                    for node in nodes:
                        # Check if we can build a settlement here eventually
                        can_build = False
                        try:
                            # Use the correct method for checking buildable nodes
                            if hasattr(board, "buildable_node"):
                                can_build = board.buildable_node(
                                    node, self.color, board.settlements, board.cities
                                )
                            # Check if the node is empty (no settlements/cities)
                            node_empty = (
                                node not in board.settlements and 
                                node not in board.cities
                            )
                            
                            if can_build and node_empty:
                                potential_settlements += 1
                                
                                # Check node production value
                                node_production = board.map.node_production.get(node, {})
                                if node_production:
                                    production_value = sum(node_production.values())
                                    # Add bonus for high production potential
                                    if production_value > 0.15:  # Significant production
                                        potential_settlements += 1
                        except Exception:
                            pass
                
                # Roads that lead to potential settlements are more valuable
                score = 30 + (potential_settlements * 5)
                
                # Early game roads are more valuable for expansion
                my_vp = get_actual_victory_points(state, self.color)
                if my_vp < 4:
                    score += 5
            except Exception:
                # Fallback for road evaluation
                my_vp = get_actual_victory_points(state, self.color)
                if my_vp < 4:  # Early game - roads are more important
                    score = 35
                else:
                    score = 30
        
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
                    target_coord = action.value[0]
                    target_color = None
                    if len(action.value) >= 2:
                        target_color = action.value[1]
                        
                    # Bonus for targeting a player (stealing)
                    if target_color is not None:
                        score += 5
                        
                        # Additional bonus for targeting stronger players
                        try:
                            target_vp = get_actual_victory_points(state, target_color)
                            score += min(target_vp, 5)  # Cap the bonus
                        except:
                            pass
            except Exception:
                pass  # Silently handle robber action errors
        
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