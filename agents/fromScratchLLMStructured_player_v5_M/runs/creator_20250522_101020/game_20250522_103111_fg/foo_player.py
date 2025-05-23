import random
from typing import List
import logging

from catanatron.models.player import Color, Player
from catanatron.models.actions import ActionType
from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE, RESOURCES
from catanatron.state_functions import (
    get_player_buildings, 
    get_player_freqdeck, 
    get_actual_victory_points, 
    player_num_resource_cards,
    player_can_afford_dev_card,
    player_resource_freqdeck_contains
)
from agents.fromScratchLLMStructured_player_v5_M.llm_tools import LLM


class FooPlayer(Player):
    def __init__(self, color, llm=None):
        super().__init__(color)
        self.color = color
        self.llm = LLM() if llm is None else llm  # use self.llm.query_llm(str prompt) to query the LLM
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
        """
        Decides on the action to take based on the current game state 
        and available actions.
        """
        if not playable_actions:
            return None
            
        # Early game - request LLM guidance for overall strategy
        if not self.strategy_analyzed:
            self.analyze_game_strategy(game)
            self.strategy_analyzed = True
        
        # Get information about the current state of the game
        state = game.state
        my_vp = get_actual_victory_points(state, self.color)
        
        # Make important decisions using LLM
        if len(playable_actions) > 3:
            # For initial settlement placement, use LLM
            if any(a.action_type == ActionType.BUILD_SETTLEMENT for a in playable_actions) and not self.initial_placements_done:
                settlement_decision = self.llm_decide_initial_settlement(game, playable_actions)
                if settlement_decision:
                    return settlement_decision
            
            # For robber placement, use LLM
            if any(a.action_type == ActionType.MOVE_ROBBER for a in playable_actions):
                robber_decision = self.llm_decide_robber_placement(game, playable_actions)
                if robber_decision:
                    return robber_decision
        
        logging.info(f"FooPlayer deciding from {len(playable_actions)} actions")
        
        # Rank actions based on their strategic value
        ranked_actions = self.rank_actions(game, playable_actions)
        
        # Choose the highest ranked action
        chosen_action = ranked_actions[0]
        logging.info(f"FooPlayer chose action: {chosen_action.action_type}")
        return chosen_action
    
    def analyze_game_strategy(self, game):
        """Use LLM to analyze the game and determine overall strategy"""
        try:
            prompt = f"""
            You are an expert Catan strategy advisor. Analyze this game board state and provide a concise strategy for the {self.color} player.
            What resources should be prioritized? What building types should be focused on first?
            Keep your response under 100 words and focus only on actionable strategy advice.
            """
            
            response = self.llm.query_llm(prompt)
            self.game_strategy = response
            logging.info(f"Strategy: {self.game_strategy}")
        except Exception as e:
            logging.warning(f"Error getting LLM strategy: {e}")
    
    def llm_decide_initial_settlement(self, game, playable_actions):
        """Use LLM to decide on initial settlement placement"""
        try:
            settlement_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT]
            if not settlement_actions:
                return None
                
            # Get board representation and available nodes
            board = game.state.board
            settlement_options = []
            
            for action in settlement_actions:
                node_id = action.value
                node_production = board.map.node_production[node_id]
                
                # Format the node production info for LLM
                resources_info = []
                total_prob = 0
                for resource, prob in node_production.items():
                    resources_info.append(f"{resource}: {prob:.1f}%")
                    total_prob += prob
                    
                port_info = "No port"
                for resource, port_nodes in board.map.port_nodes.items():
                    if node_id in port_nodes:
                        port_info = f"{resource if resource else '3:1'} port"
                
                option_info = {
                    "node_id": node_id,
                    "resources": ", ".join(resources_info),
                    "total_probability": f"{total_prob:.1f}%",
                    "resource_count": len(node_production),
                    "port": port_info
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
            4. Port access if available
            
            Return only the node_id of the best settlement location.
            """
            
            response = self.llm.query_llm(prompt)
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
        return None
        
    def llm_decide_robber_placement(self, game, playable_actions):
        """Use LLM to decide on robber placement"""
        try:
            robber_actions = [a for a in playable_actions if a.action_type == ActionType.MOVE_ROBBER]
            if not robber_actions:
                return None
                
            # Get information about other players
            player_info = []
            state = game.state
            for player in state.players:
                if player.color != self.color:
                    player_info.append({
                        "color": player.color.name if hasattr(player.color, 'name') else str(player.color),
                        "vp": get_actual_victory_points(state, player.color),
                        "cards": player_num_resource_cards(state, player.color)
                    })
            
            # Get information about robber placement options
            robber_options = []
            for action in robber_actions:
                coord, victim = action.value
                
                # Get the resources at this tile
                resource_at_tile = game.state.board.map.tiles.get(coord, None)
                dice_number = game.state.board.map.number_to_tile.get(coord, None)
                
                option = {
                    "coord": coord,
                    "victim": victim.name if victim and hasattr(victim, 'name') else str(victim) if victim else None,
                    "resource": resource_at_tile,
                    "dice_number": dice_number,
                }
                robber_options.append(option)
            
            # Request LLM guidance
            prompt = f"""
            You are an expert Catan player helping choose where to place the robber.
            
            Information about opponents:
            {player_info}
            
            Information about possible robber placement options:
            {robber_options}
            
            Choose the best robber placement strategy considering:
            1. Target the player with the most victory points
            2. Target a player with many resource cards to steal from
            3. Block an important resource tile (brick or wood are usually valuable early)
            4. Block tiles with high probability numbers (6 or 8)
            
            Return only the coordinate of your chosen placement (e.g., '(0, 0, 0)').
            """
            
            response = self.llm.query_llm(prompt)
            
            # Try to extract coordinate from response
            for action in robber_actions:
                coord_str = str(action.value[0])
                if coord_str in response:
                    logging.info(f"LLM selected robber placement at {coord_str}")
                    return action
        except Exception as e:
            logging.warning(f"Error in LLM robber decision: {e}")
            
        # Fallback to ranking algorithm
        return None
    
    def rank_actions(self, game, playable_actions):
        """
        Ranks the available actions based on strategic value.
        Returns list of actions sorted by priority (highest first).
        """
        scored_actions = []
        for action in playable_actions:
            score = self.evaluate_action(game, action)
            scored_actions.append((score, action))
        
        # Sort by score (descending) and return just the actions
        scored_actions.sort(reverse=True)
        return [action for score, action in scored_actions]
    
    def evaluate_action(self, game, action):
        """
        Evaluates an action and returns a score representing its value.
        Higher scores are better.
        """
        score = 0
        state = game.state
        
        # HIGHEST PRIORITY: Roll dice - must be done when available
        if action.action_type == ActionType.ROLL:
            return 100
            
        # HIGH PRIORITY: Play development cards when advantageous 
        if action.action_type in [ActionType.PLAY_KNIGHT_CARD, ActionType.PLAY_MONOPOLY, 
                                 ActionType.PLAY_ROAD_BUILDING, ActionType.PLAY_YEAR_OF_PLENTY]:
            return 90
        
        # Evaluate settlement placement - high priority for good locations
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            node_id = action.value
            # Get production potential of this node
            node_production = game.state.board.map.node_production[node_id]
            
            # Value based on production probability
            production_value = sum(node_production.values())
            
            # Value resource diversity, with extra weight for scarce resources
            diversity_value = len(node_production) * 3
            
            # Value brick and wood more in early game (for roads and settlements)
            early_game_bonus = 0
            my_vp = get_actual_victory_points(state, self.color)
            if my_vp < 4:  # Early game
                if BRICK in node_production:
                    early_game_bonus += node_production[BRICK] * 1.5
                if WOOD in node_production:
                    early_game_bonus += node_production[WOOD] * 1.5
                if WHEAT in node_production:
                    early_game_bonus += node_production[WHEAT] * 1.2
            
            # Check if near a port
            port_value = 0
            for resource, port_nodes in game.state.board.map.port_nodes.items():
                if node_id in port_nodes:
                    # Value 2:1 ports higher than 3:1 ports
                    port_value = 5 if resource is not None else 3
            
            # Combine scores with appropriate weights
            score = 50 + (production_value * 8) + diversity_value + port_value + early_game_bonus
            
            # Log detailed settlement evaluation for important decisions
            if len(game.state.buildings) < 8:  # Initial placement phase
                logging.info(f"Settlement at {node_id} scored: {score:.1f} (prod:{production_value:.1f}, div:{diversity_value}, port:{port_value}, bonus:{early_game_bonus:.1f})")
        
        # Evaluate city building - high priority for resource production
        elif action.action_type == ActionType.BUILD_CITY:
            node_id = action.value
            # Get production potential of this node
            node_production = game.state.board.map.node_production[node_id]
            production_value = sum(node_production.values())
            
            # Cities are more valuable on high production nodes
            score = 45 + (production_value * 3)
            
            # Prioritize cities that produce ore/wheat for more cities and dev cards
            resource_bonus = 0
            if ORE in node_production:
                resource_bonus += node_production[ORE] * 1.5
            if WHEAT in node_production:
                resource_bonus += node_production[WHEAT] * 1.5
                
            score += resource_bonus
        
        # Evaluate road building - moderate priority for expansion
        elif action.action_type == ActionType.BUILD_ROAD:
            edge = action.value
            
            # Check if this road leads to potential settlement spots
            board = game.state.board
            potential_settlements = 0
            
            try:
                # Get the nodes connected by this edge - Fixed attribute name
                u, v = board.map.edge_nodes[edge]
                
                # Check if building a road here would enable new settlement spots
                for node in [u, v]:
                    # If the node is empty and we can build there eventually
                    if (board.buildable_node(node, self.color, board.settlements, board.cities) and 
                        not board.get_node_color(node)):
                        potential_settlements += 1
                        
                        # Check the production value of potential settlements
                        if potential_settlements > 0:
                            node_production = board.map.node_production[node]
                            if node_production:
                                potential_value = sum(node_production.values())
                                # Extra points for high-value settlement spots
                                potential_settlements += (potential_value / 10)
                
                # Roads that lead to potential settlements are more valuable
                score = 30 + (potential_settlements * 5)
            except (KeyError, AttributeError) as e:
                # Fallback score if there's an issue with the edge lookup
                logging.warning(f"Error evaluating road at edge {edge}: {e}")
                score = 30  # Default score for roads
        
        # Evaluate development card purchase
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            # More valuable if we need VP or have few options
            my_vp = get_actual_victory_points(state, self.color)
            if my_vp >= 7:  # Close to winning
                score = 40  # Higher priority for potential victory points
            else:
                score = 25
            
        # Evaluate robber actions - prioritize blocking opponents
        elif action.action_type == ActionType.MOVE_ROBBER:
            target_coord, target_color = action.value
            
            # If targeting a player, evaluate their strength
            score = 20
            if target_color:
                target_vp = get_actual_victory_points(state, target_color)
                score += target_vp * 2  # Target stronger players
                
                # Target players with more cards
                target_cards = player_num_resource_cards(state, target_color)
                score += min(target_cards, 7)  # Cap the bonus to avoid excessive targeting
                
                # Look at the tile being blocked
                blocked_resource = game.state.board.map.tiles.get(target_coord, None)
                blocked_number = game.state.board.map.number_to_tile.get(target_coord, None)
                
                # Blocking high-probability tiles is valuable
                if blocked_number in [6, 8]:
                    score += 5
                elif blocked_number in [5, 9]:
                    score += 3
                
                # Blocking crucial resources is valuable
                if blocked_resource == BRICK or blocked_resource == WOOD:
                    score += 4  # Important for early expansion
                elif blocked_resource == ORE or blocked_resource == WHEAT:
                    score += 3  # Important for late game
        
        # For trading and other actions        
        elif action.action_type in [ActionType.MARITIME_TRADE, ActionType.OFFER_TRADE]:
            score = 15
            
            # Check if we're getting resources we need
            if action.action_type == ActionType.MARITIME_TRADE:
                target_resource = action.value[1]
                
                # Check if we need this resource for immediate building
                my_resources = get_player_freqdeck(game.state, self.color)
                
                # Value trades that get us brick/wood in early game
                my_vp = get_actual_victory_points(state, self.color)
                if my_vp < 4:
                    if target_resource == BRICK or target_resource == WOOD:
                        score += 10
                # Value trades that get us ore/wheat in mid/late game
                else:
                    if target_resource == ORE or target_resource == WHEAT:
                        score += 10
            
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