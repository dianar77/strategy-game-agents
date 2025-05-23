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
        
        # Late game acceleration strategy (6+ VP)
        my_vp = get_actual_victory_points(game.state, self.color)
        if my_vp >= 6:
            late_game_action = self.late_game_strategy(game, playable_actions)
            if late_game_action:
                return late_game_action
        
        # Make important decisions using LLM
        if len(playable_actions) > 3 and self.llm:
            # For initial settlement placement, use LLM
            if game.state.is_initial_build_phase and any(a.action_type == ActionType.BUILD_SETTLEMENT for a in playable_actions):
                return self.llm_decide_initial_settlement(game, playable_actions)
            
            # For robber placement, use LLM
            if any(a.action_type == ActionType.MOVE_ROBBER for a in playable_actions):
                return self.llm_decide_robber_placement(game, playable_actions)
            
        # Development card decision logic
        if any(a.action_type == ActionType.BUY_DEVELOPMENT_CARD for a in playable_actions):
            if self.should_buy_development_card(game):
                for action in playable_actions:
                    if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                        return action
            
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
        """Evaluate actions with improved strategic scoring"""
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
                
                # IMPROVED: Calculate production value with explicit weights for dice numbers
                production_value = 0
                board = game.state.board
                
                # Get the tiles adjacent to this node
                adjacent_tiles = {}
                try:
                    # Try to get adjacent tiles from board map (implementation varies)
                    if hasattr(board.map, "node_to_tiles"):
                        adjacent_tiles = board.map.node_to_tiles.get(node_id, {})
                    else:
                        # Fallback to using node_production
                        adjacent_tiles = node_production
                except Exception:
                    # Default to using node_production if everything else fails
                    adjacent_tiles = node_production
                
                # Calculate production value with explicit weights for numbers
                number_weights = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
                for tile_coord, resource in adjacent_tiles.items():
                    try:
                        # Try to get the number on this tile
                        if hasattr(board.map, "coords_to_numbers"):
                            number = board.map.coords_to_numbers.get(tile_coord)
                            if number:
                                production_value += number_weights.get(number, 0)
                    except Exception:
                        pass
                
                # If we couldn't get number weights, fall back to node_production sum
                if production_value == 0:
                    production_value = sum(node_production.values()) if node_production else 0
                
                # IMPROVED: Value resource diversity with specific weights
                resource_types = set()
                resource_score = 0
                resource_weights = {WOOD: 1.5, BRICK: 1.5, WHEAT: 1.2, SHEEP: 1.0, ORE: 0.8}
                
                # Value resources differently based on game phase
                my_vp = get_actual_victory_points(state, self.color)
                if my_vp >= 4:  # Mid-game shift to ore/wheat
                    resource_weights = {WOOD: 1.0, BRICK: 1.0, WHEAT: 1.5, SHEEP: 0.8, ORE: 1.5}
                
                for resource in node_production:
                    resource_types.add(resource)
                    resource_score += resource_weights.get(resource, 1.0) * node_production[resource]
                
                diversity_value = len(resource_types) * 3
                
                # Check if near a port (safely)
                port_value = 0
                try:
                    for resource, port_nodes in game.state.board.map.port_nodes.items():
                        if node_id in port_nodes:
                            # Value 2:1 ports higher than 3:1 ports
                            port_value = 5 if resource is not None else 3
                            
                            # Value ports more if we produce that resource
                            if resource in node_production:
                                port_value += 3
                except Exception:
                    pass  # Ignore port errors
                
                # Combine scores with appropriate weights
                score = 50 + (production_value * 8) + diversity_value + port_value + resource_score
                
                # Log detailed settlement evaluation for initial placement
                if game.state.is_initial_build_phase:
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
                
                # IMPROVED: Prioritize city upgrades on ore/wheat producing settlements
                resource_bonus = 0
                if WHEAT in node_production:
                    resource_bonus += node_production[WHEAT] * 3
                if ORE in node_production:
                    resource_bonus += node_production[ORE] * 3
                
                # Cities are more valuable on high production nodes
                score = 45 + (production_value * 2) + resource_bonus
            except Exception:
                score = 45  # Default city score
        
        # Evaluate road building - IMPROVED strategic path finding
        elif action.action_type == ActionType.BUILD_ROAD:
            edge = action.value
            score = 30  # Default road score
            
            try:
                # IMPROVED: Evaluate road potential using path-finding logic
                future_settlement_score = self.evaluate_road_potential(game.state.board, edge)
                score = 30 + future_settlement_score
                
                # Early game roads are more valuable for expansion
                my_vp = get_actual_victory_points(state, self.color)
                if my_vp < 4:
                    score += 5
            except Exception as e:
                # Fallback for road evaluation
                my_vp = get_actual_victory_points(state, self.color)
                if my_vp < 4:  # Early game - roads are more important
                    score = 35
                else:
                    score = 30
        
        # Evaluate development card purchase using game phase strategy
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            my_vp = get_actual_victory_points(state, self.color)
            
            if my_vp < 4:  # Early game
                score = 20  # Lower priority, focus on expansion
            elif my_vp < 7:  # Mid game
                score = 35  # Medium priority for balanced strategy
            else:  # Late game
                score = 45  # Higher priority for potential victory points
            
        # Evaluate robber actions - improved targeting
        elif action.action_type == ActionType.MOVE_ROBBER:
            score = 20  # Default robber score
            
            try:
                # Safe unpacking with proper error handling
                if isinstance(action.value, tuple):
                    target_coord = action.value[0]
                    target_color = None
                    if len(action.value) >= 2:
                        target_color = action.value[1]
                        
                    # IMPROVED: Evaluate the resource tile being blocked
                    blocked_resource = None
                    blocked_number = None
                    
                    # Try to identify the resource type and number being blocked
                    try:
                        board = game.state.board
                        
                        # Find the resource at this coordinate
                        if hasattr(board, "resource_at"):
                            blocked_resource = board.resource_at(target_coord)
                        
                        # Find the number at this coordinate
                        if hasattr(board.map, "coords_to_numbers"):
                            blocked_number = board.map.coords_to_numbers.get(target_coord)
                        elif hasattr(board.map, "number_to_tiles"):
                            for number, coords in board.map.number_to_tiles.items():
                                if target_coord in coords:
                                    blocked_number = number
                                    break
                    except Exception:
                        pass
                    
                    # Bonus for blocking high-probability numbers
                    if blocked_number in [6, 8]:
                        score += 10
                    elif blocked_number in [5, 9]:
                        score += 8
                    elif blocked_number in [4, 10]:
                        score += 6
                    
                    # Bonus for blocking valuable resources in early game
                    my_vp = get_actual_victory_points(state, self.color)
                    if my_vp < 5:  # Early-mid game
                        if blocked_resource in [BRICK, WOOD]:
                            score += 8
                        elif blocked_resource == WHEAT:
                            score += 6
                    else:  # Late game
                        if blocked_resource in [ORE, WHEAT]:
                            score += 8
                        
                    # Bonus for targeting a player (stealing)
                    if target_color is not None:
                        score += 5
                        
                        # Additional bonus for targeting stronger players
                        try:
                            target_vp = get_actual_victory_points(state, target_color)
                            score += target_vp * 2  # Target stronger players more aggressively
                            
                            # Target players with more cards
                            target_cards = player_num_resource_cards(state, target_color)
                            score += min(target_cards, 7)  # Cap the bonus to avoid excessive targeting
                        except Exception:
                            pass
            except Exception as e:
                logging.warning(f"Error evaluating robber action: {e}")
        
        # IMPROVED: Resource trading optimization
        elif action.action_type == ActionType.MARITIME_TRADE:
            score = 15  # Default trade score
            
            try:
                # Unpack trade details (giving, getting)
                if isinstance(action.value, tuple) and len(action.value) == 2:
                    giving, getting = action.value
                    
                    # Calculate trade value based on resource balance strategy
                    trade_value = self.evaluate_trade(game, giving, getting)
                    score = 15 + trade_value
            except Exception as e:
                logging.warning(f"Error evaluating trade: {e}")
            
        # For end turn
        elif action.action_type == ActionType.END_TURN:
            score = 10
            
        # For other action types
        else:
            score = 5
            
        # Add small randomness to avoid ties
        score += random.uniform(0, 0.1)
        
        return score
    
    def evaluate_road_potential(self, board, edge):
        """Evaluate a road's potential for future settlements"""
        future_settlement_score = 0
        
        try:
            # Get nodes connected by this edge
            nodes = None
            if hasattr(board.map, "edge_nodes"):
                nodes = board.map.edge_nodes.get(edge)
            elif hasattr(board.map, "edge_to_nodes"):
                nodes = board.map.edge_to_nodes.get(edge)
            else:
                # Assume edge itself is a tuple of nodes
                nodes = edge if isinstance(edge, tuple) and len(edge) == 2 else None
                
            if not nodes:
                return 0
                
            # Check each node for settlement potential
            for node in nodes:
                # Skip if already built on
                if node in board.settlements or node in board.cities:
                    continue
                    
                # Check distance rule (no adjacent settlements)
                can_build = True
                
                # Try to get adjacent nodes
                adjacent_nodes = []
                if hasattr(board.map, "adjacent_nodes"):
                    adjacent_nodes = board.map.adjacent_nodes.get(node, [])
                
                # Check if any adjacent nodes have settlements
                for adj_node in adjacent_nodes:
                    if adj_node in board.settlements or adj_node in board.cities:
                        can_build = False
                        break
                
                if not can_build:
                    continue
                
                # Node is potentially buildable - evaluate its resource value
                node_production = board.map.node_production.get(node, {})
                production_value = sum(node_production.values()) if node_production else 0
                node_value = production_value * 10
                
                # Calculate resource diversity
                diversity_value = len(node_production) * 2 if node_production else 0
                
                # Extra points for access to scarce resources
                resource_bonus = 0
                if node_production:
                    if BRICK in node_production:
                        resource_bonus += 5
                    if WOOD in node_production:
                        resource_bonus += 5
                    if WHEAT in node_production:
                        resource_bonus += 3
                    if ORE in node_production:
                        resource_bonus += 3
                
                future_settlement_score += node_value + diversity_value + resource_bonus
        except Exception as e:
            logging.warning(f"Error in road potential evaluation: {e}")
            
        return future_settlement_score

    def evaluate_trade(self, game, giving, getting):
        """Evaluate the value of a trade based on current needs"""
        state = game.state
        my_vp = get_actual_victory_points(state, self.color)
        my_resources = get_player_freqdeck(state, self.color)
        
        # Calculate what we need most urgently
        needed_for_settlement = {BRICK: 1, WOOD: 1, WHEAT: 1, SHEEP: 1}
        needed_for_city = {ORE: 3, WHEAT: 2}
        needed_for_road = {BRICK: 1, WOOD: 1}
        needed_for_dev_card = {ORE: 1, WHEAT: 1, SHEEP: 1}
        
        # Determine current goal based on game phase
        if my_vp < 4:
            primary_needs = needed_for_settlement
            secondary_needs = needed_for_road
        elif my_vp < 7:
            # In mid-game, balance between settlements and cities
            settlements = len([b for b in get_player_buildings(state, self.color) 
                               if b[0] == "SETTLEMENT"])
            if settlements >= 3:
                primary_needs = needed_for_city
            else:
                primary_needs = needed_for_settlement
            secondary_needs = needed_for_dev_card
        else:
            primary_needs = needed_for_city
            secondary_needs = needed_for_dev_card
        
        # Score the trade based on what we're getting vs giving
        trade_value = 0
        
        # Value what we're getting
        for resource, amount in getting.items():
            # Higher value if we need this resource for our primary goal
            if resource in primary_needs:
                current_amount = my_resources.get(resource, 0)
                needed_amount = primary_needs[resource]
                if current_amount < needed_amount:
                    trade_value += 10 * amount
                else:
                    trade_value += 5 * amount
            # Moderate value for secondary goal resources
            elif resource in secondary_needs:
                trade_value += 7 * amount
            # Base value for any resource
            else:
                trade_value += 3 * amount
        
        # Discount what we're giving away
        for resource, amount in giving.items():
            # Big discount if we have excess of this resource
            excess = my_resources.get(resource, 0) - 3
            if excess > 0:
                trade_value -= 2 * amount
            # Major discount if this is a resource we need
            elif resource in primary_needs:
                trade_value -= 8 * amount
            # Normal discount otherwise
            else:
                trade_value -= 5 * amount
                
        return trade_value
    
    def should_buy_development_card(self, game):
        """Decision framework for buying development cards"""
        state = game.state
        my_vp = get_actual_victory_points(state, self.color)
        my_resources = get_player_freqdeck(state, self.color)
        
        # Early game (VP < 4): Focus on expansion unless excess ore
        if my_vp < 4:
            # Buy dev card if we have excess ore and can't build settlements
            if my_resources.get(ORE, 0) >= 2 and (
                my_resources.get(BRICK, 0) == 0 or my_resources.get(WOOD, 0) == 0):
                return True
            return False
            
        # Mid game (VP 4-6): Balance expansion with knights/VP cards
        elif my_vp < 7:
            # If we're struggling to expand, invest in dev cards
            settlements = len([b for b in get_player_buildings(state, self.color) 
                               if b[0] == "SETTLEMENT"])
            if settlements >= 3:  # Hard to find new spots
                return True
                
            # If we have largest army or close to it, invest in knights
            largest_army = state.state_dict.get("largest_army_color", None)
            if largest_army != self.color and state.player_state[self.color].num_knights >= 1:
                return True
                
            return random.random() < 0.4  # 40% chance otherwise
            
        # Late game (VP 7+): Heavy focus on dev cards for VP
        else:
            return True  # Prioritize dev cards for hidden VP
    
    def late_game_strategy(self, game, playable_actions):
        """Special strategy for late game (6+ VP)"""
        state = game.state
        
        # 1. Prioritize hidden victory points (dev cards)
        dev_card_actions = [a for a in playable_actions 
                           if a.action_type == ActionType.BUY_DEVELOPMENT_CARD]
        if dev_card_actions and self.should_buy_development_card(game):
            logging.info("Late game strategy: Buying development card")
            return dev_card_actions[0]
            
        # 2. Prioritize city upgrades over new settlements
        city_actions = [a for a in playable_actions 
                       if a.action_type == ActionType.BUILD_CITY]
        if city_actions:
            # Find the city with highest production value
            best_city = None
            best_value = -1
            for action in city_actions:
                node_id = action.value
                node_production = game.state.board.map.node_production.get(node_id, {})
                production_value = sum(node_production.values()) if node_production else 0
                
                # Prioritize wheat/ore production for more cities
                resource_bonus = 0
                if node_production:
                    if ORE in node_production:
                        resource_bonus += node_production[ORE] * 0.5
                    if WHEAT in node_production:
                        resource_bonus += node_production[WHEAT] * 0.5
                        
                total_value = production_value + resource_bonus
                if total_value > best_value:
                    best_value = total_value
                    best_city = action
                    
            if best_city:
                logging.info(f"Late game strategy: Building city with value {best_value:.1f}")
                return best_city
                
        # 3. Consider aggressive trading for needed resources
        my_resources = get_player_freqdeck(state, self.color)
        
        # Resources needed for city (highest priority in late game)
        missing_for_city = {}
        for res, amount in {ORE: 3, WHEAT: 2}.items():
            if my_resources.get(res, 0) < amount:
                missing_for_city[res] = amount - my_resources.get(res, 0)
                
        # Evaluate trades that get us needed resources
        trade_actions = [a for a in playable_actions 
                      if a.action_type == ActionType.MARITIME_TRADE]
                      
        for action in trade_actions:
            try:
                outgoing, incoming = action.value
                if incoming in missing_for_city:
                    logging.info(f"Late game strategy: Trading for {incoming}")
                    return action
            except Exception:
                pass
        
        return None  # No special late-game action found
    
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