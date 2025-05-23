import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
from agents.fromScratchLLMStructured_player_v5_M.llm_tools import LLM
from catanatron.state_functions import (
    player_key,
    get_player_freqdeck,
    get_actual_victory_points, 
    get_visible_victory_points,
    get_player_buildings
)


class FooPlayer(Player):
    def __init__(self, color=None, name=None):
        # Accept any color parameter instead of hardcoding
        super().__init__(color, name)
        self.llm = LLM()  # use self.llm.query_llm(str prompt) to query the LLM
        self.action_history = []  # Keep track of past actions
        self.is_initial_placement_phase = True  # Track if we're in initial placement
        self.initial_placements_count = 0  # Track how many initial placements we've made

    def decide(self, game, playable_actions):
        """
        Decide which action to take based on LLM recommendations.
        
        Args:
            game (Game): complete game state. read-only.
                Defined in "catanatron/catanatron_core/catanatron/game.py"
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # Debug: Print player's assigned color and available colors
        print(f"Player color is: {self.color}")
        print(f"Game state player keys: {list(game.state.player_state.keys())}")
        
        # If no playable actions, return None (shouldn't happen but just in case)
        if not playable_actions:
            print("No playable actions available!")
            return None
        
        if len(playable_actions) == 1:
            print(f"Only one action available: {playable_actions[0]}")
            return playable_actions[0]
        
        # Print playable actions for debugging
        print(f"Available actions: {len(playable_actions)}")
        for i, action in enumerate(playable_actions[:5]):  # Print first 5 for brevity
            print(f"  {i}: {action}")
        if len(playable_actions) > 5:
            print(f"  ... and {len(playable_actions)-5} more")
        
        # Detect initial placement phase based on action types and game state
        # Initial placements are BUILD_SETTLEMENT actions during early game
        settlement_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        
        # Detect if we're in the initial placement phase
        # In initial placement, the player can only place settlements without having resources
        my_state = self._get_my_player_state(game)
        total_builds = 0
        if my_state:
            total_builds = my_state['buildings']['settlements'] + my_state['buildings']['cities']
        
        # If we have settlement options and we're still in initial phase (2 settlements + roads)
        if settlement_actions and self.is_initial_placement_phase and total_builds < 2:
            print(f"Evaluating initial settlement placement (placement #{self.initial_placements_count+1})...")
            chosen_action = self._evaluate_initial_placement(game, settlement_actions)
            self.initial_placements_count += 1
            
            # After 2 initial settlements, we're done with initial placement
            if self.initial_placements_count >= 2:
                self.is_initial_placement_phase = False
            
            return chosen_action
            
        # If this is the road after initial settlement, just pick the first one
        # (road placement is less critical than settlement placement)
        if self.is_initial_placement_phase and [a for a in playable_actions if a.action_type == ActionType.BUILD_ROAD]:
            road_action = next(a for a in playable_actions if a.action_type == ActionType.BUILD_ROAD)
            return road_action
            
        # Prepare a prompt for the LLM with game state information
        try:
            prompt = self._create_game_state_prompt(game, playable_actions)
            
            # Query the LLM for action recommendation
            print("Querying LLM for action recommendation...")
            llm_response = self.llm.query_llm(prompt)
            
            # Parse the LLM response to get the recommended action
            chosen_action = self._parse_llm_response(llm_response, playable_actions)
            
            # If we successfully got a valid action from the LLM
            if chosen_action is not None:
                print(f"LLM chose action: {chosen_action}")
                self.action_history.append(chosen_action)
                return chosen_action
                
        except Exception as e:
            print(f"Error in LLM decision process: {e}")
        
        # Fallback strategy if LLM fails
        print("Falling back to strategic heuristic")
        return self._strategic_heuristic(game, playable_actions)
    
    def _evaluate_initial_placement(self, game, settlement_actions):
        """
        Strategic evaluation of initial settlement placement options.
        This is one of the most critical decisions in the game.
        """
        if not settlement_actions:
            # Should never happen since this is called only with settlement actions
            print("Warning: No settlement actions provided for initial placement")
            return self._strategic_heuristic(game, settlement_actions)
            
        # Calculate scores for each settlement location
        settlement_scores = {}
        for action in settlement_actions:
            node_id = action.value  # The node where we're placing the settlement
            score = self._score_initial_settlement(game, node_id)
            settlement_scores[node_id] = score
            
        # Debug: Print top 3 settlement options with their scores
        top_options = sorted(settlement_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top settlement options: {top_options}")
        
        # Find the settlement action with the highest score
        best_node = max(settlement_scores, key=settlement_scores.get)
        best_action = next(a for a in settlement_actions if a.value == best_node)
        
        print(f"Chose initial settlement at node {best_node} with score {settlement_scores[best_node]}")
        return best_action
    
    def _score_initial_settlement(self, game, node_id):
        """
        Score a potential settlement location for initial placement.
        Higher scores are better.
        """
        try:
            score = 0
            board = game.board
            
            # 1. Get adjacent tiles and their resource types
            adjacent_tiles = []
            for tile_id, _ in board.map.adjacent_tiles[node_id].items():
                tile = board.map.tiles[tile_id]
                adjacent_tiles.append(tile)
            
            # 2. Resource probability (sum of probability values)
            probability_sum = 0
            resources_at_node = []
            for tile in adjacent_tiles:
                if hasattr(tile, 'resource') and tile.resource is not None:
                    # Get resource type
                    resources_at_node.append(tile.resource)
                    
                    # Add probability based on number
                    if hasattr(tile, 'number') and tile.number is not None:
                        # Calculate probability value (6 and 8 are highest at 5/36)
                        if tile.number in [6, 8]:
                            probability_sum += 5
                        elif tile.number in [5, 9]:
                            probability_sum += 4
                        elif tile.number in [4, 10]:
                            probability_sum += 3
                        elif tile.number in [3, 11]:
                            probability_sum += 2
                        elif tile.number in [2, 12]:
                            probability_sum += 1
            
            # 3. Resource diversity (unique resource types)
            unique_resources = set(resources_at_node)
            resource_diversity = len(unique_resources)
            
            # 4. Resource priority weights (brick and wood more valuable early)
            early_game_resources = 0
            if "BRICK" in resources_at_node:
                early_game_resources += 3
            if "WOOD" in resources_at_node:
                early_game_resources += 3
            if "WHEAT" in resources_at_node:
                early_game_resources += 2
            if "SHEEP" in resources_at_node:
                early_game_resources += 1
            if "ORE" in resources_at_node:
                early_game_resources += 1
                
            # 5. Port access (if we can determine this)
            port_bonus = 0
            try:
                # Check if this node has port access
                if hasattr(board.map, 'port_nodes') and node_id in board.map.port_nodes:
                    port_type = board.map.port_nodes[node_id]
                    # 3:1 ports are generally good
                    if port_type == "3:1":
                        port_bonus += 3
                    # Resource-specific ports are good if we produce that resource
                    elif port_type in resources_at_node:
                        port_bonus += 5
                    else:
                        port_bonus += 2
            except Exception as e:
                print(f"Error checking port access: {e}")
                
            # Calculate final score
            score = (probability_sum * 3) + (resource_diversity * 5) + early_game_resources + port_bonus
            
            return score
            
        except Exception as e:
            print(f"Error scoring settlement location: {e}")
            return 0
            
    def _get_my_player_state(self, game):
        """
        Get player state information using the proper state access functions.
        Returns a dictionary with relevant player state information.
        """
        try:
            my_key = player_key(game.state, self.color)
            resources = get_player_freqdeck(game.state, self.color)
            victory_points = get_visible_victory_points(game.state, self.color)
            actual_victory_points = get_actual_victory_points(game.state, self.color)
            settlements = get_player_buildings(game.state, self.color, "SETTLEMENT")
            cities = get_player_buildings(game.state, self.color, "CITY")
            roads = get_player_buildings(game.state, self.color, "ROAD")
            
            # Get opponent information as well
            opponents_info = []
            for color in game.state.colors:
                if color != self.color:  # This is an opponent
                    opponent_key = player_key(game.state, color)
                    opponent_vp = get_visible_victory_points(game.state, color)
                    opponent_settlements = get_player_buildings(game.state, color, "SETTLEMENT")
                    opponent_cities = get_player_buildings(game.state, color, "CITY")
                    opponent_roads = get_player_buildings(game.state, color, "ROAD")
                    
                    opponents_info.append({
                        "color": color,
                        "victory_points": opponent_vp,
                        "buildings": {
                            "settlements": len(opponent_settlements),
                            "cities": len(opponent_cities),
                            "roads": len(opponent_roads)
                        }
                    })
            
            return {
                "player_key": my_key,
                "resources": {
                    "WOOD": resources[0],
                    "BRICK": resources[1],
                    "SHEEP": resources[2],
                    "WHEAT": resources[3],
                    "ORE": resources[4]
                },
                "victory_points": victory_points,
                "actual_victory_points": actual_victory_points,
                "buildings": {
                    "settlements": len(settlements),
                    "cities": len(cities),
                    "roads": len(roads)
                },
                "opponents": opponents_info
            }
        except Exception as e:
            print(f"Error getting player state: {e}")
            return None
    
    def _create_game_state_prompt(self, game, playable_actions):
        """
        Create a detailed prompt describing the game state and available actions.
        """
        try:
            my_state = self._get_my_player_state(game)
            
            prompt = "Current Game State:\n"
            
            if my_state:
                prompt += f"Your color: {self.color}\n"
                prompt += f"Your victory points: {my_state['victory_points']}\n"
                prompt += f"Your resources: {my_state['resources']}\n"
                prompt += f"Your buildings: {my_state['buildings']}\n\n"
                
                # Add opponent information
                prompt += "Opponents:\n"
                for opponent in my_state['opponents']:
                    prompt += f"- {opponent['color']}: {opponent['victory_points']} victory points, "
                    prompt += f"Buildings: {opponent['buildings']}\n"
                prompt += "\n"
            
            # Add strategic guidance based on game state
            prompt += "Strategic Considerations:\n"
            
            # Early game strategy
            if my_state and my_state['buildings']['settlements'] <= 2:
                prompt += "- Early Game: Focus on expanding by building roads and settlements.\n"
                prompt += "- Prioritize securing BRICK and WOOD resources for expansion.\n"
            # Mid game strategy
            elif my_state and my_state['buildings']['settlements'] <= 4:
                prompt += "- Mid Game: Consider upgrading settlements to cities and buying development cards.\n"
                prompt += "- Prioritize WHEAT and ORE resources at this stage.\n"
            # Late game strategy 
            else:
                prompt += "- Late Game: Focus on direct path to victory points.\n"
                prompt += "- Prioritize actions that grant immediate or hidden victory points.\n"
            
            # Add information about actions
            prompt += f"\nAvailable Actions ({len(playable_actions)}):\n"
            for i, action in enumerate(playable_actions):
                prompt += f"Action {i}: {action}\n"
            
            # Add specific guidance for certain action types
            if any(a.action_type == ActionType.MOVE_ROBBER for a in playable_actions):
                prompt += "\nFor robber placement: Consider targeting the player with the most victory points.\n"
                
            if any(a.action_type == ActionType.BUY_DEVELOPMENT_CARD for a in playable_actions):
                prompt += "\nDevelopment cards provide hidden victory points, knights for largest army, and special abilities.\n"
            
            prompt += "\nAnalyze each available action and recommend the best one. Consider your current resources, "
            prompt += "victory points, and building opportunities. Aim to maximize your chances of winning by "
            prompt += "securing key resources and building strategically.\n\n"
            prompt += "Specify your choice by indicating 'RECOMMENDED ACTION: Action X' where X is the action index."
            
            return prompt
        except Exception as e:
            print(f"Error creating game state prompt: {e}")
            # Very simple fallback
            actions_text = "\n".join([f"Action {i}: {a}" for i, a in enumerate(playable_actions)])
            return f"Available actions:\n{actions_text}\nRecommend the best action as 'RECOMMENDED ACTION: Action X'."
    
    def _parse_llm_response(self, llm_response, playable_actions):
        """
        Parse the LLM response to extract the recommended action.
        Returns the chosen action or None if parsing fails.
        """
        try:
            print(f"Parsing LLM response: {llm_response[:100]}...")  # Print first 100 chars for debugging
            
            # Look for the explicit recommendation format
            if "RECOMMENDED ACTION: Action " in llm_response:
                parts = llm_response.split("RECOMMENDED ACTION: Action ")
                action_idx_str = parts[1].split()[0].strip()
                try:
                    action_idx = int(action_idx_str)
                    if 0 <= action_idx < len(playable_actions):
                        return playable_actions[action_idx]
                    else:
                        print(f"Action index {action_idx} out of range (0-{len(playable_actions)-1})")
                except ValueError:
                    print(f"Could not parse action index from '{action_idx_str}'")
            
            # Fallback: look for "Action X" pattern
            import re
            pattern = r"Action (\d+)"
            matches = re.findall(pattern, llm_response)
            
            if matches:
                # Take the last mentioned action index
                try:
                    action_idx = int(matches[-1])
                    if 0 <= action_idx < len(playable_actions):
                        return playable_actions[action_idx]
                    else:
                        print(f"Action index {action_idx} out of range (0-{len(playable_actions)-1})")
                except ValueError:
                    print(f"Could not parse action index from '{matches[-1]}'")
                    
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Raw LLM response: {llm_response}")
        
        return None
    
    def _strategic_heuristic(self, game, playable_actions):
        """
        Enhanced strategic heuristic for decision-making when LLM fails.
        Uses game state knowledge to make better decisions than simple heuristics.
        """
        # Special handling for robber placement
        robber_actions = [a for a in playable_actions if a.action_type == ActionType.MOVE_ROBBER]
        if robber_actions:
            return self._evaluate_robber_placement(game, robber_actions)
        
        # Special handling for building placements
        settlement_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        if settlement_actions:
            return self._evaluate_building_placement(game, settlement_actions, "SETTLEMENT")
            
        # Get player state for resource-based decisions
        my_state = self._get_my_player_state(game)
        
        # Development card purchase decision
        dev_card_actions = [a for a in playable_actions if a.action_type == ActionType.BUY_DEVELOPMENT_CARD]
        if dev_card_actions and my_state and self._should_buy_development_card(my_state):
            print("Strategic decision to buy development card")
            return dev_card_actions[0]
        
        # City upgrade prioritization based on resources
        city_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_CITY]
        if city_actions and my_state and my_state['resources']['WHEAT'] >= 2 and my_state['resources']['ORE'] >= 3:
            print("Strategic decision to build city")
            return city_actions[0]
            
        # Prioritize actions by type with enhanced strategic ordering
        priority_order = [
            ActionType.BUILD_CITY,      # Cities give 2 VP vs 1 for settlement
            ActionType.BUILD_SETTLEMENT, # New settlements for resource diversity
            ActionType.BUY_DEVELOPMENT_CARD,  # Can give VP, knights, or special abilities
            ActionType.BUILD_ROAD,      # Expansion for future settlements
            ActionType.MOVE_ROBBER,     # Disrupt opponents
            ActionType.MARITIME_TRADE,  # Trade if we need resources
            ActionType.ROLL,            # Roll dice if nothing else to do
            ActionType.END_TURN,        # End turn as last resort
        ]
        
        # Try to find an action by priority
        for action_type in priority_order:
            matching_actions = [a for a in playable_actions if a.action_type == action_type]
            if matching_actions:
                chosen_action = matching_actions[0]
                print(f"Strategic heuristic chose action type: {action_type}")
                return chosen_action
        
        # If no prioritized action found, choose the first available
        print("Choosing first action on default")
        return playable_actions[0]
    
    def _evaluate_robber_placement(self, game, robber_actions):
        """
        Strategic evaluation of robber placement options.
        Targets the leading opponent or blocks high-value resources.
        """
        try:
            # Find the opponent with the most victory points
            max_vp = 0
            target_opponent = None
            my_state = self._get_my_player_state(game)
            
            if my_state:
                for opponent in my_state['opponents']:
                    if opponent['victory_points'] > max_vp:
                        max_vp = opponent['victory_points']
                        target_opponent = opponent['color']
            
            # Score each robber action
            robber_scores = {}
            for action in robber_actions:
                # The hex being targeted and potentially the player to steal from
                hex_id = action.value[0]
                target_color = action.value[1] if len(action.value) > 1 else None
                
                score = 0
                
                # Prefer targeting the leading opponent
                if target_color and target_color == target_opponent:
                    score += 10
                
                # Try to evaluate the hex value (if we can)
                try:
                    tile = game.board.map.tiles.get(hex_id)
                    if tile and hasattr(tile, 'number'):
                        # Higher probability numbers get higher scores
                        if tile.number in [6, 8]:
                            score += 6
                        elif tile.number in [5, 9]:
                            score += 4
                        elif tile.number in [4, 10]:
                            score += 3
                        elif tile.number in [3, 11]:
                            score += 2
                except Exception as e:
                    print(f"Error evaluating hex: {e}")
                
                robber_scores[action] = score
                
            # Choose the highest scoring robber action
            best_action = max(robber_scores, key=robber_scores.get)
            print(f"Chose robber placement with score {robber_scores[best_action]}")
            return best_action
            
        except Exception as e:
            print(f"Error in robber placement: {e}")
            # Fallback to first robber action
            return robber_actions[0]
    
    def _evaluate_building_placement(self, game, building_actions, building_type):
        """
        Strategic evaluation of building placement options.
        """
        try:
            # For settlements, score based on resource access
            if building_type == "SETTLEMENT":
                settlement_scores = {}
                for action in building_actions:
                    node_id = action.value
                    # Simplified scoring - similar to initial settlement but less detailed
                    score = self._score_initial_settlement(game, node_id) * 0.7  # Less weight than initial
                    settlement_scores[action] = score
                
                best_action = max(settlement_scores, key=settlement_scores.get)
                print(f"Chose settlement location with score {settlement_scores[best_action]}")
                return best_action
            
            # For other building types, just pick the first one for now
            return building_actions[0]
            
        except Exception as e:
            print(f"Error in building placement: {e}")
            return building_actions[0]
    
    def _should_buy_development_card(self, my_state):
        """
        Determine if buying a development card is a good strategic move.
        """
        resources = my_state['resources']
        
        # Basic requirements
        has_resources = (resources["SHEEP"] >= 1 and resources["WHEAT"] >= 1 and resources["ORE"] >= 1)
        if not has_resources:
            return False
        
        # Factors favoring development cards:
        factors = 0
        
        # 1. If we already have most of the resources needed
        if resources["SHEEP"] >= 2 and resources["WHEAT"] >= 2 and resources["ORE"] >= 2:
            factors += 2  # We have excess resources for cards
        
        # 2. If we're close to winning
        if my_state["actual_victory_points"] >= 7:
            factors += 3  # Development cards might give victory points
        
        # 3. If we lack good building spots (indirect check via existing buildings)
        if my_state["buildings"]["settlements"] >= 3 and my_state["buildings"]["cities"] >= 1:
            factors += 1  # We may need alternate VP sources
        
        # Decide based on weighted factors
        return factors >= 2