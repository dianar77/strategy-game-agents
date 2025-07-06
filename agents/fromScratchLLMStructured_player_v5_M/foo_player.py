import os
from catanatron.models.player import Player, Color
from catanatron.game import Game
from catanatron.models.actions import ActionType
from catanatron.models.enums import (
    WOOD, BRICK, SHEEP, WHEAT, ORE, RESOURCES,
    KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT,
    SETTLEMENT, CITY, ROAD, ActionPrompt
)
from agents.fromScratchLLMStructured_player_v5_M.llm_tools import LLM


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
        self.llm = LLM()  # use self.llm.query_llm(str prompt) to query the LLM
        self.turn_count = 0
        self.debug = True  # Set to True for debug prints

    def decide(self, game, playable_actions):
        """
        Choose an action from playable_actions using LLM assistance.
        
        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        if not playable_actions:
            if self.debug:
                print("No playable actions available")
            return None

        self.turn_count += 1
        
        if self.debug:
            print(f"Turn {self.turn_count}: {len(playable_actions)} actions available")

        # Extract relevant game state information
        game_state = self._extract_game_state(game)
        
        # Format actions for the LLM
        action_descriptions = self._format_actions(playable_actions)
        
        # Create prompt for LLM
        prompt = self._create_llm_prompt(game_state, action_descriptions)
        
        try:
            # Query the LLM
            if self.debug:
                print("Querying LLM for decision...")
            
            llm_response = self.llm.query_llm(prompt)
            
            if self.debug:
                print(f"LLM Response: {llm_response[:100]}...")
            
            # Parse LLM response to select an action
            selected_action = self._parse_llm_response(llm_response, playable_actions)
            
            if selected_action is not None:
                if self.debug:
                    print(f"Selected action: {selected_action.action_type} with value {selected_action.value}")
                return selected_action
            
        except Exception as e:
            if self.debug:
                print(f"Error with LLM: {str(e)}")
        
        # Fallback strategy if LLM fails or returns invalid selection
        return self._fallback_strategy(playable_actions)

    def _extract_game_state(self, game):
        """Extract relevant information from the game state."""
        state = game.state
        color = self.color
        
        # Get player resources
        resources = {}
        resource_types = [WOOD, BRICK, SHEEP, WHEAT, ORE]  # Updated: use constants instead of enum
        for resource_type in resource_types:
            resource_key = f"P{state.colors.index(color)}_{resource_type}"
            resources[resource_type] = state.player_state.get(resource_key, 0)
        
        # Get buildings
        buildings = state.buildings_by_color.get(color, {})
        settlements = [node for node, building_type in buildings.items() if building_type == SETTLEMENT]
        cities = [node for node, building_type in buildings.items() if building_type == CITY]
        roads = [edge for edge, road_color in state.board.roads.items() if road_color == color]
        
        # Get board state
        is_initial_build = state.is_initial_build_phase
        current_prompt = state.current_prompt
        
        # Get victory points
        victory_points = 0
        for settlement in settlements:
            victory_points += 1
        for city in cities:
            victory_points += 2
        
        # Get other players' information
        other_players = {}
        for other_color in state.colors:
            if other_color != color:
                player_idx = state.colors.index(other_color)
                other_buildings = state.buildings_by_color.get(other_color, {})
                other_settlements = [node for node, building_type in other_buildings.items() if building_type == SETTLEMENT]
                other_cities = [node for node, building_type in other_buildings.items() if building_type == CITY]
                other_roads = [edge for edge, road_color in state.board.roads.items() if road_color == other_color]
                
                other_vp = len(other_settlements) + 2 * len(other_cities)
                
                other_players[other_color.name] = {
                    "settlements": len(other_settlements),
                    "cities": len(other_cities),
                    "roads": len(other_roads),
                    "victory_points": other_vp
                }
        
        # Put together game state summary
        game_state = {
            "resources": resources,
            "buildings": {
                "settlements": len(settlements),
                "cities": len(cities),
                "roads": len(roads)
            },
            "victory_points": victory_points,
            "is_initial_build": is_initial_build,
            "current_prompt": current_prompt.name if hasattr(current_prompt, "name") else str(current_prompt),
            "other_players": other_players,
            "turn_count": self.turn_count
        }
        
        return game_state

    def _format_actions(self, playable_actions):
        """Format the playable actions for the LLM."""
        action_descriptions = []
        
        for i, action in enumerate(playable_actions):
            action_type = action.action_type.name
            value = action.value
            
            description = f"Action {i}: {action_type}"
            
            # Add more details based on action type
            if action_type == "BUILD_SETTLEMENT":
                description += f" at node {value}"
            elif action_type == "BUILD_CITY":
                description += f" at node {value}"
            elif action_type == "BUILD_ROAD":
                description += f" from {value[0]} to {value[1]}"
            elif action_type == "BUY_DEVELOPMENT_CARD":
                description += " (costs 1 ORE, 1 WHEAT, 1 SHEEP)"
            elif action_type == "PLAY_YEAR_OF_PLENTY":
                description += f" selecting {value[0]} and {value[1]}"
            elif action_type == "PLAY_MONOPOLY":
                description += f" selecting resource {value}"
            elif action_type == "MARITIME_TRADE":
                description += f" giving {value[0]} {value[1]} for 1 {value[2]}"
            elif action_type == "MOVE_ROBBER":
                description += f" to coordinate {value[0]} stealing from {value[1]}"
            
            action_descriptions.append(description)
        
        return action_descriptions

    def _create_llm_prompt(self, game_state, action_descriptions):
        """Create a prompt for the LLM."""
        prompt = "You are an AI agent playing Catan. Please evaluate the current game state and select the best action to take.\n\n"
        
        # Game state information
        prompt += "GAME STATE:\n"
        prompt += f"Turn: {game_state['turn_count']}\n"
        prompt += f"Current phase: {game_state['current_prompt']}\n"
        prompt += f"Initial build phase: {game_state['is_initial_build']}\n"
        
        # Resources
        prompt += "\nMY RESOURCES:\n"
        for resource, count in game_state['resources'].items():
            prompt += f"- {resource}: {count}\n"
        
        # Buildings and VP
        prompt += "\nMY BUILDINGS:\n"
        prompt += f"- Settlements: {game_state['buildings']['settlements']}\n"
        prompt += f"- Cities: {game_state['buildings']['cities']}\n"
        prompt += f"- Roads: {game_state['buildings']['roads']}\n"
        prompt += f"- Victory Points: {game_state['victory_points']}\n"
        
        # Other players
        prompt += "\nOTHER PLAYERS:\n"
        for color, data in game_state['other_players'].items():
            prompt += f"Player {color}:\n"
            prompt += f"- Settlements: {data['settlements']}\n"
            prompt += f"- Cities: {data['cities']}\n"
            prompt += f"- Roads: {data['roads']}\n"
            prompt += f"- Victory Points: {data['victory_points']}\n"
        
        # Available actions
        prompt += "\nAVAILABLE ACTIONS:\n"
        for i, description in enumerate(action_descriptions):
            prompt += f"{description}\n"
        
        # Strategy guidance
        prompt += "\nSTRATEGY CONSIDERATIONS:\n"
        prompt += "1. In the early game, focus on resource acquisition and expansion.\n"
        prompt += "2. Build settlements on spots with good resource diversity and probability.\n"
        prompt += "3. Consider upgrading to cities when you have enough resources.\n"
        prompt += "4. Think about resource scarcity and what you need for future turns.\n"
        prompt += "5. Development cards can provide victory points or special advantages.\n"
        
        # Request format
        prompt += "\nPlease analyze the game state and available actions. Return your response in the following format:\n"
        prompt += "SELECTED: Action X\n"
        prompt += "REASONING: Your explanation of why this action is best...\n"
        
        return prompt

    def _parse_llm_response(self, response, playable_actions):
        """Parse the LLM response to get the selected action."""
        try:
            # Look for the action selection in the LLM response
            if "SELECTED:" in response:
                selected_line = [line for line in response.split('\n') if "SELECTED:" in line][0]
                action_str = selected_line.split("SELECTED:")[1].strip()
                
                # Extract the action index
                action_index = None
                for word in action_str.split():
                    if word.lower() == "action":
                        continue
                    try:
                        action_index = int(word.strip())
                        break
                    except ValueError:
                        continue
                
                if action_index is not None and 0 <= action_index < len(playable_actions):
                    return playable_actions[action_index]
            
            if self.debug:
                print("Could not parse action selection from LLM response")
            
            return None
        
        except Exception as e:
            if self.debug:
                print(f"Error parsing LLM response: {str(e)}")
            return None

    def _fallback_strategy(self, playable_actions):
        """Fallback strategy when LLM fails to provide a valid action."""
        if self.debug:
            print("Using fallback strategy")
        
        # Define action priorities (higher number = higher priority)
        action_priorities = {
            ActionType.BUILD_CITY: 5,
            ActionType.BUILD_SETTLEMENT: 4,
            ActionType.BUILD_ROAD: 3,
            ActionType.BUY_DEVELOPMENT_CARD: 2,
            ActionType.PLAY_KNIGHT_CARD: 2,
            ActionType.MARITIME_TRADE: 1
        }
        
        # Find the action with the highest priority
        best_action = None
        best_priority = -1
        
        for action in playable_actions:
            priority = action_priorities.get(action.action_type, 0)
            if priority > best_priority:
                best_action = action
                best_priority = priority
        
        if best_action:
            if self.debug:
                print(f"Fallback strategy selected: {best_action.action_type}")
            return best_action
        
        # If no prioritized action is found, return the first action
        if self.debug:
            print("Fallback to first action")
        return playable_actions[0]