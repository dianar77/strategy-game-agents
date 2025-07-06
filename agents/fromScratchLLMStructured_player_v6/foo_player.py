import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
from catanatron.models.enums import (
    WOOD, BRICK, SHEEP, WHEAT, ORE, RESOURCES,
    KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT,
    SETTLEMENT, CITY, ROAD, ActionPrompt
)
# ADK version imports the ADK-compatible LLM tools
from agents.fromScratchLLMStructured_player_v6.llm_tools_adk import LLM


class FooPlayer(Player):
    """
    ADK-compatible version of FooPlayer for playing Catan.
    
    This AI player uses Google Gemini (via ADK tools) to make strategic decisions
    in the board game Catan. It analyzes the current game state, available actions,
    and uses an LLM to select the optimal move based on game strategy.
    
    Key features:
    - Uses Google Gemini LLM through ADK-compatible interface
    - Implements strategic reasoning for Catan gameplay
    - Includes fallback decision-making when LLM fails
    - Provides detailed game state analysis to the LLM
    """
    
    def __init__(self, name=None):
        """
        Initialize the FooPlayer with ADK-compatible LLM backend.
        
        Args:
            name (str, optional): Player name for identification
        """
        super().__init__(Color.BLUE, name)
        self.llm = LLM()  # Initialize ADK-compatible LLM (Google Gemini)
        self.turn_count = 0  # Track number of turns played
        self.debug = True  # Enable debug output for troubleshooting

    def decide(self, game, playable_actions):
        """
        Main decision-making method called by the game engine.
        
        This method coordinates the entire decision process:
        1. Extracts relevant game state information
        2. Formats available actions for LLM understanding
        3. Creates a strategic prompt for the LLM
        4. Queries the LLM for decision-making
        5. Parses the LLM response to select an action
        6. Falls back to heuristic strategy if LLM fails
        
        Args:
            game (Game): Complete game state (read-only access)
            playable_actions (Iterable[Action]): Valid actions to choose from
            
        Returns:
            Action: Selected action from playable_actions, or None if no actions available
        """
        # Handle edge case of no available actions
        if not playable_actions:
            if self.debug:
                print("No playable actions available")
            return None

        # Increment turn counter for tracking game progression
        self.turn_count += 1
        
        if self.debug:
            print(f"Turn {self.turn_count}: {len(playable_actions)} actions available")

        # Step 1: Extract and structure current game state for LLM analysis
        game_state = self._extract_game_state(game)
        
        # Step 2: Convert actions into human-readable descriptions
        action_descriptions = self._format_actions(playable_actions)
        
        # Step 3: Construct comprehensive prompt for strategic decision-making
        prompt = self._create_llm_prompt(game_state, action_descriptions)
        
        try:
            # Step 4: Query the LLM using Google Gemini via ADK interface
            if self.debug:
                print("Querying ADK LLM (Google Gemini) for decision...")
            
            llm_response = self.llm.query_llm(prompt)
            
            if self.debug:
                print(f"ADK LLM Response: {llm_response[:100]}...")
            
            # Step 5: Parse LLM response to extract action selection
            selected_action = self._parse_llm_response(llm_response, playable_actions)
            
            # Return LLM-selected action if parsing was successful
            if selected_action is not None:
                if self.debug:
                    print(f"Selected action: {selected_action.action_type} with value {selected_action.value}")
                return selected_action
            
        except Exception as e:
            # Log LLM errors for debugging
            if self.debug:
                print(f"Error with ADK LLM: {str(e)}")
        
        # Step 6: Use fallback strategy if LLM fails or returns invalid selection
        return self._fallback_strategy(playable_actions)

    def _extract_game_state(self, game):
        """
        Extract and organize relevant game state information for LLM analysis.
        
        This method processes the raw game state into a structured format that
        the LLM can easily understand and reason about. It includes:
        - Player's current resources
        - Building counts and positions
        - Victory points calculation
        - Other players' status
        - Game phase information
        
        Args:
            game (Game): The current game instance
            
        Returns:
            dict: Structured game state information
        """
        state = game.state
        color = self.color
        
        # Extract current player's resource counts
        # Resources are stored in game state with keys like "P0_WOOD", "P1_BRICK", etc.
        resources = {}
        resource_types = [WOOD, BRICK, SHEEP, WHEAT, ORE]  # All resource types in Catan
        for resource_type in resource_types:
            # Build the state key for this player's resources
            resource_key = f"P{state.colors.index(color)}_{resource_type}"
            resources[resource_type] = state.player_state.get(resource_key, 0)
        
        # Extract building information from game state
        buildings = state.buildings_by_color.get(color, {})
        # Separate settlements and cities by building type
        settlements = [node for node, building_type in buildings.items() if building_type == SETTLEMENT]
        cities = [node for node, building_type in buildings.items() if building_type == CITY]
        # Roads are stored separately in the board state
        roads = state.board.roads_by_color.get(color, [])
        
        # Extract game phase information
        is_initial_build = state.is_initial_build_phase  # Special initial placement phase
        current_prompt = state.current_prompt  # What type of action is currently required
        
        # Calculate victory points from buildings
        # Settlements = 1 VP each, Cities = 2 VP each
        victory_points = 0
        for settlement in settlements:
            victory_points += 1
        for city in cities:
            victory_points += 2
        
        # Gather information about other players for competitive analysis
        other_players = {}
        for other_color in state.colors:
            if other_color != color:  # Skip self
                player_idx = state.colors.index(other_color)
                # Extract other player's buildings
                other_buildings = state.buildings_by_color.get(other_color, {})
                other_settlements = [node for node, building_type in other_buildings.items() if building_type == SETTLEMENT]
                other_cities = [node for node, building_type in other_buildings.items() if building_type == CITY]
                other_roads = state.board.roads_by_color.get(other_color, [])
                
                # Calculate other player's victory points
                other_vp = len(other_settlements) + 2 * len(other_cities)
                
                # Store competitor information
                other_players[other_color.name] = {
                    "settlements": len(other_settlements),
                    "cities": len(other_cities),
                    "roads": len(other_roads),
                    "victory_points": other_vp
                }
        
        # Compile all game state information into structured format
        game_state = {
            "resources": resources,  # Current resource counts
            "buildings": {  # Current building counts
                "settlements": len(settlements),
                "cities": len(cities),
                "roads": len(roads)
            },
            "victory_points": victory_points,  # Current VP total
            "is_initial_build": is_initial_build,  # Game phase flag
            "current_prompt": current_prompt.name if hasattr(current_prompt, "name") else str(current_prompt),
            "other_players": other_players,  # Competitor status
            "turn_count": self.turn_count  # Game progression tracking
        }
        
        return game_state

    def _format_actions(self, playable_actions):
        """
        Convert playable actions into human-readable descriptions for LLM.
        
        This method takes the raw action objects and converts them into
        descriptive text that the LLM can understand and reason about.
        Each action gets a clear description with relevant details.
        
        Args:
            playable_actions (list): List of Action objects from the game engine
            
        Returns:
            list: List of human-readable action descriptions
        """
        action_descriptions = []
        
        # Process each action and create descriptive text
        for i, action in enumerate(playable_actions):
            action_type = action.action_type.name  # Get action type name (e.g., "BUILD_SETTLEMENT")
            value = action.value  # Get action-specific data (e.g., node ID, coordinates)
            
            # Start with basic action description
            description = f"Action {i}: {action_type}"
            
            # Add specific details based on action type
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
        """
        Construct a comprehensive prompt for LLM strategic decision-making.
        
        This method creates a detailed prompt that includes:
        - Current game state analysis
        - Available action descriptions
        - Strategic considerations and guidelines
        - Specific response format requirements
        
        The prompt is designed to help the LLM make informed strategic decisions
        by providing all necessary context and guidance.
        
        Args:
            game_state (dict): Structured game state information
            action_descriptions (list): Human-readable action descriptions
            
        Returns:
            str: Complete prompt for LLM query
        """
        # Start with role definition and task description
        prompt = "You are an AI agent playing Catan. Please evaluate the current game state and select the best action to take.\n\n"
        
        # Section 1: Current game state information
        prompt += "GAME STATE:\n"
        prompt += f"Turn: {game_state['turn_count']}\n"
        prompt += f"Current phase: {game_state['current_prompt']}\n"
        prompt += f"Initial build phase: {game_state['is_initial_build']}\n"
        
        # Section 2: Player's current resources
        prompt += "\nMY RESOURCES:\n"
        for resource, count in game_state['resources'].items():
            prompt += f"- {resource}: {count}\n"
        
        # Section 3: Player's current buildings and victory points
        prompt += "\nMY BUILDINGS:\n"
        prompt += f"- Settlements: {game_state['buildings']['settlements']}\n"
        prompt += f"- Cities: {game_state['buildings']['cities']}\n"
        prompt += f"- Roads: {game_state['buildings']['roads']}\n"
        prompt += f"- Victory Points: {game_state['victory_points']}\n"
        
        # Section 4: Competitor analysis
        prompt += "\nOTHER PLAYERS:\n"
        for color, data in game_state['other_players'].items():
            prompt += f"Player {color}:\n"
            prompt += f"- Settlements: {data['settlements']}\n"
            prompt += f"- Cities: {data['cities']}\n"
            prompt += f"- Roads: {data['roads']}\n"
            prompt += f"- Victory Points: {data['victory_points']}\n"
        
        # Section 5: Available actions for this turn
        prompt += "\nAVAILABLE ACTIONS:\n"
        for i, description in enumerate(action_descriptions):
            prompt += f"{description}\n"
        
        # Section 6: Strategic guidance for decision-making
        prompt += "\nSTRATEGY CONSIDERATIONS:\n"
        prompt += "1. In the early game, focus on resource acquisition and expansion.\n"
        prompt += "2. Build settlements on spots with good resource diversity and probability.\n"
        prompt += "3. Consider upgrading to cities when you have enough resources.\n"
        prompt += "4. Think about resource scarcity and what you need for future turns.\n"
        prompt += "5. Development cards can provide victory points or special advantages.\n"
        
        # Section 7: Response format specification
        prompt += "\nPlease analyze the game state and available actions. Return your response in the following format:\n"
        prompt += "SELECTED: Action X\n"
        prompt += "REASONING: Your explanation of why this action is best...\n"
        
        return prompt

    def _parse_llm_response(self, response, playable_actions):
        """
        Parse the LLM response to extract the selected action.
        
        This method looks for the "SELECTED:" keyword in the LLM response
        and attempts to extract the action index. It handles various
        response formats and provides error handling for invalid selections.
        
        Args:
            response (str): Raw response text from the LLM
            playable_actions (list): List of available actions for validation
            
        Returns:
            Action or None: Selected action if parsing successful, None otherwise
        """
        try:
            # Look for the action selection marker in the LLM response
            if "SELECTED:" in response:
                # Extract the line containing the selection
                selected_line = [line for line in response.split('\n') if "SELECTED:" in line][0]
                action_str = selected_line.split("SELECTED:")[1].strip()
                
                # Parse the action index from the selection string
                # Look for the first integer in the action string
                action_index = None
                for word in action_str.split():
                    if word.lower() == "action":  # Skip the word "action"
                        continue
                    try:
                        action_index = int(word.strip())
                        break  # Found a valid integer
                    except ValueError:
                        continue  # Keep looking for an integer
                
                # Validate the action index and return the corresponding action
                if action_index is not None and 0 <= action_index < len(playable_actions):
                    return playable_actions[action_index]
            
            # Log parsing failure for debugging
            if self.debug:
                print("Could not parse action selection from ADK LLM response")
            
            return None
        
        except Exception as e:
            # Handle any unexpected errors during parsing
            if self.debug:
                print(f"Error parsing ADK LLM response: {str(e)}")
            return None

    def _fallback_strategy(self, playable_actions):
        """
        Implement fallback decision-making when LLM fails or returns invalid action.
        
        This method provides a simple heuristic-based strategy that prioritizes
        actions based on their general strategic value in Catan. It ensures
        the player can always make a valid move even when the LLM fails.
        
        Priority order (highest to lowest):
        1. Build City - High VP value and resource generation
        2. Build Settlement - VP and resource access
        3. Build Road - Expansion and longest road potential
        4. Buy Development Card - VP and special abilities
        5. Play Knight Card - Robber control
        6. Maritime Trade - Resource conversion
        
        Args:
            playable_actions (list): Available actions to choose from
            
        Returns:
            Action: Selected action using heuristic strategy
        """
        if self.debug:
            print("Using fallback strategy (ADK version)")
        
        # Define strategic priorities for different action types
        # Higher numbers indicate higher priority
        action_priorities = {
            ActionType.BUILD_CITY: 5,           # Highest priority - 2 VP + better resources
            ActionType.BUILD_SETTLEMENT: 4,     # High priority - 1 VP + resource access
            ActionType.BUILD_ROAD: 3,           # Medium priority - expansion + longest road
            ActionType.BUY_DEVELOPMENT_CARD: 2, # Lower priority - potential VP/abilities
            ActionType.PLAY_KNIGHT_CARD: 2,     # Lower priority - robber control
            ActionType.MARITIME_TRADE: 1        # Lowest priority - resource conversion
        }
        
        # Find the action with the highest strategic priority
        best_action = None
        best_priority = -1
        
        for action in playable_actions:
            priority = action_priorities.get(action.action_type, 0)  # Default priority 0
            if priority > best_priority:
                best_action = action
                best_priority = priority
        
        # Return the highest priority action if found
        if best_action:
            if self.debug:
                print(f"Fallback strategy selected: {best_action.action_type}")
            return best_action
        
        # Ultimate fallback - return the first available action
        # This ensures we always have a valid move
        if self.debug:
            print("Fallback to first action")
        return playable_actions[0] 