import os
from catanatron.models.player import Player, Color
from catanatron.models.actions import Action, ActionType
from catanatron.models.enums import RESOURCES, WOOD, BRICK, SHEEP, WHEAT, ORE
from catanatron.state_functions import (
    get_player_freqdeck,
    player_num_resource_cards,
    get_player_buildings, 
    get_actual_victory_points
)
from agents.fromScratchLLMStructured_player_v5_M.llm_tools import LLM


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
        self.llm = LLM()  # use self.llm.query_llm(str prompt) to query the LLM

    def decide(self, game, playable_actions):
        """
        Make a decision using LLM based on the current game state and available actions.
        
        Args:
            game (Game): complete game state. read-only. 
                Defined in in "catanatron/catanatron_core/catanatron/game.py"
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        if not playable_actions:
            return None

        # Create a prompt with game state information and available actions
        prompt = self._create_prompt(game, playable_actions)
        
        # Query the LLM for a decision
        print("Querying LLM for decision...")
        try:
            llm_response = self.llm.query_llm(prompt)
            print(f"LLM Response: {llm_response}")
            
            # Parse the LLM response to get an action index
            action_index = self._parse_llm_response(llm_response, len(playable_actions))
            
            # Use the LLM's choice if valid, otherwise default to the first action
            if action_index is not None:
                print(f"Choosing Action {action_index} based on LLM recommendation")
                return playable_actions[action_index]
        except Exception as e:
            print(f"Error querying LLM: {str(e)}")
        
        # Fallback to first action if LLM fails
        print("Falling back to first action")
        return playable_actions[0]

    def _create_prompt(self, game, playable_actions):
        """
        Create a prompt describing the game state and available actions.
        
        Args:
            game: The current game state
            playable_actions: List of available actions
        Returns:
            str: A prompt for the LLM
        """
        # Get player resources
        my_resources = get_player_freqdeck(game.state, self.color)
        resource_names = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
        resource_str = ", ".join([f"{resource_names[i]}: {my_resources[i]}" for i in range(len(resource_names))])
        
        # Get player buildings
        my_settlements = get_player_buildings(game.state, self.color, "SETTLEMENT")
        my_cities = get_player_buildings(game.state, self.color, "CITY")
        my_roads = get_player_buildings(game.state, self.color, "ROAD")
        
        # Get current victory points
        my_vp = get_actual_victory_points(game.state, self.color)
        
        # Get opponent information
        other_colors = [c for c in game.state.colors if c != self.color]
        opponent_info = ""
        for color in other_colors:
            opp_vp = get_actual_victory_points(game.state, color)
            opp_settlements = len(get_player_buildings(game.state, color, "SETTLEMENT"))
            opp_cities = len(get_player_buildings(game.state, color, "CITY"))
            opponent_info += f"Opponent {color} has {opp_vp} victory points, {opp_settlements} settlements, {opp_cities} cities\n"
        
        # Format the available actions
        actions_str = ""
        for i, action in enumerate(playable_actions):
            action_desc = self._format_action(action)
            actions_str += f"{i}: {action_desc}\n"
        
        # Create the full prompt
        prompt = f"""
You are an AI playing Catan. Here is the current game state:

YOUR STATE:
- Resources: {resource_str}
- Buildings: {len(my_settlements)} settlements, {len(my_cities)} cities, {len(my_roads)} roads
- Victory Points: {my_vp}

OPPONENTS:
{opponent_info}

AVAILABLE ACTIONS:
{actions_str}

Analyze the game state and choose the best action. Consider your resources, building options, 
and opponent positions. What is the optimal move for long-term victory?

Return only the index number of the best action to take (e.g., "2").
"""
        
        print("Prompt created for LLM:")
        print(prompt)
        return prompt

    def _format_action(self, action):
        """Format an action into a readable string."""
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            return f"BUILD_SETTLEMENT at node {action.value}"
        elif action.action_type == ActionType.BUILD_CITY:
            return f"BUILD_CITY at node {action.value}"
        elif action.action_type == ActionType.BUILD_ROAD:
            return f"BUILD_ROAD at edge {action.value}"
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return "BUY_DEVELOPMENT_CARD"
        elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
            return f"PLAY_KNIGHT_CARD and move robber to {action.value}"
        elif action.action_type == ActionType.MARITIME_TRADE:
            return f"MARITIME_TRADE: {action.value}"
        elif action.action_type == ActionType.ROLL:
            return "ROLL the dice"
        elif action.action_type == ActionType.END_TURN:
            return "END_TURN"
        else:
            return f"{action.action_type} {action.value}"

    def _parse_llm_response(self, response, num_actions):
        """
        Parse the LLM response to extract an action index.
        
        Args:
            response: The LLM response string
            num_actions: The number of available actions
        Returns:
            int or None: The parsed action index or None if invalid
        """
        try:
            # Try to extract a number from the response
            # First check for simple number response
            response = response.strip()
            if response.isdigit():
                action_index = int(response)
            else:
                # Try to find any number in the response
                import re
                numbers = re.findall(r'\d+', response)
                if numbers:
                    action_index = int(numbers[0])
                else:
                    print("No number found in LLM response")
                    return None
            
            # Validate the action index
            if 0 <= action_index < num_actions:
                return action_index
            else:
                print(f"Invalid action index: {action_index}, must be between 0 and {num_actions-1}")
                return None
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            return None