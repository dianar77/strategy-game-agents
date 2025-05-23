import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
from agents.fromScratchLLMStructured_player_v5_M.llm_tools import LLM


class FooPlayer(Player):
    def __init__(self, color=None, name=None):
        # Accept any color parameter instead of hardcoding
        super().__init__(color, name)
        self.llm = LLM()  # use self.llm.query_llm(str prompt) to query the LLM
        self.action_history = []  # Keep track of past actions

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
        
        # Print playable actions for debugging
        print(f"Available actions: {len(playable_actions)}")
        for i, action in enumerate(playable_actions[:5]):  # Print first 5 for brevity
            print(f"  {i}: {action}")
        if len(playable_actions) > 5:
            print(f"  ... and {len(playable_actions)-5} more")
            
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
        print("Falling back to simple heuristic strategy")
        return self._simple_heuristic(game, playable_actions)
    
    def _get_my_player_state(self, game):
        """
        Helper method to safely get player state regardless of color representation.
        Returns a tuple of (player_state, color_key) or (None, None) if not found.
        """
        try:
            # Direct access if color works as a key
            if self.color in game.state.player_state:
                return (game.state.player_state[self.color], self.color)
            
            # Try to find player state by matching player IDs
            for color_key, player_state in game.state.player_state.items():
                # Check for players array or direct equality
                if hasattr(game, 'players') and self in game.players:
                    index = game.players.index(self)
                    # Check if this is player P{index} (like P0, P1)
                    if str(color_key).startswith(f"P{index}_") or str(color_key) == f"P{index}":
                        return (player_state, color_key)
                
            # Alternative approach: just get the current player's state
            if hasattr(game.state, 'current_player_color'):
                current_color = game.state.current_player_color
                if current_color in game.state.player_state:
                    return (game.state.player_state[current_color], current_color)
            
            print(f"Could not find player state for color {self.color}")
            return (None, None)
            
        except Exception as e:
            print(f"Error getting player state: {e}")
            return (None, None)
    
    def _create_game_state_prompt(self, game, playable_actions):
        """
        Create a simplified prompt that doesn't rely on complex state access.
        """
        try:
            # Format available actions for LLM
            action_descriptions = [f"Action {i}: {action}" for i, action in enumerate(playable_actions)]
            action_list = "\n".join(action_descriptions)
            
            # Get basic game state info - safely
            my_state, my_key = self._get_my_player_state(game)
            
            # If we have player state, include it. Otherwise use simplified prompt
            if my_state is not None:
                # Try to extract some useful information if available
                victory_points = getattr(my_state, 'victory_points', "unknown")
                resources = getattr(my_state, 'resource_deck', "unknown")
                
                prompt = f"""
You are an AI assistant helping a player make strategic decisions in a game of Catan.

GAME STATE:
- My Color: {self.color}
- My Victory Points: {victory_points}
- My Resources: {resources}

AVAILABLE ACTIONS:
{action_list}

Which action should I choose? Analyze each option and recommend the best action by returning the action index (e.g., "Action 2").
Explain your reasoning briefly and provide your final recommendation at the end in the format: "RECOMMENDED ACTION: Action X"
"""
            else:
                # Simplified prompt if player state is not accessible
                prompt = f"""
You are an AI assistant helping a player make strategic decisions in a game of Catan.

AVAILABLE ACTIONS:
{action_list}

Which action should I choose? Analyze each option and recommend the best action by returning the action index (e.g., "Action 2").
Provide your final recommendation at the end in the format: "RECOMMENDED ACTION: Action X"
"""
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
    
    def _simple_heuristic(self, game, playable_actions):
        """
        Fallback strategy using simple heuristics when LLM fails.
        More sophisticated than just taking the first action.
        """
        # Prioritize actions by type
        priority_order = [
            ActionType.BUILD_CITY,  # Cities give more VP
            ActionType.BUILD_SETTLEMENT,  # Settlements give VP and resources
            ActionType.BUY_DEVELOPMENT_CARD,  # Development cards can be valuable
            ActionType.BUILD_ROAD,  # Roads help expand
            # FIXED: Removed ActionType.PLAY_DEVELOPMENT_CARD which doesn't exist
            ActionType.MOVE_ROBBER,  # Disrupt opponents
            ActionType.MARITIME_TRADE,  # Trade if we need resources
            ActionType.ROLL,  # Roll dice if nothing else to do
            ActionType.END_TURN,  # End turn as last resort
        ]
        
        # Try to find an action by priority
        for action_type in priority_order:
            matching_actions = [a for a in playable_actions if a.action_type == action_type]
            if matching_actions:
                chosen_action = matching_actions[0]
                print(f"Heuristic chose action type: {action_type}")
                return chosen_action
        
        # If no prioritized action found, choose the first available
        print("Choosing first action on default")
        return playable_actions[0]