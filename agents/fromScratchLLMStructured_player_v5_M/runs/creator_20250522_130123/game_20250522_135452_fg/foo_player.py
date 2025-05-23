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
            
            # Add information about actions
            prompt += f"Available Actions ({len(playable_actions)}):\n"
            for i, action in enumerate(playable_actions):
                prompt += f"Action {i}: {action}\n"
            
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