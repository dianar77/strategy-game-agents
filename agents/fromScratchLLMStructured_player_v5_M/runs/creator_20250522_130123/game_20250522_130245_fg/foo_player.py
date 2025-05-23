import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType
from agents.fromScratchLLMStructured_player_v5_M.llm_tools import LLM


class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
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
        # If no playable actions, return None (shouldn't happen but just in case)
        if not playable_actions:
            print("No playable actions available!")
            return None
            
        # Prepare a prompt for the LLM with game state information
        prompt = self._create_game_state_prompt(game, playable_actions)
        
        try:
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
            print(f"Error using LLM for decision: {e}")
        
        # Fallback strategy if LLM fails
        print("Falling back to simple heuristic strategy")
        return self._simple_heuristic(game, playable_actions)
    
    def _create_game_state_prompt(self, game, playable_actions):
        """
        Create a detailed prompt for the LLM with game state information.
        """
        # Get my state
        my_state = game.state.player_state[self.color]
        my_resources = my_state.resource_deck
        
        # Get opponent information
        opponents = []
        for color, state in game.state.player_state.items():
            if color != self.color:
                opponents.append({
                    "color": color.name,
                    "victory_points": state.victory_points,
                    "settlements": len(state.buildings[ActionType.BUILD_SETTLEMENT]),
                    "cities": len(state.buildings[ActionType.BUILD_CITY]),
                    "roads": len(state.buildings[ActionType.BUILD_ROAD]),
                })
        
        # Format available actions for LLM
        action_descriptions = []
        for i, action in enumerate(playable_actions):
            action_descriptions.append(f"Action {i}: {action}")
        
        # Create the prompt
        prompt = f"""
You are an AI assistant helping a player make strategic decisions in a game of Catan.

GAME STATE:
- My Color: {self.color.name}
- My Victory Points: {my_state.victory_points}
- My Resources: {dict(my_resources)}
- My Settlements: {len(my_state.buildings[ActionType.BUILD_SETTLEMENT])}
- My Cities: {len(my_state.buildings[ActionType.BUILD_CITY])}
- My Roads: {len(my_state.buildings[ActionType.BUILD_ROAD])}

OPPONENTS:
{opponents}

AVAILABLE ACTIONS:
{action_descriptions}

Which action should I choose? Analyze each option and recommend the best action by returning the action index (e.g., "Action 2").
Explain your reasoning briefly and provide your final recommendation at the end in the format: "RECOMMENDED ACTION: Action X"
"""
        return prompt
    
    def _parse_llm_response(self, llm_response, playable_actions):
        """
        Parse the LLM response to extract the recommended action.
        Returns the chosen action or None if parsing fails.
        """
        try:
            # Look for the explicit recommendation format
            if "RECOMMENDED ACTION: Action " in llm_response:
                parts = llm_response.split("RECOMMENDED ACTION: Action ")
                action_idx = int(parts[1].split()[0])
                if 0 <= action_idx < len(playable_actions):
                    return playable_actions[action_idx]
            
            # Fallback: look for "Action X" pattern
            import re
            pattern = r"Action (\d+)"
            matches = re.findall(pattern, llm_response)
            
            if matches:
                # Take the last mentioned action index
                action_idx = int(matches[-1])
                if 0 <= action_idx < len(playable_actions):
                    return playable_actions[action_idx]
                    
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
            ActionType.PLAY_DEVELOPMENT_CARD,  # Use cards we've bought
            ActionType.MOVE_ROBBER,  # Disrupt opponents
            ActionType.MARITIME_TRADE,  # Trade if we need resources
        ]
        
        # Sort actions by priority
        for action_type in priority_order:
            for action in playable_actions:
                if action.action_type == action_type:
                    print(f"Heuristic chose action type: {action_type}")
                    return action
        
        # If no prioritized action found, choose the first available
        print("No prioritized action found, choosing first action")
        return playable_actions[0]