import os
from catanatron import Player
from catanatron.game import Game
from agents.fromScratchLLMStructured_player.llm_tools import LLM
from catanatron.models.actions import ActionType

class FooPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name)
        self.llm = LLM()  # Includes LLM class with llm.query_llm(prompt: str) -> str method
        self.resources = {}  # Initialize resources as an empty dictionary
        self.settlements = []  # Initialize settlements as an empty list
        self.roads = []  # Initialize roads as an empty list

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): Complete game state. Read-only. 
                Defined in "catanatron/catanatron_core/catanatron/game.py"
            playable_actions (Iterable[Action]): Options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        if not playable_actions:
            return None

        def score_action(action):
            """
            Score an action based on its potential value to the player.

            Settlement/road building actions will focus on resource diversity,
            hex probabilities, expansion potential, and blocking opponents.

            Args:
                action (Action): A single playable action to evaluate.
            Returns:
                score (float): The heuristic score for the action.
            """
            score = 0
            valid_action_types = {
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_ROAD,
                ActionType.END_TURN
            }

            # Determine the action type using hasattr or other known attributes
            action_type = getattr(action, "action_type", None)

            if action_type not in valid_action_types:
                print(f"Unhandled or invalid action type: {action_type}")
                return -float('inf')  # Assign a default score to skip invalid actions

            if action_type == ActionType.BUILD_SETTLEMENT:  # Check if it's a settlement action
                position = getattr(action, "position", None)
                if position is not None:
                    adjacent_resources = get_adjacent_resources(game, position)
                    probabilities = get_hex_probabilities(game, position)

                    # Reward diversity and high probabilities
                    score += len(set(adjacent_resources))  # Resource diversity
                    score += sum(probabilities)           # Hex probabilities

            elif action_type == ActionType.BUILD_ROAD:  # Check if it's a road action
                position = getattr(action, "position", None)
                if position is not None:
                    if leads_to_new_settlement(game, position):
                        score += 5
                    if blocks_opponent_expansion(game, position):
                        score += 3

            elif action_type == ActionType.END_TURN:  # Check if it's an end turn action
                score += 1  # Encourage ending turn if no better actions are available

            return score

        def get_adjacent_resources(game, position):
            """Get resource types for hexes adjacent to a position."""
            return [hex.resource for hex in game.board.get_adjacent_hexes(position)]

        def get_hex_probabilities(game, position):
            """Get probabilities for dice rolls on hexes adjacent to a position."""
            return [hex.probability for hex in game.board.get_adjacent_hexes(position)]

        def leads_to_new_settlement(game, position):
            """Check if a road position can lead to a new settlement."""
            return position in game.board.available_settlement_positions(self.color)

        def blocks_opponent_expansion(game, position):
            """Check if a road position blocks an opponent's expansion."""
            for opp_color in game.players.keys():
                if opp_color != self.color:
                    if position in game.board.available_settlement_positions(opp_color):
                        return True
            return False

        # Evaluate actions
        scored_actions = [(action, score_action(action)) for action in playable_actions]

        # Sort actions by score
        scored_actions.sort(key=lambda x: x[1], reverse=True)

        # Debugging log: Print the top scored actions
        for action, score in scored_actions[:5]:
            print(f"Action: {action}, Score: {score}")

        # Choose the best action based on score
        best_action = scored_actions[0][0]

        # Use the LLM for additional insights if needed
        game_state_summary = self.summarize_game_state(game, playable_actions)
        llm_prompt = f"Given the game state: {game_state_summary}, and available actions: {playable_actions}, suggest the best action."
        llm_suggestion = self.llm.query_llm(llm_prompt)
        print(f"LLM Suggestion: {llm_suggestion}")

        # Optionally use or override based on LLM suggestion here

        return best_action

    def summarize_game_state(self, game, playable_actions):
        """Create a summary of the current game state for LLM analysis."""
        return {
            "resources": self.resources,
            "settlements": len(self.settlements),
            "roads": len(self.roads),
            "available_actions": len(playable_actions),
        }