import os
from catanatron import Player
from datetime import datetime
from agents.fromScratchLLMStructured_player.llm_tools import LLM

class FooPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name)

    def evaluate_action(self, game, action):
        """Assigns a heuristic score to an action based on its potential impact.

        Args:
            game (Game): The current game state.
            action (Action): The action to evaluate.

        Returns:
            int: Heuristic score for the action.
        """
        score = 0

        # Determine the type of the action correctly
        action_type = action.action_type  # Accessing the "action_type" attribute directly.

        if action_type == 'BUILD_SETTLEMENT':
            score += 5  # Settlements are critical for resource production and VPs.
        elif action_type == 'BUILD_CITY':
            score += 8  # Cities provide more resources and significant VPs.
        elif action_type == 'BUILD_ROAD':
            score += 3  # Roads help expand and secure longest road.
        elif action_type == 'BUY_DEVELOPMENT_CARD':
            score += 2  # Development cards can provide flexibility and VPs.

        # Additional considerations:
        # - Blocking opponents' expansion
        # - Securing access to key resources
        # - Maximizing resource diversity

        # Example: Evaluate if the action disrupts an opponent
        if hasattr(action, 'blocks_opponent') and callable(action.blocks_opponent):
            try:
                if action.blocks_opponent():
                    score += 4
            except Exception as e:
                print(f"Error checking blocks_opponent for action {action}: {e}")

        return score

    def decide(self, game, playable_actions):
        """Chooses the best action based on heuristic evaluation.

        Args:
            game (Game): Complete game state (read-only).
            playable_actions (Iterable[Action]): Options to choose from.

        Returns:
            Action: Chosen element of playable_actions.
        """
        if not playable_actions:
            print("No playable actions available.")
            return None

        # Evaluate all actions and select the one with the highest score
        scored_actions = []
        for action in playable_actions:
            try:
                score = self.evaluate_action(game, action)
                scored_actions.append((action, score))
                print(f"Action: {action}, Score: {score}")  # Debugging log
            except Exception as e:
                print(f"Error evaluating action {action}: {e}")

        # Sort actions by score (descending) and select the best one
        if scored_actions:
            scored_actions.sort(key=lambda x: x[1], reverse=True)
            best_action, best_score = scored_actions[0]
            print(f"Chosen Action: {best_action} with Score: {best_score}")
            return best_action

        print("No valid actions were scored.")
        # Fallback: Select the first playable action as a safe default
        fallback_action = playable_actions[0]
        print(f"Fallback Action: {fallback_action} selected.")
        return fallback_action