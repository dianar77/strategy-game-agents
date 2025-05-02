import time
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import base_llm
from typing import List
import random
from catanatron.models.player import Player
from catanatron.models.enums import Action, ActionType, RESOURCES

# Utility function to prioritize actions involving resources
def calculate_resource_priorities(state, playable_actions):
    """Identify high-priority actions based on resource diversity and probability."""
    priorities = {}

    for action in playable_actions:
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            # Example logic to prioritize settlement spots (assigning dummy values for now)
            priorities[action.value] = random.randint(1, 10)  # Replace with actual scoring

    return priorities

class PromptRefiningLLMPlayer(Player):
    def __init__(self, color, name=None):
        super().__init__(color, name)

    def decide(self, state, playable_actions: List[Action]) -> Action:
        # Calculate priorities for actions
        action_priorities = calculate_resource_priorities(state, playable_actions)

        # Sort actions by priority value
        sorted_actions = sorted(playable_actions, key=lambda action: action_priorities.get(action.value, 0), reverse=True)

        # Select the highest-priority action
        best_action = sorted_actions[0] if sorted_actions else random.choice(playable_actions)

        return best_action