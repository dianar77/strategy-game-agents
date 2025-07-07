import os
from catanatron import Player
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.actions import ActionType

# Add debugging
print("🔍 FooPlayer: Starting imports...")

try:
    from agents.fromScratchLLMStructured_player_v7.llm_tools import LLM
    print("✅ FooPlayer: Successfully imported LLM from v7")
except Exception as e:
    print(f"❌ FooPlayer: Failed to import LLM from v7: {e}")
    # Fallback to prevent crashes
    class LLM:
        def __init__(self):
            print("⚠️ Using fallback LLM")
        def query_llm(self, prompt):
            return "fallback response"

class FooPlayer(Player):
    def __init__(self, name=None):
        print("🔍 FooPlayer: Initializing...")
        super().__init__(Color.BLUE, name)
        
        try:
            self.llm = LLM() # use self.llm.query_llm(str prompt) to query the LLM
            print("✅ FooPlayer: LLM initialized successfully")
        except Exception as e:
            print(f"❌ FooPlayer: LLM initialization failed: {e}")
            self.llm = None
        
        print(f"✅ FooPlayer: Initialization complete. Name: {name}, Color: {self.color}")

    def decide(self, game, playable_actions):
        # Should return one of the playable_actions.

        # Args:
        #     game (Game): complete game state. read-only. 
        #         Defined in in "catanatron/catanatron_core/catanatron/game.py"
        #     playable_actions (Iterable[Action]): options to choose from
        # Return:
        #     action (Action): Chosen element of playable_actions
        
        print(f"🎮 FooPlayer: Making decision. Available actions: {len(playable_actions)}")
        
        try:
            # ===== YOUR CODE HERE =====
            # As an example we simply return the first action:
            action = playable_actions[0]
            print(f"✅ FooPlayer: Choosing action: {action.action_type}")
            return action
            # ===== END YOUR CODE =====
        except Exception as e:
            print(f"❌ FooPlayer: Error in decide: {e}")
            print(f"   Playable actions: {playable_actions}")
            # Return first action as fallback
            if playable_actions:
                return playable_actions[0]
            else:
                raise Exception("No playable actions available")

