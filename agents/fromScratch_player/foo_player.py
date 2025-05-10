from catanatron import Player

class FooPlayer(Player):
    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # Strategic prioritization of actions
        for action in playable_actions:
            # Always try to build a settlement if possible
            if action.action_type.name == "BUILD_SETTLEMENT":
                return action
            # Try to build a city next
            elif action.action_type.name == "BUILD_CITY":
                return action
            # Build a road if settlements or cities are not possible
            elif action.action_type.name == "BUILD_ROAD":
                return action
            # Buy a development card if building is not viable
            elif action.action_type.name == "BUY_DEVELOPMENT_CARD":
                return action
            # Play specific development cards
            elif action.action_type.name in ["PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY"]:
                return action
            # Trade resources if no other options are available
            elif action.action_type.name == "MARITIME_TRADE":
                return action
        
        # Roll dice if it's the player's turn and required
        for action in playable_actions:
            if action.action_type.name == "ROLL":
                return action
        
        # Default to ending the turn if no other actions are better
        return playable_actions[-1]