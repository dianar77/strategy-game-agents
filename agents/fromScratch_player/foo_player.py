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
        from random import choice

        # If we can build a settlement, prioritize it
        for action in playable_actions:
            if action.action_type == "BUILD_SETTLEMENT":
                return action

        # If we can build a city, prioritize it next
        for action in playable_actions:
            if action.action_type == "BUILD_CITY":
                return action

        # If we can buy a development card, prioritize it
        for action in playable_actions:
            if action.action_type == "BUY_DEVELOPMENT_CARD":
                return action

        # Use the robber strategically (e.g., target leader)
        for action in playable_actions:
            if action.action_type == "MOVE_ROBBER":
                leader = max(game.players, key=lambda p: p.victory_points)
                if action.details and leader.color in action.details:
                    return action

        # Prioritize longest road if advantageous
        for action in playable_actions:
            if action.action_type == "BUILD_ROAD":
                return action

        # Default to a random action if no preferences
        return choice(playable_actions)