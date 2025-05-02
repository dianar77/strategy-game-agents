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
        # ===== YOUR CODE HERE =====
        # Expert Strategy:
        # 1. Prioritize "Ore/Wheat" for early cities and dev cards.
        # 2. Expand roads to block opponents and reach high-value spots.
        # 3. Trade effectively to achieve step 1 and 2.
        # 4. Conquer Longest Road and Largest Army midgame.

        def evaluate_action(action):
            # Assign weight based on optimal strategy goals
            action_str = str(action)
            if 'build_city' in action_str:
                return 100  # Cities boost resource production
            if 'build_settlement' in action_str:
                return 90  # Settlements secure presence
            if 'build_road' in action_str:
                return 70  # Roads enable expansion
            if 'trade' in action_str:
                return 50  # Trades unblock resource constraints
            if 'play_dev_card' in action_str:
                return 40  # Development cards give bonuses
            if 'block_opponent' in action_str:
                return 30  # Disrupt rivals
            return 10  # Lower priority actions

        # Select action with the highest score
        best_action = max(playable_actions, key=evaluate_action)
        return best_action
        # ===== END YOUR CODE =====