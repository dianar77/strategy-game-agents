import math
import random
import time
from typing import Iterable, List, Tuple

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player

########################################################################
# HELPER FUNCTIONS
########################################################################

def _get_vp(game: Game, color) -> int:
    idx = game.state.color_to_index[color]
    key = f"P{idx}_VICTORY_POINTS"
    return game.state.player_state[key]

########################################################################
# A Monte Carlo Tree Search approach with more simulations
########################################################################

MAX_SIMULATIONS = 300  # increase number of simulations to aim for better performance
MAX_TIME_SECS = 10.0   # limit time to 10 seconds
UCB_CONST = 1.4142

class MCTSNode:
    def __init__(self, game: Game, parent=None):
        self.game = game
        self.parent = parent
        self.children = dict()  # action -> list of (childNode, prob)
        self.visits = 0
        self.wins = 0.0
        self.terminal = (game.winning_color() is not None)

    def is_leaf(self):
        return len(self.children) == 0

class FooPlayer(Player):
    """A MCTS-based Catan player that tries to compete strongly"""

    def decide(self, game: Game, playable_actions: Iterable[Action]) -> Action:
        actions_list = list(playable_actions)
        if len(actions_list) == 1:
            return actions_list[0]

        start_time = time.time()
        root = MCTSNode(game.copy(), None)

        # Expand root
        self._expand(root)

        sim_count = 0
        while sim_count < MAX_SIMULATIONS and (time.time() - start_time) < MAX_TIME_SECS:
            leaf = self._select(root)
            result_color = self._rollout(leaf.game)
            self._backpropagate(leaf, result_color)
            sim_count += 1

        # choose best action by visits
        best_child = None
        best_visits = -1
        best_action = None
        for action, outcomes in root.children.items():
            # average visits among children states
            sum_visits = sum(childNode.visits for (childNode, _) in outcomes)
            if sum_visits > best_visits:
                best_visits = sum_visits
                best_action = action
                best_child = outcomes

        if best_action is None:
            return random.choice(actions_list)
        return best_action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node by the UCB1 formula recursively."""
        current = node
        while not current.is_leaf() and not current.terminal:
            current = self._ucb_select_child(current)
            # possibly expand if leaf
            if current.is_leaf() and not current.terminal:
                self._expand(current)
        return current

    def _expand(self, node: MCTSNode) -> None:
        if node.terminal:
            return
        actions = list(node.game.state.playable_actions)
        if not actions:
            return
        for action in actions:
            outcomes = self._simulate_action(node.game, action)
            node.children[action] = []
            for (child_game, prob) in outcomes:
                child_node = MCTSNode(child_game, node)
                node.children[action].append((child_node, prob))

    def _rollout(self, rollout_game: Game):
        """Run a random playout from the given state to the end, returns the winner color."""
        while rollout_game.winning_color() is None:
            actions = list(rollout_game.state.playable_actions)
            if not actions:
                break
            action = random.choice(actions)
            rollout_game.apply_action(action)
        return rollout_game.winning_color()

    def _backpropagate(self, node: MCTSNode, winner_color) -> None:
        current = node
        while current is not None:
            current.visits += 1
            if winner_color == self.color:
                current.wins += 1
            current = current.parent

    def _ucb_select_child(self, node: MCTSNode) -> MCTSNode:
        best_score = -999.0
        best_child = None
        # For each action, we might have multiple children states with probabilities
        # We'll pick one child state stochastically.
        # But we first pick the action via sum of UCB.
        best_action, best_action_score = None, -999.0

        for action, child_list in node.children.items():
            # We'll compute the average child UCB to pick an action.
            # Then pick a child from that action randomly, weighted by proba.
            # This is a heuristic.
            sum_scores = 0.0
            for (childNode, prob) in child_list:
                sum_scores += prob * self._ucb_score(childNode, node.visits)
            if sum_scores > best_action_score:
                best_action_score = sum_scores
                best_action = action
        # now pick a child from best_action.
        if best_action is None:
            # fallback
            return node
        child_list = node.children[best_action]
        # Weighted random selection by distribution of prob.
        # But we have a small # of child states typically, so let's do a standard approach.
        thresholds = []
        sum_prob = 0.0
        for (childNode, prob) in child_list:
            sum_prob += prob
            thresholds.append(sum_prob)
        r = random.random() * sum_prob
        idx = 0
        while idx < len(thresholds) and r > thresholds[idx]:
            idx += 1
        if idx == len(child_list):
            idx = len(child_list) - 1
        return child_list[idx][0]

    def _ucb_score(self, child: MCTSNode, parent_visits: int) -> float:
        if child.visits == 0:
            return 999999.0  # encourage exploration
        win_rate = child.wins / child.visits
        exploration = math.sqrt(math.log(parent_visits) / child.visits)
        return win_rate + UCB_CONST * exploration

    def _simulate_action(self, game: Game, action: Action) -> List[Tuple[Game, float]]:
        """Returns list of (newGame, probability). If there's dice rolling or similar, handle it."""
        try:
            from catanatron_experimental.machine_learning.players.tree_search_utils import execute_spectrum
            out = execute_spectrum(game, action)
            return [(gcopy, p) for (gcopy, p) in out]
        except Exception:
            # fallback single-outcome approach
            gcopy = game.copy()
            gcopy.apply_action(action)
            return [(gcopy, 1.0)]
