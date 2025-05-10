from collections import namedtuple

from rich.table import Table

from catanatron.models.player import RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

# from catanatron_experimental.mcts_score_collector import (
#     MCTSScoreCollector,
#     MCTSPredictor,
# )
# from catanatron_experimental.machine_learning.players.reinforcement import (
#     QRLPlayer,
#     TensorRLPlayer,
#     VRLPlayer,
#     PRLPlayer,
# )
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    SameTurnAlphaBetaPlayer,
)
from catanatron.players.search import VictoryPointPlayer
from catanatron_experimental.machine_learning.players.mcts import MCTSPlayer
from catanatron_experimental.machine_learning.players.playouts import (
    GreedyPlayoutsPlayer,
)

from agents.llm_player.llm_player import LLMPlayer
from agents.vanillaLLM_player.vanillaLLM_player import VanillaLLMPlayer
from agents.basicLang_player.basicLang_player import BasicLangPlayer
from agents.toolCallLLM_player.toolCallLLM_player import ToolCallLLMPlayer
from agents.promptRefiningLLM_player.promptRefiningLLM_player import PromptRefiningLLMPlayer
from agents.codeRefiningLLM_player.codeRefiningLLM_player import CodeRefiningLLMPlayer

from agents.fromScratch_player.foo_player import FooPlayer as FooScratchPlayer
from agents.fromScratchLLM_player.foo_player import FooPlayer as FooLLMPlayer
from agents.fromScratchLLM_player_v2.foo_player import FooPlayer as FooLLMPlayerV2

from agents.fromScratchLLM_player_v2.runs.creator_20250508_112135_hitl.foo_player import FooPlayer as FooLLMPlayerV2_1
# from catanatron_experimental.machine_learning.players.online_mcts_dqn import (
#     OnlineMCTSDQNPlayer,
# )

# PLAYER_CLASSES = {
#     "O": OnlineMCTSDQNPlayer,
#     "S": ScikitPlayer,
#     "Y": MyPlayer,
#     # Used like: --players=V:path/to/model.model,T:path/to.model
#     "C": ForcePlayer,
#     "VRL": VRLPlayer,
#     "Q": QRLPlayer,
#     "P": PRLPlayer,
#     "T": TensorRLPlayer,
#     "D": DQNPlayer,
#     "CO": MCTSScoreCollector,
#     "COP": MCTSPredictor,
# }

# Player must have a CODE, NAME, DESCRIPTION, CLASS.
CliPlayer = namedtuple("CliPlayer", ["code", "name", "description", "import_fn"])
CLI_PLAYERS = [
    CliPlayer("R", "RandomPlayer", "Chooses actions at random.", RandomPlayer),
    CliPlayer(
        "W",
        "WeightedRandomPlayer",
        "Like RandomPlayer, but favors buying cities, settlements, and dev cards when possible.",
        WeightedRandomPlayer,
    ),
    CliPlayer(
        "VP",
        "VictoryPointPlayer",
        "Chooses randomly from actions that increase victory points immediately if possible, else at random.",
        VictoryPointPlayer,
    ),
    CliPlayer(
        "G",
        "GreedyPlayoutsPlayer",
        "For each action, will play N random 'playouts'. "
        + "Takes the action that led to best winning percent. "
        + "First param is NUM_PLAYOUTS",
        GreedyPlayoutsPlayer,
    ),
    CliPlayer(
        "M",
        "MCTSPlayer",
        "Decides according to the MCTS algorithm. First param is NUM_SIMULATIONS.",
        MCTSPlayer,
    ),
    CliPlayer(
        "F",
        "ValueFunctionPlayer",
        "Chooses the action that leads to the most immediate reward, based on a hand-crafted value function.",
        ValueFunctionPlayer,
    ),
    CliPlayer(
        "AB",
        "AlphaBetaPlayer",
        "Implements alpha-beta algorithm. That is, looks ahead a couple "
        + "levels deep evaluating leafs with hand-crafted value function. "
        + "Params are DEPTH, PRUNNING",
        AlphaBetaPlayer,
    ),
    CliPlayer(
        "SAB",
        "SameTurnAlphaBetaPlayer",
        "AlphaBeta but searches only within turn",
        SameTurnAlphaBetaPlayer,
    ),
    CliPlayer(
        "VLLM",
        "VanillaLLMPlayer",
        "Initial Vanilla LLM Player with no additions",
        VanillaLLMPlayer,
    ),
    CliPlayer(
        "LLM",
        "LLMPlayer",
        "LLM with adjusted prompt and code to fix bugs with vanilla llm",
        LLMPlayer,
    ),
    CliPlayer(
        "BL",
        "BasicLangPlayer",
        "First iteration of migration to using Lang Chain instead of custom base_llm class. "
        + "Has inneficient memory that is saves number of messages which is set to a variable ",
        BasicLangPlayer,
    ),
    CliPlayer(
        "TC",
        "ToolCallLLMPlayer",
        "LLM Player with tool call capabilities. Uses LangChain to call tools."
        + " Has access to web search, ...",
        ToolCallLLMPlayer,
    ),
    CliPlayer(
        "FOO_S",
        "FooPlayer_Scratch",
        "Player being created by creator agent",
        FooScratchPlayer
    ),
    CliPlayer(
        "FOO_LLM",
        "FooPlayer_LLM",
        "Player being created by creator agent. FooPlayer has access to query the LLM",
        FooLLMPlayer
    ),
    CliPlayer(
        "FOO_LLM_V2",
        "FooPlayer_LLM_V2",
        "Player being created by creator agent that has more tools to edit FooPlayer's code. FooPlayer has access to query the LLM",
        FooLLMPlayerV2
    ),
    CliPlayer(
        "PR_LLM",
        "PromptRefiningLLMPlayer",
        "LLM Player That has had the prompt refined by the creator agent",
        PromptRefiningLLMPlayer,
    ),
    CliPlayer(
        "CR_LLM",
        "CodeRefiningLLMPlayer",
        "LLM Player That has had the code refined by the creator agent",
        CodeRefiningLLMPlayer,
    ),
    CliPlayer(
        "FOO_LLM_V2_1",
        "FooPlayer_LLM_V2_1",
        "Player being created by creator agent that has more tools to edit FooPlayer's code. FooPlayer has access to query the LLM",
        FooLLMPlayerV2_1
    ),

]


def register_player(code):
    def decorator(player_class):
        CLI_PLAYERS.append(
            CliPlayer(
                code,
                player_class.__name__,
                player_class.__doc__,
                player_class,
            ),
        )

    return decorator


CUSTOM_ACCUMULATORS = []


def register_accumulator(accumulator_class):
    CUSTOM_ACCUMULATORS.append(accumulator_class)


def player_help_table():
    table = Table(title="Player Legend")
    table.add_column("CODE", justify="center", style="cyan", no_wrap=True)
    table.add_column("PLAYER")
    table.add_column("DESCRIPTION")
    for player in CLI_PLAYERS:
        table.add_row(player.code, player.name, player.description)
    return table
