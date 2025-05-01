# strategy-game-agents

NEW WAY: Add to cli_players.py your agent (make sure to include __init__.py in directory)
catanatron-play --players=LLM,VanillaLLM --num=1 --output=data/ --json


DEPRECIATED Example Use Case for Code
catanatron-play --code=agents/vanillaLLM_player/vanillaLLM_player.py  --players=AB,VanillaLLM --num=1 --output=data/ --json

How To View Commands
catanatron-play --help

How To View Players
catanatron-play --help-players


Current Players. 
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ CODE ┃ PLAYER                  ┃ DESCRIPTION                                                                                         ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│  R   │ RandomPlayer            │ Chooses actions at random.                                                                          │
│  W   │ WeightedRandomPlayer    │ Like RandomPlayer, but favors buying cities, settlements, and dev cards when possible.              │
│  VP  │ VictoryPointPlayer      │ Chooses randomly from actions that increase victory points immediately if possible, else at random. │
│  G   │ GreedyPlayoutsPlayer    │ For each action, will play N random 'playouts'. Takes the action that led to best winning percent.  │
│      │                         │ First param is NUM_PLAYOUTS                                                                         │
│  M   │ MCTSPlayer              │ Decides according to the MCTS algorithm. First param is NUM_SIMULATIONS.                            │
│  F   │ ValueFunctionPlayer     │ Chooses the action that leads to the most immediate reward, based on a hand-crafted value function. │
│  AB  │ AlphaBetaPlayer         │ Implements alpha-beta algorithm. That is, looks ahead a couple levels deep evaluating leafs with    │
│      │                         │ hand-crafted value function. Params are DEPTH, PRUNNING                                             │
│ SAB  │ SameTurnAlphaBetaPlayer │ AlphaBeta but searches only within turn                                                             │
│ VLLM │ VanillaLLMPlayer        │ Initial Vanilla LLM Player with no additions                                                        │
│ LLM  │ LLMPlayer               │ LLM with adjusted prompt and code to fix bugs with vanilla llm                                      │
│  BL  │ BasicLangPlayer         │ First iteration of migration to using Lang Chain instead of custom base_llm class      
