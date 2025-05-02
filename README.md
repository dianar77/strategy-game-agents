## setup

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt
cd catanatron && uv pip install -r requirements.txt
cd catanatron/catanatron_core && uv pip install -e .
cd ../catanatron_experimental && uv pip install -e .
cd ../catanatron_gym && uv pip install -e .
cd ../.. && uv pip install -e .
python testing.py
```

# strategy-game-agents

NEW WAY: Add to cli_players.py your agent (make sure to include **init**.py in directory)
`catanatron-play --players=LLM,R --num=1 --output=data/ --json`

or

`AZURE_OPENAI_API_KEY=azure_open_ai_key catanatron-play --players=LLM,R --num=1 --output=data/ --json`

DEPRECIATED Example Use Case for Code
catanatron-play --code=agents/vanillaLLM_player/vanillaLLM_player.py --players=AB,VanillaLLM --num=1 --output=data/ --json

How To View Commands
catanatron-play --help

How To View Players
catanatron-play --help-players

Current Players.

| Code | Player                  | Description                                                                                                                                                           |
| ---- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R    | RandomPlayer            | Chooses actions at random.                                                                                                                                            |
| W    | WeightedRandomPlayer    | Like RandomPlayer, but favors buying cities, settlements, and dev cards when possible.                                                                                |
| VP   | VictoryPointPlayer      | Chooses randomly from actions that increase victory points immediately if possible, else at random.                                                                   |
| G    | GreedyPlayoutsPlayer    | For each action, will play N random 'playouts'. Takes the action that led to best winning percent. First param is NUM_PLAYOUTS                                        |
| M    | MCTSPlayer              | Decides according to the MCTS algorithm. First param is NUM_SIMULATIONS.                                                                                              |
| F    | ValueFunctionPlayer     | Chooses the action that leads to the most immediate reward, based on a hand-crafted value function.                                                                   |
| AB   | AlphaBetaPlayer         | Implements alpha-beta algorithm. That is, looks ahead a couple levels deep evaluating leafs with hand-crafted value function. Params are DEPTH, PRUNNING              |
| SAB  | SameTurnAlphaBetaPlayer | AlphaBeta but searches only within turn                                                                                                                               |
| VLLM | VanillaLLMPlayer        | Initial Vanilla LLM Player with no additions                                                                                                                          |
| LLM  | LLMPlayer               | LLM with adjusted prompt and code to fix bugs with vanilla llm                                                                                                        |
| BL   | BasicLangPlayer         | First iteration of migration to using Lang Chain instead of custom base_llm class. Has inneficient memory that is saves number of messages which is set to a variable |
| TC   | ToolCallLLMPlayer       | LLM Player with tool call capabilities. Uses LangChain to call tools. Has access to web search, ...                                                                   |
| FOO  | FooPlayer               | Player being created by creator agent                                                                                                                                 |
