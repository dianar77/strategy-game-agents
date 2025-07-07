# Summary of Subagent Inputs

This document outlines what data or files are read and analyzed by each of the six subagents in the `fromScratchLLMStructured_player_v7` system, based on the information in `outputs.md`.

## 1. Analyzer Agent
- **Analyzes**: Game performance metrics.
- **Source**: The output of an internal `get_current_metrics()` function. The agent's own output structure changes depending on whether this data is available.

## 2. Coder Agent
- **Reads**: The agent's Python source code.
- **Source**: The `foo_player.py` file. The output of its analysis is conditional on successfully reading the file.

## 3. Evolver Agent
- **Analyzes**: Game performance data.
- **Source**: The output of `get_current_metrics()`. The values in its output JSON are populated based on this data.

## 4. Player Agent
- **Analyzes**: The output of a game execution.
- **Source**: It runs a test game using a `run_testfoo()` function and processes the resulting output stream to check for success, errors, or exceptions.

## 5. Researcher Agent
- **Reads/Analyzes**: Does not analyze dynamic data.
- **Source**: Provides hardcoded, fixed JSON information about Catanatron game mechanics and its API.

## 6. Strategizer Agent
- **Analyzes**: The effectiveness of the current game strategy.
- **Source**: It uses performance metrics from `get_current_metrics()`. Its output differs depending on whether the data retrieval is successful.

---

## Data Read by the Evolver Agent (Coordinator)

The Evolver Agent acts as the central coordinator and is the final agent in the loop to synthesize information before deciding on the next action. It reads the outputs from the following agents:

### 1. From the Analyzer Agent
The Evolver Agent consumes a detailed performance breakdown from the Analyzer. Key data points include:
- **`game_performance`**: `win_rate` (current vs. previous), `average_score`, `games_played`.
- **`strategic_metrics`**: `resource_efficiency`, `trading_frequency`, `building_strategy`.
- **`error_analysis`**: `syntax_errors`, `runtime_exceptions`, `invalid_moves`.
- **`opponent_analysis`**: Win/loss rates against specific opponent types.

### 2. From the Strategizer Agent
It reads an evaluation of the current strategy's effectiveness to guide improvements. Key data points include:
- **`strategy_evaluation`**: `overall_effectiveness` and `strategy_category`.
- **`current_strengths`**: The effectiveness of early, mid, and late game tactics.
- **`strategic_weaknesses`**: Specific, prioritized issues to be fixed.
- **`opponent_adaptation`**: How well the strategy adapts to different opponents.

### 3. From the Researcher Agent
It can read static, hardcoded information about game rules and APIs to provide context for its evolutionary decisions. Key data includes:
- **`catanatron_mechanics`**: Details on game phases, actions, and scoring.
- **`api_documentation`**: Information on available functions and data structures.

---

## Data Read by the Coder Agent

The Coder Agent consumes two main types of data:

1.  **Player's Source Code (`foo_player.py`):** This is the primary file it reads. It analyzes this code for quality, complexity, and compliance before making any modifications.
2.  **Modification Instructions:** While not a file, it consumes a set of instructions, presumably from the Evolver Agent. These instructions detail the strategic changes or bug fixes that need to be implemented in the source code.

---

## Data Read by the Player Agent

The Player Agent consumes the following to do its job:

1.  **Player's Source Code (`foo_player.py`):** It reads and executes the agent's code to actually play a game of Catanatron.
2.  **Game Configuration:** It takes parameters that set up the game, most notably the type of opponent to play against (e.g., `alpha_beta`, `random`). This is implied by the `game_type` field in its output. 