# 🎮 Strategy Game Agents - How It Works

This system creates and evolves AI agents that play the board game **Catan** using Large Language Models (LLMs). Here's everything you need to know:

## 🎯 What You're Working With

You're using the **`fromScratchLLMStructured_player_v5_M`** agent - the most advanced version that:
- Uses **Claude 3.7** as its brain
- Has a **multi-node architecture** for different thinking tasks
- **Self-evolves** by testing itself and improving its code
- Can beat most opponents when properly trained

## 🏗️ System Architecture

### Core Components

1. **`foo_player.py`** - The actual Catan player that gets evolved
   - Contains game decision logic
   - Uses LLM to make strategic decisions
   - Gets rewritten/improved by the creator agent

2. **`creator_agent.py`** - The AI that evolves foo_player.py
   - Multi-node architecture with specialized roles:
     - 🧠 **Meta Node**: High-level coordination  
     - 📊 **Analyzer Node**: Analyzes game performance
     - 🎯 **Strategizer Node**: Develops strategies
     - 🔍 **Researcher Node**: Gathers information
     - 💻 **Coder Node**: Implements improvements

3. **`main.py`** - Runs the creator agent evolution process
4. **`testing.py`** - Evaluates agent performance against opponents

## 🎲 How Catan AI Works

### Game Basics
- **Goal**: First to 10 victory points wins
- **Resources**: Wood, Brick, Sheep, Wheat, Ore
- **Buildings**: Roads (0 VP), Settlements (1 VP), Cities (2 VP)
- **Strategy**: Balance resource collection, building placement, and opponent blocking

### AI Decision Process
1. **Game State Analysis**: Read current resources, buildings, opponent status
2. **LLM Query**: Ask Claude "What should I do next?"
3. **Action Selection**: Parse LLM response and choose valid action
4. **Fallback**: If LLM fails, use simple heuristics

## 🔄 Evolution Process

### How the Creator Agent Works

1. **🎮 Test Current Agent**
   ```
   Run 3 games: foo_player vs AlphaBetaPlayer
   Record wins/losses and detailed game logs
   ```

2. **📊 Analyze Performance**
   ```
   - What went wrong?
   - Where did we lose resources?  
   - Did we build in good locations?
   - How did opponents beat us?
   ```

3. **🎯 Develop Strategy**
   ```
   - Research successful Catan strategies
   - Identify specific improvements needed
   - Plan code modifications
   ```

4. **💻 Implement Changes**  
   ```
   - Modify foo_player.py code
   - Improve decision logic
   - Better resource management
   - Smarter building placement
   ```

5. **🔄 Repeat Until Success**
   ```
   Keep evolving until agent wins consistently
   Target: 60%+ win rate vs AlphaBetaPlayer
   ```

## 🚀 How to Run the System

### Option 1: Quick Test (No Evolution)
```bash
python run_example.py
# Choose option 1: Quick Test Games
```
Tests current agent without changing it.

### Option 2: Evolution Mode (Recommended)  
```bash
python run_example.py  
# Choose option 2: Run Creator Agent
```
This runs the full evolution process:
- Takes 30-60 minutes
- Agent tests, analyzes, and improves itself
- You'll see lots of console output showing progress

### Option 3: Full Evaluation
```bash
python run_example.py
# Choose option 3: Full Evaluation  
```
Tests evolved agent against multiple opponents.

### Option 4: Direct Commands
```bash
# Evolution (main process)
python main.py

# Testing only
python testing.py

# Quick manual test
.venv\Scripts\catanatron-play.exe --players=AB,FOO_LLM_S5_M --num=3
```

## 📊 Understanding the Output

### During Evolution
```
🚀 STARTING GAME SESSION
   Player 1 (Testing): FooPlayer_LLM_Structured_V5_M (FOO_LLM_S5_M)  
   Player 2 (Opponent): AlphaBetaPlayer (AB)
   Number of games: 3
   Victory points needed: 10

📝 Command to execute: catanatron-play --players=AB,FOO_LLM_S5_M --num=3
⏱️  Starting game execution...
✅ Games completed in 45.3 seconds

🏆 FINAL SCORES:
   FooPlayer_LLM_Structured_V5_M: 1/3 wins (33.3%)
   AlphaBetaPlayer: 2/3 wins (66.7%)

😞 FooPlayer_LLM_Structured_V5_M loses to AlphaBetaPlayer (1/3)
   Need to improve the agent...
```

### Evolution Progress
```
🧠 Meta Node: Analyzing overall performance...
📊 Analyzer Node: Found issues with resource management
🎯 Strategizer Node: Developing new trading strategy  
💻 Coder Node: Implementing improved decision logic
✅ Code updated successfully - testing new version...
```

### Final Success
```
🎉 FooPlayer_LLM_Structured_V5_M WINS against AlphaBetaPlayer! (2/3)
✅ Agent evolution successful
📁 Results saved in: agents/fromScratchLLMStructured_player_v5_M/runs/creator_20250101_120000
```

## 🎯 Available Opponents (Difficulty)

From strongest to weakest:

1. **AB (AlphaBetaPlayer)** - 🔴 HARD
   - Uses minimax algorithm with alpha-beta pruning
   - Looks ahead several moves  
   - Your target to beat!

2. **F (ValueFunctionPlayer)** - 🟡 MEDIUM
   - Uses hand-crafted evaluation function
   - Makes smart immediate decisions

3. **M (MCTSPlayer)** - 🟡 MEDIUM  
   - Monte Carlo Tree Search
   - Simulates many possible futures

4. **W (WeightedRandomPlayer)** - 🟢 EASY
   - Random but prefers good moves
   - Good for initial testing

5. **R (RandomPlayer)** - 🟢 VERY EASY
   - Completely random moves
   - Easiest opponent

## 💡 Tips for Success

### Environment Setup
- Make sure you have AWS credentials set for Claude API
- Virtual environment should be activated  
- All dependencies installed

### Monitoring Progress
- Evolution takes time - be patient!
- Watch console output for progress
- Check `runs/` directory for detailed logs
- Each evolution cycle creates timestamped folders

### Performance Targets
- **30%+ win rate vs AB**: Good progress
- **50%+ win rate vs AB**: Very good  
- **60%+ win rate vs AB**: Excellent
- **70%+ win rate vs AB**: Outstanding

### Common Issues
- **LLM Timeouts**: Normal, system will retry
- **Low Win Rate**: Needs more evolution cycles
- **API Errors**: Check your AWS credentials
- **Long Runtime**: Evolution is slow by design

## 📁 File Structure

```
strategy-game-agents/
├── main.py                    # Run creator agent
├── testing.py                 # Evaluate agent performance  
├── run_example.py            # Interactive menu system
├── agents/
│   └── fromScratchLLMStructured_player_v5_M/
│       ├── creator_agent.py   # The evolution brain
│       ├── foo_player.py      # The player being evolved
│       ├── llm_tools.py       # LLM interface
│       └── runs/              # Evolution results
│           └── creator_YYYYMMDD_HHMMSS/
│               ├── performance_history.json
│               ├── llm_log_claude-3.7.txt  
│               └── game_YYYYMMDD_HHMMSS_fg/
└── results/                   # Evaluation results
    └── trial_YYYYMMDD_HHMMSS/
        ├── summary.txt
        └── claude-3.7_FOO_LLM_S5_M_vs_AB.txt
```

## 🎮 What Happens Next

1. **Start with**: `python run_example.py`
2. **Choose option 2**: Run Creator Agent
3. **Wait 30-60 minutes**: Let it evolve
4. **Check results**: Look for improved win rates
5. **Test further**: Run evaluations vs other opponents
6. **Iterate**: Re-run creator if needed

The system will automatically improve your Catan AI through iterative self-play and analysis. Each evolution cycle makes it smarter at resource management, building placement, and strategic decision-making!

## 🔧 Advanced Usage

### Custom Opponents
Edit `testing.py` to change the opponent list:
```python
OPPONENTS = ["AB", "F", "M"]  # Test vs multiple opponents
```

### Faster Testing  
Reduce games per test:
```python
NUM_GAMES = 1  # Quick testing
```

### Different LLMs
Modify `creator_agent.py` to use different models:
```python
# Switch to GPT-4o instead of Claude
self.llm_name = "gpt-4o"
```

The system is designed to be flexible and self-improving. The more you run it, the better your Catan AI becomes! 🚀 