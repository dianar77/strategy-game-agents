# FromScratchLLMStructured Player v6 - ADK Version

This is the **Google ADK-powered** version of the multi-agent system for evolving Catan game players. This version has been completely migrated from LangChain to Google's Agent Development Kit (ADK) for better performance and integration.

## 🚀 Key Features

- **Google ADK Integration**: Uses Google's Agent Development Kit for multi-agent coordination
- **Google Gemini Models**: Powered by Gemini 2.0 Flash for consistent performance
- **Multi-Agent Architecture**: 5 specialized agents working together
- **Autonomous Evolution**: Runs 20 evolution cycles automatically
- **Comprehensive Logging**: Detailed performance tracking and analysis

## 📁 Files Overview

### Core System Files
- **`creator_agent.py`** - Main multi-agent evolution system (951 lines)
- **`main_adk.py`** - Entry point for running the full system
- **`adk_agent.py`** - Simple interactive single-agent version for testing

### Player and LLM Files
- **`foo_player.py`** - The player that gets evolved by the agents
- **`foo_player_adk.py`** - Full-featured example player implementation
- **`llm_tools_adk.py`** - Google Gemini interface for game players
- **`__TEMPLATE__foo_player.py`** - Template for starting new players

### Configuration
- **`requirements_adk.txt`** - Python dependencies for ADK system
- **`__init__.py`** - Module initialization
- **`README_ADK.md`** - Detailed technical documentation

## 🏗️ Architecture

```
┌─────────────────────┐    coordinates    ┌─────────────────────┐
│   Meta Supervisor   │ ──────────────────►│ Specialist Agents   │
│  (Decides Strategy) │                    │ (Execute Tasks)     │
└─────────────────────┘                    └─────────────────────┘
         │                                           │
         │                                           │
         ▼                                           ▼
┌─────────────────────┐    evolves        ┌─────────────────────┐
│   Evolution Cycle   │ ──────────────────►│   foo_player.py     │
│   (20 iterations)   │                    │ (Game Player)       │
└─────────────────────┘                    └─────────────────────┘
```

### The 5 Specialized Agents

1. **Meta Supervisor** - Coordinates overall strategy and decides which agent to use
2. **Analyzer Agent** - Analyzes game results and performance data
3. **Strategizer Agent** - Develops new game strategies and approaches  
4. **Researcher Agent** - Researches Catan mechanics and implementation details
5. **Coder Agent** - Implements code changes to the player

## ⚙️ Setup

### 1. Install Dependencies

```bash
cd agents/fromScratchLLMStructured_player_v6
pip install -r requirements_adk.txt
```

### 2. Get Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Set environment variable:

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

### 3. Configure Catanatron Player Name

The system is configured to use player name `FOO_LLM_V6`. Make sure your Catanatron setup recognizes this player name, or update the command in `creator_agent.py`:

```python
FOO_RUN_COMMAND = "catanatron-play --players=AB,FOO_LLM_V6  --num=3  --config-vps-to-win=10"
```

## 🚀 Usage

### Quick Start - Interactive Agent

For testing and experimentation:

```bash
python adk_agent.py
```

This starts an interactive session where you can:
- Analyze the current player
- Get improvement suggestions  
- Write new code
- Run test games

### Full Evolution System

For autonomous evolution over 20 cycles:

```bash
python main_adk.py
```

This will:
1. Start with the current `foo_player.py`
2. Run 20 evolution cycles
3. Save results in `runs_adk/`
4. Create a final evolved player

### Programmatic Usage

```python
from agents.fromScratchLLMStructured_player_v6 import CreatorAgentADK

# Create and run the evolution system
creator = CreatorAgentADK()
creator.run_evolution_cycle()
```

## 📊 Output and Results

Results are saved in `runs_adk/adk_creator_YYYYMMDD_HHMMSS/`:

- **`performance_history.json`** - Evolution metrics and statistics
- **`evolution_state.json`** - Current state snapshots
- **`game_YYYYMMDD_HHMMSS_fg/`** - Individual game test results
- **`final_adk_YYYYMMDD_HHMMSS_foo_player.py`** - Final evolved player

## 🔄 Migration from v5_M

This v6 version **replaces** the ADK files from v5_M:

| v5_M File | v6 Equivalent | Status |
|-----------|---------------|---------|
| `creator_agent.py` | `creator_agent.py` | ✅ Migrated |
| `llm_tools.py` | `llm_tools_adk.py` | ✅ Migrated |
| `foo_player.py` | `foo_player.py` | ✅ Updated |

### Key Changes from v5_M:
- ✅ **No LangChain dependencies** - Pure Google ADK
- ✅ **Consistent model usage** - All agents use Gemini 2.0 Flash
- ✅ **Updated player name** - `FOO_LLM_V6` instead of `FOO_LLM_S5_M`
- ✅ **Better error handling** - Improved retry logic and logging
- ✅ **Simplified setup** - Only Google API key required

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install -r requirements_adk.txt
```

**2. Google API Key Issues**
```bash
# Check if key is set
echo $GOOGLE_API_KEY

# Set the key
export GOOGLE_API_KEY='your-api-key-here'
```

**3. Player Name Not Recognized**
Update the command in `creator_agent.py` to match your Catanatron setup.

**4. Permission/Path Issues**
Make sure you're running from the correct directory and have write permissions for creating `runs_adk/`.

### Debug Mode

Enable debug mode in any player file:
```python
self.debug = True  # In FooPlayer.__init__()
```

## 🔮 What's New in v6

- **Google ADK Native**: Built specifically for ADK, not a port
- **Gemini 2.0 Flash**: Latest Google model for all agents
- **Better Coordination**: Improved multi-agent communication
- **Enhanced Logging**: More detailed evolution tracking
- **Simpler Dependencies**: No complex LangChain setup needed

## 📈 Performance Expectations

- **Faster startup** - No LangChain initialization overhead
- **Better model consistency** - All agents use same Gemini model
- **Improved coordination** - ADK's native multi-agent features
- **Enhanced debugging** - Better error messages and logging

## 🤝 Contributing

When making changes:
1. Update both the working player (`foo_player.py`) and template
2. Test with both interactive (`adk_agent.py`) and full system (`main_adk.py`)
3. Update version info in `__init__.py`
4. Add entries to performance tracking

## 📄 License

Same license as the main project. See project root for details.

---

**Ready to evolve some Catan players? 🎲**

```bash
python main_adk.py
``` 