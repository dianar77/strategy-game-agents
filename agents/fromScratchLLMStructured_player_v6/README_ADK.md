# Catanatron Creator Agent - Google ADK Version

This is a conversion of the original LangGraph-based multi-agent system to use Google's Agent Development Kit (ADK). The ADK version provides the same functionality but uses Google's framework for building and orchestrating AI agents.

## What Changed

### From LangGraph to Google ADK

**Original (LangGraph)**:
- Used LangChain components and LangGraph for orchestration
- Complex state management with TypedDict classes
- Manual message routing and tool calling
- Required multiple LangChain dependencies

**New (Google ADK)**:
- Uses Google ADK's LlmAgent and Runner classes
- Built-in session management and state handling
- Simplified agent creation and tool integration
- Powered by Google Gemini models

### Key Benefits of ADK Version

1. **Simplified Architecture**: Less boilerplate code for agent creation
2. **Better Integration**: Native Google services integration
3. **Improved Performance**: Optimized for Gemini models
4. **Easier Deployment**: ADK provides built-in deployment options
5. **Better Debugging**: Built-in evaluation and monitoring tools

## Files

- `creator_agent.py` - Main multi-agent system using ADK
- `adk_agent.py` - Simple single-agent version for testing
- `main_adk.py` - Entry point for the full multi-agent system
- `requirements_adk.txt` - ADK-specific dependencies

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_adk.txt
```

### 2. Get Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Set it as an environment variable:

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

### 3. Run the Agent

#### Simple Single Agent (Recommended for Testing)

```bash
python adk_agent.py
```

This starts an interactive session where you can:
- Analyze the current foo_player.py
- Get improvement suggestions
- Write new code
- Run test games

#### Full Multi-Agent System

```bash
python main_adk.py
```

This runs the complete evolution cycle with multiple specialized agents:
- **Meta Agent**: Coordinates the overall strategy
- **Analyzer Agent**: Analyzes game results and performance
- **Strategizer Agent**: Develops new game strategies
- **Researcher Agent**: Researches game mechanics and APIs
- **Coder Agent**: Implements code changes

## Architecture Comparison

### Original LangGraph System

```python
class CreatorGraphState(TypedDict):
    meta_messages: list[AnyMessage]
    analyzer_messages: list[AnyMessage]
    # ... complex state management

def tool_calling_state_graph(sys_msg, msgs, tools):
    # Complex graph construction
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    # ... manual routing logic
```

### New ADK System

```python
# Simple agent creation
analyzer_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="analyzer",
    description="Analyzes game performance",
    instruction="Your role instructions...",
    tools=[read_local_file, view_last_game_llm_query]
)

# Automatic session management
runner = Runner(
    agent=analyzer_agent,
    app_name=app_name,
    session_service=session_service
)
```

## Agent Roles

### Meta Supervisor
- Coordinates the multi-agent system
- Determines which specialist agent to consult
- Sets high-level and low-level goals

### Analyzer Agent
- Analyzes game outputs and performance history
- Identifies syntax errors and implementation problems
- Provides detailed performance reports

### Strategizer Agent
- Develops new game strategies
- Analyzes effectiveness of previous approaches
- Searches for innovative strategic ideas

### Researcher Agent
- Researches Catanatron game mechanics
- Finds implementation details and API documentation
- Provides code examples and syntax information

### Coder Agent
- Implements code changes to foo_player.py
- Fixes bugs and syntax errors
- Writes clean, well-commented Python code

## Tools Available

Each agent has access to relevant tools:

- `read_foo()` - Read current player code
- `write_foo()` - Update player code
- `run_testfoo()` - Run game tests
- `read_local_file()` - Access game files
- `web_search()` - Search for information
- `view_last_game_llm_query()` - View game results
- `list_catanatron_files()` - List available game files

## Output and Logging

The ADK version saves results in `runs_adk/` directory:
- Evolution state and history
- Game test outputs
- Performance metrics
- Final evolved player code

## Migration Notes

If you're migrating from the LangGraph version:

1. **Environment**: Replace LangChain API keys with Google API key
2. **Dependencies**: Install ADK requirements instead of LangChain
3. **Execution**: Use the new entry points (main_adk.py or adk_agent.py)
4. **Output**: Check `runs_adk/` instead of `runs/` for results

## Troubleshooting

### API Key Issues
- Ensure `GOOGLE_API_KEY` is set correctly
- Verify the key has appropriate permissions
- Check quota limits in Google AI Studio

### Import Errors
- Install all dependencies: `pip install -r requirements_adk.txt`
- Ensure you're using Python 3.9+
- Check that google-adk is properly installed

### Game Execution Issues
- Verify Catanatron is properly installed
- Check that the game command path is correct
- Ensure foo_player.py template exists

## Future Enhancements

The ADK version opens up new possibilities:
- **Vertex AI Integration**: Deploy to Google Cloud
- **Advanced Evaluation**: Use ADK's built-in evaluation tools
- **Multi-modal Agents**: Add vision capabilities for game analysis
- **Streaming Responses**: Real-time agent interactions
- **Agent Teams**: More sophisticated multi-agent coordination

## Support

For issues specific to the ADK conversion, check:
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Google AI Studio](https://aistudio.google.com/)
- Original project documentation for game-specific issues 