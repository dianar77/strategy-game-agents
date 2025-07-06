"""
fromScratchLLMStructured_player_v6 - ADK Version

Google ADK-powered multi-agent system for evolving Catan game players.
This is the ADK version of the multi-agent evolution system.
"""

# Import the main player first (this should always work)
from .foo_player import FooPlayer

# Try to import ADK components (these may fail if Google ADK is not installed)
try:
    from .creator_agent import CreatorAgent
    ADK_AVAILABLE = True
except ImportError:
    CreatorAgent = None
    ADK_AVAILABLE = False

try:
    from .adk_agent import create_evolution_agent
except ImportError:
    create_evolution_agent = None

try:
    from .llm_tools_adk import LLM, create_llm
except ImportError:
    LLM = None
    create_llm = None

__version__ = "6.0.0"
__author__ = "ADK Evolution Team"
__description__ = "ADK-powered multi-agent system for evolving Catan players"

# Build __all__ list dynamically based on what's available
__all__ = ["FooPlayer", "ADK_AVAILABLE"]

if CreatorAgent is not None:
    __all__.append("CreatorAgent")
if create_evolution_agent is not None:
    __all__.append("create_evolution_agent")
if LLM is not None:
    __all__.extend(["LLM", "create_llm"]) 