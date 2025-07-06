"""
fromScratchLLMStructured_player_v6 - ADK Version

Google ADK-powered multi-agent system for evolving Catan game players.
This is the ADK version of the multi-agent evolution system.
"""

from .creator_agent import CreatorAgentADK
from .adk_agent import create_evolution_agent
from .llm_tools_adk import LLM, create_llm
from ..fromScratchLLMStructured_player_v7.foo_player import FooPlayer

__version__ = "6.0.0"
__author__ = "ADK Evolution Team"
__description__ = "ADK-powered multi-agent system for evolving Catan players"

__all__ = [
    "CreatorAgentADK",
    "create_evolution_agent", 
    "LLM",
    "create_llm",
    "FooPlayer"
] 