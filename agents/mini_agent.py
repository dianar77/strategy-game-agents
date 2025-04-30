from agents.base_llm import BaseLLM, OpenAILLM
from typing import Dict, Any

class MiniAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def act(self, game_state: Dict[str, Any]) -> str:
        prompt = f"You are an agent playing a strategy game. Given the following game state, decide your next move:\n\n{game_state}\n\nRespond with your action."
        response = self.llm.query(prompt)
        if isinstance(response, dict) and "response" in response:
            return response["response"]
        return str(response)
