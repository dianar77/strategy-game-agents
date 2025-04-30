import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
from agents.base_llm import OpenAILLM, MistralLLM, AzureOpenAILLM


from catanatron import Game, RandomPlayer, Color
from agents.llm_player.llm_player import LLMPlayer  # Import your LLMPlayer

def main():
    # Create players: 3 random, 1 LLM agent
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        LLMPlayer(Color.ORANGE, llm=AzureOpenAILLM(model_name="gpt-4o")),
        LLMPlayer(Color.WHITE, llm=MistralLLM(model_name="mistral-large-latest"))
    ]
    game = Game(players)
    print(game.play())

if __name__ == "__main__":
    main()