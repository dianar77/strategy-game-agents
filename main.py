import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "catanatron"))
minimax_dir = os.path.join(os.path.dirname(__file__), "catanatron/catanatron_experimental/catanatron_experimental/machine_learning/players")
sys.path.append(minimax_dir)


from agents.base_llm import OpenAILLM, MistralLLM, AzureOpenAILLM, AnthropicLLM
from catanatron import Game, RandomPlayer, Color

from agents.fromScratchLLM_player_v2.creator_agent import CreatorAgent as ScratchCreatorLLMAgentV2


from minimax import AlphaBetaPlayer
from catanatron_server.utils import open_link

def main():



    cA = ScratchCreatorLLMAgentV2()
    cA.run_react_graph()



if __name__ == "__main__":
    main()