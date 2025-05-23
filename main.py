import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "catanatron"))
minimax_dir = os.path.join(os.path.dirname(__file__), "catanatron/catanatron_experimental/catanatron_experimental/machine_learning/players")
sys.path.append(minimax_dir)


from agents.base_llm import OpenAILLM, MistralLLM, AzureOpenAILLM, AnthropicLLM
from catanatron import Game, RandomPlayer, Color

from agents.promptEvolver.creator_agent import CreatorAgent as promptEvolver
from agents.agentEvolver.creator_agent import CreatorAgent as agentEvolver
from agents.llmAgentEvolver.creator_agent import CreatorAgent as llmAgentEvolver

from minimax import AlphaBetaPlayer
from catanatron_server.utils import open_link

def main():

    #Choose Your Desired Evolver (Comment out the others)

    #evolver = promptEvolver()
    #evolver = agentEvolver()
    evolver = llmAgentEvolver()

    # Run The Evolver
    evolver.run_react_graph()



if __name__ == "__main__":
    main()