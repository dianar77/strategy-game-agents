import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "catanatron"))
minimax_dir = os.path.join(os.path.dirname(__file__), "catanatron/catanatron_experimental/catanatron_experimental/machine_learning/players")
sys.path.append(minimax_dir)

from agents.base_llm import OpenAILLM, MistralLLM, AzureOpenAILLM
from catanatron import Game, RandomPlayer, Color
from agents.llm_player.llm_player import LLMPlayer  # Import your LLMPlayer
from agents.basicLang_player.basicLang_player import BasicLangPlayer
from agents.toolCallLLM_player.toolCallLLM_player import ToolCallLLMPlayer
from agents.fromScratch_player.creator_agent import read_foo, write_foo, run_testfoo, list_local_files, read_local_file 
from agents.fromScratch_player.creator_agent import CreatorAgent as ScratchCreatorAgent 
from agents.promptRefiningLLM_player.creator_agent import CreatorAgent as PromptRefiningCreatorAgent
from agents.promptRefiningLLM_player.creator_agent import read_foo, write_foo, run_testfoo, list_local_files, read_local_file 
from agents.codeRefiningLLM_player.creator_agent import CreatorAgent as CodeRefiningCreatorAgent
from agents.codeRefiningLLM_player.creator_agent import read_foo, write_foo, run_testfoo, list_local_files, read_local_file

from minimax import AlphaBetaPlayer
from catanatron_server.utils import open_link

def main():

    cA = CodeRefiningCreatorAgent()
    #print(write_foo("print('Hello, world!')"))  # Write to foo_player.py
    #print(read_foo())
    #print(run_testfoo())
    #print(list_local_files())
    cA.run_react_graph()

    # players = [
    #     RandomPlayer(Color.RED),
    #     #AlphaBetaPlayer(Color.BLUE),
    #     ToolCallLLMPlayer(Color.ORANGE),
    #     #BasicLangPlayer(Color.BLUE),
    #     #LLMPlayer(Color.WHITE, llm=MistralLLM(model_name="mistral-large-latest"))
    # ]
    # game = Game(players)
    # #open_link(game)  # opens game in browser...not working yet
    # print(game.play())

if __name__ == "__main__":
    main()