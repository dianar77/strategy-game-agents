import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "catanatron"))
minimax_dir = os.path.join(os.path.dirname(__file__), "catanatron/catanatron_experimental/catanatron_experimental/machine_learning/players")
sys.path.append(minimax_dir)

print("="*80)
print("STRATEGY GAME AGENTS - AI CREATOR SYSTEM")
print("="*80)
print("🚀 Starting AI Creator Agent for Catan Strategy Game")
print("🎯 Target: fromScratchLLMStructured_player_v5_M (FOO_LLM_S5_M)")
print("🧠 This system uses LLM to iteratively improve game strategy")
print("="*80)

from agents.base_llm import OpenAILLM, MistralLLM, AzureOpenAILLM, AnthropicLLM
from catanatron import Game, RandomPlayer, Color
from agents.llm_player.llm_player import LLMPlayer  # Import your LLMPlayer
from agents.vanillaLLM_player.vanillaLLM_player import VanillaLLMPlayer
from agents.basicLang_player.basicLang_player import BasicLangPlayer
from agents.toolCallLLM_player.toolCallLLM_player import ToolCallLLMPlayer

print("✅ Core imports loaded successfully")

# Import creator agents
from agents.fromScratch_player.creator_agent import read_foo, write_foo, run_testfoo, list_local_files, read_local_file 
from agents.fromScratch_player.creator_agent import CreatorAgent as ScratchCreatorAgent 
from agents.promptRefiningLLM_player.creator_agent import CreatorAgent as PromptRefiningCreatorAgent
from agents.promptRefiningLLM_player.creator_agent import read_foo, write_foo, list_local_files, read_local_file 
from agents.codeRefiningLLM_player.creator_agent import CreatorAgent as CodeRefiningCreatorAgent
from agents.codeRefiningLLM_player.creator_agent import read_foo, write_foo, list_local_files, read_local_file

print("✅ Creator agent imports loaded")

# Import the v5_M creator agent specifically
from agents.fromScratchLLMStructured_player_v5_M.creator_agent import CreatorAgent as V5MCreatorAgent
print("✅ fromScratchLLMStructured_player_v5_M creator agent loaded")

from minimax import AlphaBetaPlayer
from catanatron_server.utils import open_link

print("✅ All imports completed successfully")
print("="*80)

def main():
    print("🎮 MAIN EXECUTION START")
    print("="*60)
    
    print("🤖 Initializing fromScratchLLMStructured_player_v5_M Creator Agent...")
    print("   This agent uses advanced multi-node architecture:")
    print("   - 🧠 Meta Node: High-level strategy coordination")
    print("   - 📊 Analyzer Node: Game performance analysis")  
    print("   - 🎯 Strategizer Node: Strategy development")
    print("   - 🔍 Researcher Node: Information gathering")
    print("   - 💻 Coder Node: Code implementation")
    print()
    
    try:
        # Initialize the V5_M creator agent
        print("🔧 Creating V5_M Creator Agent instance...")
        cA = V5MCreatorAgent()
        print("✅ V5_M Creator Agent initialized successfully!")
        
        print("\n🏗️  Agent Configuration:")
        print(f"   - LLM Model: {cA.llm_name}")
        print(f"   - Target File: foo_player.py")
        print(f"   - Run Directory: {cA.run_dir}")
        print(f"   - Max Evolution Cycles: 20")
        print(f"   - Victory Points to Win: 10 (quick games)")
        print()
        
        print("🚀 Starting ReAct Graph Execution...")
        print("   The agent will:")
        print("   1. 📖 Read current player code")
        print("   2. 🎮 Run test games against AlphaBeta opponent") 
        print("   3. 📈 Analyze performance results")
        print("   4. 🧠 Develop improvement strategies")
        print("   5. 🔍 Research game mechanics if needed")
        print("   6. 💻 Implement code improvements")
        print("   7. 🔄 Repeat until performance target met")
        print()
        print("⚠️  This may take 30-60 minutes depending on LLM response times")
        print("💡 Watch the console for detailed progress updates!")
        print("="*80)
        
        # Run the react graph - this is where the magic happens
        result = cA.run_react_graph()
        
        print("\n" + "="*80)
        print("🎉 CREATOR AGENT EXECUTION COMPLETED!")
        print("="*80)
        print("📊 Final Results Summary:")
        if result:
            print(f"✅ Agent execution successful")
            print(f"📁 Results saved in: {cA.run_dir}")
            print(f"🎯 Check the final foo_player.py for improved strategy")
        else:
            print("⚠️  Agent execution completed with warnings")
            
        print("\n🔍 Next Steps:")
        print("   1. Check the results directory for detailed logs")
        print("   2. Run 'python testing.py' to evaluate the improved agent")
        print("   3. Compare performance against different opponents")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR during Creator Agent execution:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   This might be due to:")
        print(f"   - Missing API keys (AWS_ACCESS_KEY, AWS_SECRET_KEY)")
        print(f"   - Network connectivity issues")
        print(f"   - LLM service rate limits")
        print(f"   - File permission issues")
        print("\n💡 Check your environment variables and try again")
        print("="*80)
        raise

if __name__ == "__main__":
    main()