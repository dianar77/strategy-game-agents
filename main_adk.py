import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "catanatron"))
minimax_dir = os.path.join(os.path.dirname(__file__), "catanatron/catanatron_experimental/catanatron_experimental/machine_learning/players")
sys.path.append(minimax_dir)

print("="*80)
print("STRATEGY GAME AGENTS - ADK CREATOR SYSTEM")
print("="*80)
print("🚀 Starting Google ADK Creator Agent for Catan Strategy Game")
print("🎯 Target: fromScratchLLMStructured_player_v7/v8 (ADK)")
print("🧠 This system uses Google ADK with multiple specialized agents")
print("="*80)

from agents.base_llm import OpenAILLM, MistralLLM, AzureOpenAILLM, AnthropicLLM
from catanatron import Game, RandomPlayer, Color
from agents.llm_player.llm_player import LLMPlayer
from agents.vanillaLLM_player.vanillaLLM_player import VanillaLLMPlayer
from agents.basicLang_player.basicLang_player import BasicLangPlayer
from agents.toolCallLLM_player.toolCallLLM_player import ToolCallLLMPlayer

print("✅ Core imports loaded successfully")

# Import the ADK creator agents
try:
    from agents.fromScratchLLMStructured_player_v7.creator_agent import CreatorAgent as V7CreatorAgent
    print("✅ fromScratchLLMStructured_player_v7 creator agent loaded")
    V7_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ v7 creator agent not available: {e}")
    V7_AVAILABLE = False

try:
    from agents.fromScratchLLMStructured_player_v8.creator_agent import CreatorAgent as V8CreatorAgent
    print("✅ fromScratchLLMStructured_player_v8 creator agent loaded")
    V8_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ v8 creator agent not available: {e}")
    V8_AVAILABLE = False

from minimax import AlphaBetaPlayer
from catanatron_server.utils import open_link

print("✅ All imports completed successfully")
print("="*80)

async def main():
    print("🎮 MAIN EXECUTION START")
    print("="*60)
    
    # Choose which version to run
    print("🤖 Available ADK Creator Agents:")
    if V7_AVAILABLE:
        print("   7️⃣ fromScratchLLMStructured_player_v7 (Multi-runner approach)")
    if V8_AVAILABLE:
        print("   8️⃣ fromScratchLLMStructured_player_v8 (Single-runner approach)")
    
    # Default to v8 if available, otherwise v7
    if V8_AVAILABLE:
        version = "v8"
        CreatorAgent = V8CreatorAgent
        print("🎯 Running v8 Creator Agent (Single-runner orchestration)")
    elif V7_AVAILABLE:
        version = "v7"
        CreatorAgent = V7CreatorAgent
        print("🎯 Running v7 Creator Agent (Multi-runner orchestration)")
    else:
        print("❌ No ADK Creator Agents available!")
        return
    
    print(f"   This agent uses Google ADK framework with:")
    print(f"   - 🧠 StrategizerAgent: Strategy development")
    print(f"   - 💻 CoderAgent: Code implementation")
    print(f"   - 🔍 ResearcherAgent: Information gathering")
    print(f"   - 📊 AnalyzerAgent: Performance analysis")
    print(f"   - 🧬 EvolverAgent: Evolution planning")
    print(f"   - 🎮 PlayerAgent: Game execution")
    print()
    
    try:
        # Initialize the ADK creator agent
        print(f"🔧 Creating {version} Creator Agent instance...")
        creator_agent = CreatorAgent()
        print(f"✅ {version} Creator Agent initialized successfully!")
        
        print(f"\n🏗️  Agent Configuration:")
        print(f"   - Framework: Google ADK")
        print(f"   - Model: {getattr(creator_agent, 'model', 'Ollama/LiteLLM')}")
        print(f"   - Target System: Catan Strategy Agent")
        print(f"   - Evolution Approach: Multi-agent orchestration")
        print()
        
        # Configure the evolution parameters
        target_system = "Catan Strategy Player"
        improvement_goal = "Develop optimal strategy for Catan game to beat AlphaBeta player"
        
        print("🚀 Starting ADK Evolution Process...")
        print("   The system will execute these phases:")
        print("   1. 🔍 Research & Strategy (Parallel execution)")
        print("      - Research best practices for Catan strategy")
        print("      - Create improvement strategy")
        print("   2. 🧬 Evolution Planning")
        print("      - Create detailed evolution plan")
        print("      - Use Chain-of-Thought reasoning")
        print("   3. 🛠️ Implementation & Analysis (Sequential execution)")
        print("      - Implement improvements")
        print("      - Execute and test")
        print("      - Analyze results")
        print("   4. 📊 Final Assessment")
        print("      - Evaluate evolution results")
        print("      - Plan next iteration")
        print()
        print("⚠️  This may take 60-90 minutes depending on LLM response times")
        print("💡 Watch the console for detailed progress updates!")
        print("="*80)
        
        # Run the evolution process
        result = await creator_agent.evolve_system(target_system, improvement_goal)
        
        print("\n" + "="*80)
        print("🎉 ADK CREATOR AGENT EXECUTION COMPLETED!")
        print("="*80)
        print("📊 Final Results Summary:")
        if result and result.get("status") == "completed":
            print(f"✅ Evolution process successful")
            print(f"🎯 Target System: {result.get('system', 'Unknown')}")
            print(f"📋 Goal: {result.get('goal', 'Unknown')}")
            if result.get("research_strategy"):
                print(f"🔍 Research & Strategy: ✅ Completed")
            if result.get("evolution_plan"):
                print(f"🧬 Evolution Plan: ✅ Completed")
            if result.get("implementation"):
                print(f"🛠️ Implementation: ✅ Completed")
            if result.get("assessment"):
                print(f"📊 Assessment: ✅ Completed")
        else:
            print("⚠️  Evolution process completed with warnings")
            if result:
                print(f"Status: {result.get('status', 'Unknown')}")
                
        print("\n🔍 Next Steps:")
        print("   1. Check the agent database (my_agent_data.db) for detailed logs")
        print("   2. Review the evolution results in the console output")
        print("   3. Run continuous evolution for multiple iterations")
        print("   4. Test the evolved strategy against different opponents")
        print("="*80)
        
        # Option to run continuous evolution
        print("\n🔄 Would you like to run continuous evolution? (Multiple iterations)")
        print("   This will run 3 evolution cycles to further improve the agent")
        # For now, let's just show the option without prompting
        print("   (To enable: uncomment the continuous_evolution call below)")
        
        # Uncomment this to run continuous evolution
        # print("\n🔄 Starting continuous evolution (3 iterations)...")
        # continuous_result = await creator_agent.continuous_evolution(target_system, iterations=3)
        # print("🎉 Continuous evolution completed!")
        # print(f"📊 Total iterations: {continuous_result.get('total_iterations', 0)}")
        
    except Exception as e:
        print(f"\n❌ ERROR during ADK Creator Agent execution:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   This might be due to:")
        print(f"   - Missing Google ADK dependencies")
        print(f"   - Ollama server not running (check http://localhost:11434)")
        print(f"   - LiteLLM configuration issues")
        print(f"   - Database connection problems")
        print(f"   - Required environment variables not set")
        print("\n💡 Troubleshooting steps:")
        print("   1. Check if Ollama is running: ollama serve")
        print("   2. Verify model is available: ollama pull llama3.1:8b")
        print("   3. Check environment variables in .env file")
        print("   4. Verify Google ADK installation")
        print("="*80)
        raise

def run_main():
    """Wrapper to run the async main function"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user")
        print("="*80)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        print("="*80)

if __name__ == "__main__":
    run_main()
