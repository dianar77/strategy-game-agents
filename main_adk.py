import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "catanatron"))
sys.path.append(os.path.join(os.path.dirname(__file__), "catanatron", "catanatron_core"))
# Add specific paths for v7 and v8 creator agents
sys.path.append(os.path.join(os.path.dirname(__file__), "agents", "fromScratchLLMStructured_player_v7"))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents", "fromScratchLLMStructured_player_v8"))
minimax_dir = os.path.join(os.path.dirname(__file__), "catanatron/catanatron_experimental/catanatron_experimental/machine_learning/players")
sys.path.append(minimax_dir)

print("="*80)
print("STRATEGY GAME AGENTS - ADK CREATOR SYSTEM")
print("="*80)
print("🚀 Starting Google ADK Creator Agent for Catan Strategy Game")
print("🎯 Target: fromScratchLLMStructured_player_v7/v8 (ADK)")
print("🧠 This system uses Google ADK with multiple specialized agents")
print("="*80)

try:
    from agents.base_llm import OpenAILLM, MistralLLM, AzureOpenAILLM, AnthropicLLM
    print("✅ Base LLM imports loaded successfully")
except ImportError as e:
    print(f"⚠️ Base LLM imports failed: {e}")

try:
    # Try the standard import first (if packages are properly installed)
    from catanatron import Game, RandomPlayer, Color
    print("✅ Catanatron imports loaded successfully")
except ImportError:
    try:
        # Fallback to direct path import
        from catanatron_core.catanatron import Game, RandomPlayer, Color
        print("✅ Catanatron imports loaded successfully (fallback path)")
    except ImportError as e:
        print(f"❌ Catanatron imports failed: {e}")
        print("💡 You need to install the catanatron packages first!")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)

try:
    from agents.llm_player.llm_player import LLMPlayer
    from agents.vanillaLLM_player.vanillaLLM_player import VanillaLLMPlayer
    from agents.basicLang_player.basicLang_player import BasicLangPlayer
    from agents.toolCallLLM_player.toolCallLLM_player import ToolCallLLMPlayer
    print("✅ Agent imports loaded successfully")
except ImportError as e:
    print(f"⚠️ Some agent imports failed: {e}")

print("✅ Core imports completed")

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

try:
    from minimax import AlphaBetaPlayer
    print("✅ AlphaBetaPlayer import loaded successfully")
except ImportError as e:
    print(f"⚠️ AlphaBetaPlayer import failed: {e}")
    print("   This is optional for the ADK creator agent")

try:
    from catanatron_server.utils import open_link
    print("✅ Catanatron server utils loaded successfully")
except ImportError as e:
    print(f"⚠️ Catanatron server utils failed: {e}")
    print("   This is optional for the ADK creator agent")

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
    
    # Default to v7 if available, otherwise v8 (v7 is more stable)
    if V7_AVAILABLE:
        version = "v7"
        CreatorAgent = V7CreatorAgent
        print("🎯 Running v7 Creator Agent (Multi-runner orchestration)")
    elif V8_AVAILABLE:
        version = "v8"
        CreatorAgent = V8CreatorAgent
        print("🎯 Running v8 Creator Agent (Single-runner orchestration)")
        print("⚠️  Note: v8 may have agent parent validation issues, v7 is recommended")
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
        # Initialize the ADK creator agent with proper resource management
        print(f"🔧 Creating {version} Creator Agent instance...")
        
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
        
        # Use context manager for proper resource cleanup
        async with CreatorAgent() as creator_agent:
            print(f"✅ {version} Creator Agent initialized successfully!")
            
            print(f"\n🏗️  Agent Configuration:")
            print(f"   - Framework: Google ADK")
            print(f"   - Model: {getattr(creator_agent, 'model', 'Ollama/LiteLLM')}")
            print(f"   - Target System: Catan Strategy Agent")
            print(f"   - Evolution Approach: Multi-agent orchestration")
            print()
            
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
            print("\n🔄 Starting continuous evolution (3 iterations)...")
            continuous_result = await creator_agent.continuous_evolution(target_system, iterations=3)
            print("🎉 Continuous evolution completed!")
            print(f"📊 Total iterations: {continuous_result.get('total_iterations', 0)}")
            
            # Note: The async context manager will automatically clean up resources here
            print("🧹 Cleaning up resources...")
        
    except Exception as e:
        print(f"\n❌ ERROR during ADK Creator Agent execution:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        
        # Check for specific validation errors related to agent parents
        if "already has a parent agent" in str(e) and version == "v8":
            print(f"\n🔧 SOLUTION: This is a known issue with v8 agent orchestration.")
            print(f"   Try running v7 instead by modifying the version selection logic.")
            print(f"   v7 uses multi-runner approach which avoids this issue.")
            
            # Try to fall back to v7 if available
            if V7_AVAILABLE:
                print(f"\n🔄 Attempting fallback to v7 Creator Agent...")
                try:
                    fallback_agent = V7CreatorAgent()
                    print(f"✅ v7 Creator Agent fallback initialized successfully!")
                    fallback_result = await fallback_agent.evolve_system(target_system, improvement_goal)
                    print(f"✅ v7 Creator Agent fallback completed successfully!")
                    return
                except Exception as fallback_e:
                    print(f"❌ v7 fallback also failed: {fallback_e}")
        
        print(f"\n   This might be due to:")
        print(f"   - Agent parent validation issues (use v7 instead of v8)")
        print(f"   - Missing Google ADK dependencies")
        print(f"   - Ollama server not running (check http://localhost:11434)")
        print(f"   - LiteLLM configuration issues")
        print(f"   - Database connection problems")
        print(f"   - Required environment variables not set")
        print("\n💡 Troubleshooting steps:")
        print("   1. Use v7 Creator Agent (more stable)")
        print("   2. Check if Ollama is running: ollama serve")
        print("   3. Verify model is available: ollama pull llama3.1:8b")
        print("   4. Check environment variables in .env file")
        print("   5. Verify Google ADK installation")
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
