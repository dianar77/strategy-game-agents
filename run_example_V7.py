#!/usr/bin/env python3
"""
STRATEGY GAME AGENTS - EXAMPLE RUNNER
====================================

This script demonstrates how to run different parts of the system:
1. Quick agent test games
2. Creator agent (evolves strategies)  
3. Full evaluation testing

For your fromScratchLLMStructured_player_v7 agent specifically.
Uses context manager for proper resource cleanup.
"""

import os
import sys
import subprocess
import asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add paths for creator agent import
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents", "fromScratchLLMStructured_player_v7"))

print("="*80)
print("🎮 CATAN STRATEGY GAME AGENTS - EXAMPLE RUNNER")
print("="*80)
print("Target Agent: fromScratchLLMStructured_player_v7 (FOO_LLM_S7)")
print("="*80)

def show_menu():
    """Display the main menu options."""
    print("\n🎯 WHAT WOULD YOU LIKE TO DO?")
    print("="*50)
    print("1. 🎮 Run Quick Test Games (3 games vs AlphaBeta)")
    print("2. 🧠 Run Creator Agent (Evolve Strategy)")
    print("3. 🔄 Run Continuous Evolution (3 iterations)")
    print("4. 📊 Run Full Evaluation (vs Multiple Opponents)")
    print("5. 📁 View Recent Results")
    print("6. ❓ Show System Info")
    print("7. 🚪 Exit")
    print("="*50)
    print("💡 Creator Agent now uses context manager for clean resource management!")

def run_quick_test():
    """Run a quick test of the current agent."""
    print("\n🎮 QUICK TEST MODE")
    print("="*40)
    print("Running 3 games against AlphaBetaPlayer...")
    print("This will test the current foo_player.py without modification")
    
    try:
        cmd = '.venv\\Scripts\\catanatron-play.exe --players=AB,FOO_LLM_S7 --num=3 --output=data/ --json --config-vps-to-win=10'
        print(f"Command: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Games completed successfully!")
            print("\nGame output summary:")
            output_lines = result.stdout.split('\n')
            for line in output_lines[-20:]:  # Show last 20 lines
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"❌ Error running games (exit code: {result.returncode})")
            if result.stderr:
                print(f"Error details: {result.stderr}")
                
    except subprocess.TimeoutExpired:
        print("⏰ Games timed out after 10 minutes")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to continue...")

async def run_creator_agent_async():
    """Run the creator agent to evolve the strategy using context manager."""
    print("\n🧠 CREATOR AGENT MODE")
    print("="*40)
    print("This will run the AI creator that evolves foo_player.py")
    print("⚠️  This may take 30-60 minutes!")
    print("💡 The agent will test, analyze, and improve the strategy iteratively")
    print("✅ Using context manager for proper resource cleanup")
    
    try:
        # Import the creator agent
        from agents.fromScratchLLMStructured_player_v7.creator_agent import CreatorAgent
        
        # Configure evolution parameters
        target_system = "Catan Strategy Player"
        improvement_goal = "Develop optimal strategy for Catan game to beat AlphaBeta player"
        
        print(f"\n🎯 Target System: {target_system}")
        print(f"📋 Improvement Goal: {improvement_goal}")
        print()
        
        # Use context manager for automatic resource cleanup
        print("🔧 Initializing Creator Agent with context manager...")
        async with CreatorAgent() as creator_agent:
            print("✅ Creator Agent initialized successfully!")
            print("🛡️  Resources will be cleaned up automatically")
            print()
            
            # Display agent configuration
            print("🏗️  Agent Configuration:")
            print(f"   - Framework: Google ADK")
            print(f"   - Model: {getattr(creator_agent, 'model', 'Ollama/LiteLLM')}")
            print(f"   - Evolution Approach: Multi-agent orchestration")
            print()
            
            # Run the evolution process
            print("🚀 Starting evolution process...")
            print("   Evolution phases:")
            print("   1. 🔍 Research & Strategy (Parallel)")
            print("   2. 🧬 Evolution Planning")
            print("   3. 🛠️ Implementation & Analysis (Sequential)")
            print("   4. 📊 Final Assessment")
            print()
            
            result = await creator_agent.evolve_system(target_system, improvement_goal)
            
            # Display results
            print("\n" + "="*60)
            print("🎉 CREATOR AGENT EXECUTION COMPLETED!")
            print("="*60)
            
            if result and result.get("status") == "completed":
                print("✅ Evolution process successful!")
                print(f"🎯 Target System: {result.get('system', 'Unknown')}")
                print(f"📋 Goal: {result.get('goal', 'Unknown')}")
                print()
                
                # Show phase completion status
                phases = [
                    ("research_strategy", "🔍 Research & Strategy"),
                    ("evolution_plan", "🧬 Evolution Plan"),
                    ("implementation", "🛠️ Implementation"),
                    ("assessment", "📊 Assessment")
                ]
                
                print("📊 Phase Results:")
                for key, name in phases:
                    if result.get(key):
                        print(f"   {name}: ✅ Completed")
                    else:
                        print(f"   {name}: ⚠️  Incomplete")
                        
            else:
                print("⚠️  Evolution process completed with warnings")
                if result:
                    print(f"Status: {result.get('status', 'Unknown')}")
                    if result.get('error'):
                        print(f"Error: {result.get('error')}")
            
            print("\n🔍 Next Steps:")
            print("   1. Check my_agent_data.db for detailed logs")
            print("   2. Review foo_player.py for improvements")
            print("   3. Run full evaluation to test performance")
            print("   4. Consider running continuous evolution")
            
        # Resources automatically cleaned up here
        print("\n🧹 Resources automatically cleaned up!")
        print("✅ No 'unclosed client session' warnings!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the v7 creator agent is available")
        print("   Check: agents/fromScratchLLMStructured_player_v7/creator_agent.py")
        
    except Exception as e:
        print(f"❌ Error during evolution: {e}")
        print("💡 Check the error details above")
        import traceback
        traceback.print_exc()

def run_creator_agent():
    """Wrapper function to run the async creator agent."""
    confirm = input("\nProceed with Creator Agent? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    try:
        print("\n🚀 Starting Creator Agent with context manager...")
        # Run the async creator agent
        asyncio.run(run_creator_agent_async())
        
    except KeyboardInterrupt:
        print("\n🛑 Creator agent interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to continue...")

async def run_continuous_evolution_async():
    """Run continuous evolution with multiple iterations using context manager."""
    print("\n🔄 CONTINUOUS EVOLUTION MODE")
    print("="*40)
    print("This will run 3 evolution cycles to continuously improve the strategy")
    print("⚠️  This may take 2-3 hours!")
    print("💡 Each cycle builds on the previous improvements")
    print("✅ Using context manager for proper resource cleanup")
    
    try:
        # Import the creator agent
        from agents.fromScratchLLMStructured_player_v7.creator_agent import CreatorAgent
        
        # Configure evolution parameters
        target_system = "Catan Strategy Player"
        
        print(f"\n🎯 Target System: {target_system}")
        print(f"🔄 Iterations: 3")
        print()
        
        # Use context manager for automatic resource cleanup
        print("🔧 Initializing Creator Agent for continuous evolution...")
        async with CreatorAgent() as creator_agent:
            print("✅ Creator Agent initialized successfully!")
            print("🛡️  Resources will be cleaned up automatically")
            print()
            
            # Run continuous evolution
            print("🚀 Starting continuous evolution process...")
            print("   This will run 3 complete evolution cycles")
            print("   Each cycle includes all 4 phases of evolution")
            print()
            
            result = await creator_agent.continuous_evolution(target_system, iterations=3)
            
            # Display results
            print("\n" + "="*60)
            print("🎉 CONTINUOUS EVOLUTION COMPLETED!")
            print("="*60)
            
            if result and result.get("status") == "completed":
                print("✅ Continuous evolution successful!")
                print(f"🔄 Total iterations: {result.get('total_iterations', 0)}")
                print(f"✅ Successful iterations: {result.get('successful_iterations', 0)}")
                print(f"⚠️  Failed iterations: {result.get('total_iterations', 0) - result.get('successful_iterations', 0)}")
                
                success_rate = (result.get('successful_iterations', 0) / result.get('total_iterations', 1)) * 100
                print(f"📊 Success rate: {success_rate:.1f}%")
                
            else:
                print("⚠️  Continuous evolution completed with issues")
                if result:
                    print(f"Status: {result.get('status', 'Unknown')}")
                    if result.get('error'):
                        print(f"Error: {result.get('error')}")
            
            print("\n🔍 Next Steps:")
            print("   1. Check my_agent_data.db for detailed evolution logs")
            print("   2. Review foo_player.py for all accumulated improvements")
            print("   3. Run full evaluation to test final performance")
            print("   4. Compare with baseline performance")
            
        # Resources automatically cleaned up here
        print("\n🧹 Resources automatically cleaned up!")
        print("✅ No 'unclosed client session' warnings!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the v7 creator agent is available")
        
    except Exception as e:
        print(f"❌ Error during continuous evolution: {e}")
        print("💡 Check the error details above")
        import traceback
        traceback.print_exc()

def run_continuous_evolution():
    """Wrapper function to run the async continuous evolution."""
    confirm = input("\nProceed with Continuous Evolution (3 iterations)? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    try:
        print("\n🚀 Starting Continuous Evolution with context manager...")
        # Run the async continuous evolution
        asyncio.run(run_continuous_evolution_async())
        
    except KeyboardInterrupt:
        print("\n🛑 Continuous evolution interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to continue...")

def run_full_evaluation():
    """Run the full evaluation testing."""
    print("\n📊 FULL EVALUATION MODE")  
    print("="*40)
    print("This will run testing_adk.py to evaluate agent performance")
    print("Tests the evolved agent against different opponents")
    
    try:
        print("\n🚀 Starting testing_adk.py...")
        result = subprocess.run([sys.executable, "testing_adk.py"], 
                              capture_output=False, text=True, timeout=1800)
        
        if result.returncode == 0:
            print("✅ Evaluation completed successfully!")
        else:
            print(f"⚠️  Evaluation finished with exit code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Evaluation timed out after 30 minutes")
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to continue...")

def view_results():
    """Show recent results."""
    print("\n📁 RECENT RESULTS")
    print("="*40)
    
    # Check results directory
    if os.path.exists("results"):
        result_dirs = [d for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))]
        result_dirs.sort(reverse=True)  # Most recent first
        
        if result_dirs:
            print("Recent evaluation runs:")
            for i, dirname in enumerate(result_dirs[:5]):  # Show last 5
                print(f"   {i+1}. {dirname}")
        else:
            print("No evaluation results found.")
    else:
        print("No results directory found.")
    
    # Check agent runs
    agent_runs_dir = "agents/fromScratchLLMStructured_player_v7/runs"
    if os.path.exists(agent_runs_dir):
        run_dirs = [d for d in os.listdir(agent_runs_dir) if os.path.isdir(os.path.join(agent_runs_dir, d))]
        run_dirs.sort(reverse=True)
        
        if run_dirs:
            print("\nRecent creator agent runs:")
            for i, dirname in enumerate(run_dirs[:5]):
                print(f"   {i+1}. {dirname}")
        else:
            print("No creator agent runs found.")
    
    input("\nPress Enter to continue...")

def show_system_info():
    """Show system information and status."""
    print("\n❓ SYSTEM INFORMATION")
    print("="*40)
    
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check if virtual environment is active
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment: Active")
    else:
        print("⚠️  Virtual environment: Not detected")
    
    # Check important files
    important_files = [
        "main_adk.py",
        "testing_adk.py", 
        "agents/fromScratchLLMStructured_player_v7/creator_agent.py",
        "agents/fromScratchLLMStructured_player_v7/foo_player.py",
        ".venv/Scripts/catanatron-play.exe"
    ]
    
    print("\n📋 File status:")
    for file_path in important_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")
    
    # Check environment variables
    print("\n🔑 Environment variables:")
    env_vars = ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"]
    for var in env_vars:
        if os.environ.get(var):
            print(f"   ✅ {var}: Set")
        else:
            print(f"   ❌ {var}: Not set")
    
    print("\n💡 HOW THE SYSTEM WORKS:")
    print("1. foo_player.py: The Catan player agent that gets evolved")
    print("2. creator_agent.py: AI that analyzes and improves foo_player.py")
    print("3. testing_adk.py: Evaluates the agent against different opponents")
    print("4. main_adk.py: Runs the creator agent evolution process")
    
    print("\n🎯 AGENT TYPES:")
    print("- FOO_LLM_S7: Your advanced structured LLM player")
    print("- AB: AlphaBetaPlayer (strongest opponent)")
    print("- F: ValueFunctionPlayer (strong opponent)")
    print("- R: RandomPlayer (weakest opponent)")
    
    input("\nPress Enter to continue...")



def main():
    """Main menu loop."""
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                run_quick_test()
            elif choice == "2":
                run_creator_agent()
            elif choice == "3":
                run_full_evaluation()
            elif choice == "4":
                view_results()
            elif choice == "5":
                show_system_info()
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 