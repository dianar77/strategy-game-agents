#!/usr/bin/env python3
"""
STRATEGY GAME AGENTS - EXAMPLE RUNNER
====================================

This script demonstrates how to run different parts of the system:
1. Quick agent test games
2. Creator agent (evolves strategies)  
3. Full evaluation testing

For your fromScratchLLMStructured_player_v7 agent specifically.
"""

import os
import sys
import subprocess
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

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
    print("3. 📊 Run Full Evaluation (vs Multiple Opponents)")
    print("4. 📁 View Recent Results")
    print("5. ❓ Show System Info")
    print("6. 🚪 Exit")
    print("="*50)

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

def run_creator_agent():
    """Run the creator agent to evolve the strategy."""
    print("\n🧠 CREATOR AGENT MODE")
    print("="*40)
    print("This will run the AI creator that evolves foo_player.py")
    print("⚠️  This may take 30-60 minutes!")
    print("💡 The agent will test, analyze, and improve the strategy iteratively")
    
    confirm = input("\nProceed? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    try:
        print("\n🚀 Starting main_adk.py (Creator Agent)...")
        result = subprocess.run([sys.executable, "main_adk.py"], 
                              capture_output=False, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ Creator agent completed successfully!")
        else:
            print(f"⚠️  Creator agent finished with exit code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Creator agent timed out after 1 hour")
    except KeyboardInterrupt:
        print("\n🛑 Creator agent interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to continue...")

def run_full_evaluation():
    """Run the full evaluation testing."""
    print("\n📊 FULL EVALUATION MODE")  
    print("="*40)
    print("This will run testing.py to evaluate agent performance")
    print("Tests the evolved agent against different opponents")
    
    try:
        print("\n🚀 Starting testing.py...")
        result = subprocess.run([sys.executable, "testing.py"], 
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
        "testing.py", 
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
    print("3. testing.py: Evaluates the agent against different opponents")
    print("4. main_adk.py: Runs the creator agent evolution process")
    
    print("\n🎯 AGENT TYPES:")
    print("- FOO_LLM_S5_M: Your advanced structured LLM player")
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