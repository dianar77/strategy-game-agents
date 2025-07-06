import os
import subprocess
import shlex
import time
import re
from datetime import datetime

LLM = "claude-3.7"  # The LLM used for the v5_M agent
EVAL_PLAYER = "FOO_LLM_S5_M"  # The player you want to evaluate - fromScratchLLMStructured_player_v5_M
NUM_GAMES = 3  # Reduced for faster testing

print("="*80)
print("STRATEGY GAME AGENTS - TESTING CONFIGURATION")
print("="*80)
print(f"🤖 Testing Agent: {EVAL_PLAYER} (fromScratchLLMStructured_player_v5_M)")
print(f"🧠 Using LLM: {LLM}")
print(f"🎯 Number of games per opponent: {NUM_GAMES}")
print("="*80)

# Dictionary mapping player codes to their full names as they appear in summary
AGENTS = {
    "R": "RandomPlayer",
    "W": "WeightedRandomPlayer", 
    "VP": "VictoryPointPlayer",
    "G": "GreedyPlayoutsPlayer",
    "M": "MCTSPlayer",
    "F": "ValueFunctionPlayer",
    "AB": "AlphaBetaPlayer",
    "SAB": "SameTurnAlphaBetaPlayer",
    "VLLM": "VanillaLLMPlayer",
    "LLM": "LLMPlayer",
    "BL": "BasicLangPlayer",
    "TC": "ToolCallLLMPlayer",
    "FOO_S": "FooPlayer_Scratch",
    "FOO_LLM": "FooPlayer_LLM",
    "FOO_LLM_V2": "FooPlayer_LLM_V2",
    "FOO_LLM_S": "FooPlayer_LLM_Structured",
    "FOO_LLM_S2": "FooPlayer_LLM_Structured_V2", 
    "FOO_LLM_S3": "FooPlayer_LLM_Structured_V3",
    "FOO_LLM_S4": "FooPlayer_LLM_Structured_V4",
    "FOO_LLM_S5_M": "FooPlayer_LLM_Structured_V5_M",
    "PR_LLM": "PromptRefiningLLMPlayer",
    "CR_LLM": "CodeRefiningLLMPlayer",
}

# List opponents from strongest to weakest
print("🎯 Available opponents (strongest to weakest):")
print("   - AB (AlphaBetaPlayer): Advanced AI using minimax with alpha-beta pruning")
print("   - F (ValueFunctionPlayer): Uses hand-crafted value function")
print("   - M (MCTSPlayer): Monte Carlo Tree Search player") 
print("   - W (WeightedRandomPlayer): Smarter than random, prefers good moves")
print("   - VP (VictoryPointPlayer): Focuses on immediate victory points")
print("   - R (RandomPlayer): Completely random moves")
print()

# Start with a moderate opponent for testing
OPPONENTS = ["AB"]  # Start with the strongest opponent
print(f"🏆 Testing against: {[AGENTS[opp] + f' ({opp})' for opp in OPPONENTS]}")
print("="*80)

def run_game(eval_player, opponent, num_games):
    print(f"\n🚀 STARTING GAME SESSION")
    print(f"   Player 1 (Testing): {AGENTS[eval_player]} ({eval_player})")
    print(f"   Player 2 (Opponent): {AGENTS[opponent]} ({opponent})")
    print(f"   Number of games: {num_games}")
    print(f"   Victory points needed: 10 (reduced for faster games)")
    
    # Use the full path to the catanatron-play executable in the virtual environment
    catanatron_exe = os.path.join(os.getcwd(), ".venv", "Scripts", "catanatron-play.exe")
    command = f'"{catanatron_exe}" --players={opponent},{eval_player} --num={num_games} --output=data/ --json --config-vps-to-win=10'
    
    print(f"📝 Command to execute: {command}")
    print("⏱️  Starting game execution...")
    
    start = time.time()
    result = subprocess.run(
        shlex.split(command),
        capture_output=True,
        text=True,
        timeout=14400,  # 4 hour timeout
        check=False
    )
    elapsed = time.time() - start
    
    print(f"✅ Games completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    if result.returncode != 0:
        print(f"⚠️  WARNING: Command returned exit code {result.returncode}")
        if result.stderr:
            print(f"❌ STDERR: {result.stderr}")
    
    print(f"📊 Output length: {len(result.stdout)} characters")
    return result.stdout + result.stderr


def parse_agent_wins(output, agent_code):
    """
    Extracts the number of wins for agent from the Player Summary section,
    or falls back to counting wins from the Last 10 Games table.
    
    Args:
        output: The complete output text from the game
        agent_code: The code for the agent (e.g., "LLM", "AB")
    
    Returns:
        Number of wins for the agent
    """
    agent_name = AGENTS.get(agent_code, agent_code)  # Get full name or use code if not found
    
    # Try to parse from Player Summary first
    in_summary = False
    for line in output.splitlines():
        if "Player Summary" in line:
            in_summary = True
            print("Found Player Summary: " + line)
            continue
        if in_summary:
            # Look for the line that contains the agent name
            if agent_name in line:
                # Split on │ and get the WINS column (first after the player name)
                parts = [p.strip() for p in line.split("│")]
                if len(parts) > 1 and parts[1].isdigit():
                    return int(parts[1])
                # fallback: try to find the first integer after the player name
                for part in parts[1:]:
                    if part.isdigit():
                        return int(part)
            # End of summary table (empty line or table border)
            if line.strip() == "" or line.strip().startswith("+"):
                break
    
    # Fallback: count wins from Last 10 Games table
    print(f"Player Summary parsing failed, counting wins from game table for {agent_name}")
    wins = 0
    lines = output.splitlines()
    in_games_table = False
    
    for line in lines:
        if "Last 10 Games" in line or "Last " in line and "Games" in line:
            in_games_table = True
            continue
        if in_games_table and "|" in line and "WINNER" not in line and "----" not in line and "+---" not in line:
            # This is a game result line
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 7:  # Should have empty, game#, seating, turns, red_vp, blue_vp, winner
                winner = parts[6].strip()
                # Check if this agent won based on color mapping
                if (agent_code == "AB" and "BLUE" in winner) or (agent_code == "R" and "RED" in winner):
                    wins += 1
    return wins

def main():
    print("🎮 CATAN STRATEGY GAME EVALUATION")
    print("=" * 80)
    
    # Create a unique directory name based on current date/time
    timestamp = datetime.now().strftime("trial_%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Get the full name of the evaluation player
    eval_player_name = AGENTS.get(EVAL_PLAYER, EVAL_PLAYER)
    
    print(f"📁 Results directory: {results_dir}")
    print(f"🤖 Testing: {eval_player_name} ({EVAL_PLAYER})")
    print(f"🧠 LLM Model: {LLM}")
    print("=" * 80)

    # Track all results for summary
    summary_results = []

    for i, opponent in enumerate(OPPONENTS, 1):
        opponent_name = AGENTS.get(opponent, opponent)
        print(f"\n🎯 MATCH {i}/{len(OPPONENTS)}: {eval_player_name} vs {opponent_name}")
        print("-" * 60)
        
        try:
            output = run_game(EVAL_PLAYER, opponent, NUM_GAMES)
            
            print("💾 Saving game output...")
            # Save output to the timestamped directory
            filename = f"{LLM}_{EVAL_PLAYER}_vs_{opponent}.txt"
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "w") as f:
                f.write(output)
            print(f"   Saved to: {filename}")

            print("📈 Analyzing results...")
            # Parse wins for both players
            eval_wins = parse_agent_wins(output, EVAL_PLAYER)
            opponent_wins = parse_agent_wins(output, opponent)
            
            print(f"🏆 FINAL SCORES:")
            print(f"   {eval_player_name}: {eval_wins}/{NUM_GAMES} wins ({eval_wins/NUM_GAMES*100:.1f}%)")
            print(f"   {opponent_name}: {opponent_wins}/{NUM_GAMES} wins ({opponent_wins/NUM_GAMES*100:.1f}%)")
            
            # Store results for summary
            summary_results.append({
                "opponent": opponent,
                "opponent_name": opponent_name,
                "eval_wins": eval_wins,
                "opponent_wins": opponent_wins
            })

            if eval_wins > NUM_GAMES / 2:
                print(f"🎉 {eval_player_name} WINS against {opponent_name}! ({eval_wins}/{NUM_GAMES})")
                print("   Moving to next opponent...")
            else:
                print(f"😞 {eval_player_name} loses to {opponent_name} ({eval_wins}/{NUM_GAMES})")
                print("   Need to improve the agent...")
                break
                
        except subprocess.TimeoutExpired:
            print("⏰ ERROR: Game execution timed out after 4 hours!")
            break
        except Exception as e:
            print(f"❌ ERROR during game execution: {e}")
            break

    print("\n" + "="*80)
    print("📋 GENERATING SUMMARY REPORT")
    print("="*80)
    
    # Create a summary file
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"CATAN STRATEGY GAME EVALUATION SUMMARY\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Trial ID: {timestamp}\n")
        f.write(f"LLM Used: {LLM}\n")
        f.write(f"Agent Tested: {EVAL_PLAYER} ({eval_player_name})\n")
        f.write(f"Games per opponent: {NUM_GAMES}\n")
        f.write(f"Victory points needed: 10\n\n")
        f.write("RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        total_games = 0
        total_wins = 0
        
        for result in summary_results:
            win_rate = result['eval_wins'] / NUM_GAMES * 100 if NUM_GAMES > 0 else 0
            f.write(f"vs {result['opponent']} ({result['opponent_name']}):\n")
            f.write(f"  Score: {result['eval_wins']}-{result['opponent_wins']}\n")
            f.write(f"  Win Rate: {win_rate:.1f}%\n\n")
            
            total_games += NUM_GAMES
            total_wins += result['eval_wins']
        
        if total_games > 0:
            overall_win_rate = total_wins / total_games * 100
            f.write(f"OVERALL PERFORMANCE:\n")
            f.write(f"Total Wins: {total_wins}/{total_games} ({overall_win_rate:.1f}%)\n")

    print(f"📄 Summary saved to: {summary_path}")
    print(f"📁 All results in: {results_dir}")
    print("\n🎯 EVALUATION COMPLETE!")
    
    # Show final summary in console
    if summary_results:
        print("\n📊 QUICK SUMMARY:")
        for result in summary_results:
            win_rate = result['eval_wins'] / NUM_GAMES * 100
            status = "✅ WIN" if result['eval_wins'] > NUM_GAMES/2 else "❌ LOSS"
            print(f"   {status} vs {result['opponent_name']}: {result['eval_wins']}/{NUM_GAMES} ({win_rate:.1f}%)")

if __name__ == "__main__":
    main()