import os
import subprocess
import shlex
import time
import re
from datetime import datetime

LLM = "gpt-4o"  # The LLM used
EVAL_PLAYER = "LLM"  # The player you want to evaluate
AGENT_NAME = "LLMPlayer"
NUM_GAMES = 15

# List opponents from strongest to weakest
OPPONENTS = ["AB", "F", "G", "M", "W", "VP", "R"]
#OPPONENTS = ["VP", "R"]

def run_game(eval_player, opponent, num_games):
    command = f"catanatron-play --players={opponent},{eval_player} --num={num_games} --output=data/ --json"
    print(f"Running: {command}")
    start = time.time()
    result = subprocess.run(
        shlex.split(command),
        capture_output=True,
        text=True,
        timeout=14400,
        check=False
    )
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.1f}s")
    return result.stdout + result.stderr


def parse_agent_wins(output, agent_name):
    """
    Extracts the number of wins for agent_name from the Player Summary section,
    regardless of color or extra info.
    """
    in_summary = False
    for line in output.splitlines():
        if "Player Summary" in line:
            in_summary = True
            print("Found Player Summary: " + line)
            continue
        if in_summary:
            # Look for the line that starts with the agent name (ignoring leading/trailing spaces)
            if line.strip().startswith(agent_name + ":"):
                # Split on │ and get the WINS column (first after the player name)
                parts = [p.strip() for p in line.split("│")]
                if len(parts) > 1 and parts[1].isdigit():
                    return int(parts[1])
                # fallback: try to find the first integer after the player name
                for part in parts[1:]:
                    if part.isdigit():
                        return int(part)
                return 0
            # End of summary table (empty line or table border)
            if line.strip() == "" or line.strip().startswith("╵"):
                break
    return 0

def main():
    # Create a unique directory name based on current date/time
    timestamp = datetime.now().strftime("trial_%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")

    for opponent in OPPONENTS:
        print(f"\n=== Evaluating {EVAL_PLAYER} vs {opponent} ===")
        output = run_game(EVAL_PLAYER, opponent, NUM_GAMES)

        # Save output to the timestamped directory
        filename = f"{LLM}_{EVAL_PLAYER}_vs_{opponent}.txt"
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "w") as f:
            f.write(output)

        # Parse wins
        llm_wins = parse_agent_wins(output, AGENT_NAME)
        print(f"{AGENT_NAME} wins: {llm_wins}/{NUM_GAMES}")

        if llm_wins > NUM_GAMES / 2:
            print(f"{AGENT_NAME} beats {opponent} ({llm_wins}/{NUM_GAMES}). Stopping evaluation.")
            break
        else:
            print(f"{AGENT_NAME} did not beat {opponent} ({llm_wins}/{NUM_GAMES}). Trying next opponent...")

    # Create a summary file
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(f"Trial: {timestamp}\n")
        f.write(f"LLM: {LLM}\n")
        f.write(f"Player: {EVAL_PLAYER}\n")
        f.write(f"Games per opponent: {NUM_GAMES}\n\n")
        f.write("Results:\n")
        # You could add more summary information here if needed

    print(f"\nEvaluation complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main()