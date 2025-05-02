import os
import subprocess
import shlex
import time
import re

LLM = "gpt-4o"  # The LLM used
EVAL_PLAYER = "LLM"  # The player you want to evaluate
NUM_GAMES = 5

# List opponents from strongest to weakest
OPPONENTS = ["AB", "F", "G", "M", "W", "VP", "R"]

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

def parse_llm_wins(output, eval_player):
    # Try to count wins for eval_player in the output (looks for "Winner: LLM" or similar)
    # Adjust regex if your output format is different
    pattern = re.compile(r'Winner:\s*' + re.escape(eval_player))
    return len(pattern.findall(output))

def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for opponent in OPPONENTS:
        print(f"\n=== Evaluating {EVAL_PLAYER} vs {opponent} ===")
        output = run_game(EVAL_PLAYER, opponent, NUM_GAMES)

        # Save output
        filename = f"{LLM}_{EVAL_PLAYER}_vs_{opponent}.txt"
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "w") as f:
            f.write(output)

        # Parse wins
        llm_wins = parse_llm_wins(output, EVAL_PLAYER)
        print(f"{EVAL_PLAYER} wins: {llm_wins}/{NUM_GAMES}")

        if llm_wins > NUM_GAMES / 2:
            print(f"{EVAL_PLAYER} beats {opponent} ({llm_wins}/{NUM_GAMES}). Stopping evaluation.")
            break
        else:
            print(f"{EVAL_PLAYER} did not beat {opponent} ({llm_wins}/{NUM_GAMES}). Trying next opponent...")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()