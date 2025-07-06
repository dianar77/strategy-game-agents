"""
Shared Tools Module - Real functionality for Catanatron agent evolution

This module provides real tools for file operations, game execution, 
and performance tracking, replacing the hardcoded mock data.
"""

import time
import os
import sys
import json
import shutil
import subprocess
import shlex
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


# Configuration - adjust these paths as needed
LOCAL_CATANATRON_BASE_DIR = (Path(__file__).parent.parent.parent / "catanatron").resolve()
FOO_TARGET_FILENAME = "foo_player.py"
FOO_MAX_BYTES = 64_000
FOO_RUN_COMMAND = "catanatron-play --players=AB,FOO_LLM_S7  --num=3  --config-vps-to-win=10"

# Global run directory - will be set by first agent that initializes
RUN_DIR = None
CURRENT_EVOLUTION = 0


def initialize_run_directory():
    """Initialize the run directory for this evolution session"""
    global RUN_DIR
    if RUN_DIR is None:
        agent_dir = Path(__file__).parent
        runs_dir = agent_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        run_id = datetime.now().strftime("creator_%Y%m%d_%H%M%S")
        RUN_DIR = runs_dir / run_id
        RUN_DIR.mkdir(exist_ok=True)
        
        # Copy template player file to working location
        template_file = agent_dir / f"__TEMPLATE__{FOO_TARGET_FILENAME}"
        target_file = agent_dir / FOO_TARGET_FILENAME
        if template_file.exists():
            shutil.copy2(template_file, target_file)
    
    return RUN_DIR


def get_foo_target_file() -> Path:
    """Get the path to the current foo player file"""
    return Path(__file__).parent / FOO_TARGET_FILENAME


# File Operations
def list_catanatron_files() -> List[str]:
    """Return all files beneath BASE_DIR"""
    try:
        files = []
        for p in LOCAL_CATANATRON_BASE_DIR.glob("**/*"):
            if p.is_file() and p.suffix in {".py", ".txt", ".md"}:
                files.append(str(p.relative_to(LOCAL_CATANATRON_BASE_DIR)))
        return files
    except Exception as e:
        return [f"Error listing files: {e}"]


def read_local_file(rel_path: str) -> str:
    """Return the text content of rel_path if it's accessible"""
    try:
        # Handle foo player file
        if rel_path == FOO_TARGET_FILENAME:
            return read_foo()
        
        # Handle Catanatron files
        if rel_path.startswith("catanatron/"):
            candidate = (LOCAL_CATANATRON_BASE_DIR / rel_path.replace("catanatron/", "")).resolve()
            if not str(candidate).startswith(str(LOCAL_CATANATRON_BASE_DIR)) or not candidate.is_file():
                raise ValueError("Access denied or not a file")
            if candidate.stat().st_size > 64_000:
                raise ValueError("File too large")
            return candidate.read_text(encoding="utf-8", errors="ignore")
        
        # Handle run directory files
        run_dir = initialize_run_directory()
        run_path = run_dir / rel_path
        if run_path.exists() and run_path.is_file():
            if run_path.stat().st_size > 64_000:
                raise ValueError("File too large")
            return run_path.read_text(encoding="utf-8", errors="ignore")
        
        # Check Catanatron directory
        candidate = (LOCAL_CATANATRON_BASE_DIR / rel_path).resolve()
        if not str(candidate).startswith(str(LOCAL_CATANATRON_BASE_DIR)) or not candidate.is_file():
            raise ValueError(f"Access denied or file not found: {rel_path}")
        if candidate.stat().st_size > 64_000:
            raise ValueError("File too large")
        return candidate.read_text(encoding="utf-8", errors="ignore")
        
    except Exception as e:
        return f"Error reading file {rel_path}: {e}"


def read_foo() -> str:
    """Return the UTF-8 content of the current player file"""
    try:
        foo_file = get_foo_target_file()
        if not foo_file.exists():
            return "Player file does not exist"
        if foo_file.stat().st_size > FOO_MAX_BYTES:
            return "File too large for the agent"
        return foo_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading player file: {e}"


def write_foo(new_text: str) -> str:
    """Overwrite the player file with new_text"""
    try:
        if len(new_text.encode()) > FOO_MAX_BYTES:
            return "Refusing to write >64 kB"
        
        foo_file = get_foo_target_file()
        foo_file.write_text(new_text, encoding="utf-8")
        return f"{FOO_TARGET_FILENAME} updated successfully"
    except Exception as e:
        return f"Error writing player file: {e}"


# Game Execution
def run_testfoo(short_game: bool = False) -> str:
    """Run one Catanatron match and return results"""
    global CURRENT_EVOLUTION
    
    try:
        run_dir = initialize_run_directory()
        
        if short_game:
            run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S_vg")
        else:
            run_id = datetime.now().strftime("game_%Y%m%d_%H%M%S_fg")
        
        game_run_dir = run_dir / run_id
        game_run_dir.mkdir(exist_ok=True)
        
        # Save current player file
        cur_foo_path = game_run_dir / FOO_TARGET_FILENAME
        foo_file = get_foo_target_file()
        if foo_file.exists():
            shutil.copy2(foo_file, cur_foo_path)
        
        MAX_CHARS = 20_000
        timeout = 30 if short_game else 14400
        
        # Run the game
        result = subprocess.run(
            shlex.split(FOO_RUN_COMMAND),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        
        stdout_limited = result.stdout[-MAX_CHARS:] if result.stdout else ""
        stderr_limited = result.stderr[-MAX_CHARS:] if result.stderr else ""
        game_results = (stdout_limited + stderr_limited).strip()
        
        # Save output
        output_file_path = game_run_dir / "game_output.txt"
        with open(output_file_path, "w") as f:
            f.write(game_results)
        
        # Extract and copy JSON results
        json_path = None
        path_match = re.search(r'results_file_path:([^\s]+)', game_results)
        if path_match:
            json_path = path_match.group(1).strip()
        
        json_content = {}
        json_copy_path = "None"
        if json_path and Path(json_path).exists():
            json_filename = Path(json_path).name
            json_copy_path = game_run_dir / json_filename
            shutil.copy2(json_path, json_copy_path)
            
            try:
                with open(json_path, 'r') as f:
                    json_content = json.load(f)
            except json.JSONDecodeError:
                json_content = {"error": "Failed to parse JSON file"}
        
        # Update performance history for full games
        if not short_game:
            _update_performance_history(game_run_dir, output_file_path, cur_foo_path, json_copy_path, json_content)
            CURRENT_EVOLUTION += 1
        
        return game_results
        
    except subprocess.TimeoutExpired as e:
        stdout_output = e.stdout or ""
        stderr_output = e.stderr or ""
        if isinstance(stdout_output, bytes):
            stdout_output = stdout_output.decode('utf-8', errors='ignore')
        if isinstance(stderr_output, bytes):
            stderr_output = stderr_output.decode('utf-8', errors='ignore')
        return "Game Ended From Timeout.\n\n" + (stdout_output + stderr_output)[-MAX_CHARS:]
    except Exception as e:
        return f"Error running game: {e}"


def _update_performance_history(game_run_dir, output_file_path, cur_foo_path, json_copy_path, json_content):
    """Update the performance history with game results"""
    global CURRENT_EVOLUTION
    
    run_dir = initialize_run_directory()
    performance_history_path = run_dir / "performance_history.json"
    
    try:
        with open(performance_history_path, 'r') as f:
            performance_history = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        performance_history = {}
    
    # Extract stats from JSON
    wins = 0
    avg_score = 0
    avg_turns = 0
    
    try:
        if "Player Summary" in json_content:
            for player, stats in json_content["Player Summary"].items():
                if player.startswith("FooPlayer"):
                    wins = stats.get("WINS", 0)
                    avg_score = stats.get("AVG VP", 0)
        
        if "Game Summary" in json_content:
            avg_turns = json_content["Game Summary"].get("AVG TURNS", 0)
    except Exception as e:
        print(f"Error extracting stats: {e}")
    
    # Update performance history
    evolution_key = f"Evolution {CURRENT_EVOLUTION}"
    performance_history[evolution_key] = {
        "timestamp": datetime.now().isoformat(),
        "wins": wins,
        "avg_score": avg_score,
        "avg_turns": avg_turns,
        "full_game_log_path": str(output_file_path.relative_to(run_dir)),
        "cur_foo_player_path": str(cur_foo_path.relative_to(run_dir)),
        "json_game_results_path": str(json_copy_path.relative_to(run_dir)) if json_copy_path != "None" else "None"
    }
    
    with open(performance_history_path, 'w') as f:
        json.dump(performance_history, f, indent=2)


# Performance Tracking
def read_full_performance_history() -> str:
    """Return the content of performance_history.json"""
    try:
        run_dir = initialize_run_directory()
        performance_history_path = run_dir / "performance_history.json"
        
        if not performance_history_path.exists():
            return "Performance history file does not exist."
        
        if performance_history_path.stat().st_size > 64_000:
            return "Performance history file is too large (>64 KB)."
        
        with open(performance_history_path, 'r') as f:
            performance_history = json.load(f)
            return json.dumps(performance_history, indent=2)
    except Exception as e:
        return f"Error reading performance history: {e}"


def read_game_output_file(num: int = -1) -> str:
    """Return the contents of the game log for the chosen Evolution"""
    entry, err = _get_evolution_entry(num)
    if err:
        return err
    
    path = entry.get("full_game_log_path")
    if not path or path == "None":
        return f"No game-output file recorded for Evolution {num}."
    
    try:
        return read_local_file(path)
    except Exception as e:
        return f"Error reading '{path}': {e}"


def read_game_results_file(num: int = -1) -> str:
    """Return the contents of the JSON game-results file for the chosen Evolution"""
    entry, err = _get_evolution_entry(num)
    if err:
        return err
    
    path = entry.get("json_game_results_path")
    if not path or path == "None":
        return f"No game-results file recorded for Evolution {num}."
    
    try:
        return read_local_file(path)
    except Exception as e:
        return f"Error reading '{path}': {e}"


def read_older_foo_file(num: int = -1) -> str:
    """Return the contents of the foo_player.py file saved for the chosen Evolution"""
    entry, err = _get_evolution_entry(num)
    if err:
        return err
    
    path = entry.get("cur_foo_player_path")
    if not path or path == "None":
        return f"No foo-player file recorded for Evolution {num}."
    
    try:
        return read_local_file(path)
    except Exception as e:
        return f"Error reading '{path}': {e}"


def _get_evolution_entry(num: int) -> Tuple[Dict[str, Any], str]:
    """Return (entry, "") on success or (None, error_msg) on failure"""
    perf_str = read_full_performance_history()
    try:
        perf = json.loads(perf_str)
    except json.JSONDecodeError:
        return None, f"Could not parse performance history JSON:\n{perf_str}"
    
    if not perf:
        return None, "Performance history is empty."
    
    # Choose evolution index
    if num == -1:
        # Latest evolution
        nums = [int(k.split()[1]) for k in perf if k.startswith("Evolution ")]
        if not nums:
            return None, "No Evolution entries found."
        num = max(nums)
    
    key = f"Evolution {num}"
    if key not in perf:
        return None, f"{key} not found in performance history."
    
    return perf[key], ""


# Analysis Functions
def analyze_performance_trends() -> Dict[str, Any]:
    """Analyze performance trends from real data"""
    try:
        perf_str = read_full_performance_history()
        perf_data = json.loads(perf_str)
        
        if not perf_data:
            return {"error": "No performance data available"}
        
        # Extract metrics from all evolutions
        evolutions = []
        for key, data in perf_data.items():
            if key.startswith("Evolution "):
                num = int(key.split()[1])
                evolutions.append({
                    "number": num,
                    "wins": data.get("wins", 0),
                    "avg_score": data.get("avg_score", 0),
                    "avg_turns": data.get("avg_turns", 0),
                    "timestamp": data.get("timestamp", "")
                })
        
        if not evolutions:
            return {"error": "No evolution data found"}
        
        evolutions.sort(key=lambda x: x["number"])
        
        # Calculate trends
        win_rates = [e["wins"] for e in evolutions]
        avg_scores = [e["avg_score"] for e in evolutions]
        
        current_win_rate = win_rates[-1] if win_rates else 0
        current_avg_score = avg_scores[-1] if avg_scores else 0
        
        # Calculate trend
        win_trend = "improving" if len(win_rates) > 1 and win_rates[-1] > win_rates[-2] else "declining"
        
        return {
            "current_performance": {
                "win_rate": current_win_rate / 3.0 if current_win_rate else 0,  # Assuming 3 games per test
                "average_score": current_avg_score,
                "evolution_count": len(evolutions)
            },
            "trends": {
                "win_rate_trend": win_trend,
                "score_progression": avg_scores,
                "latest_evolutions": evolutions[-3:] if len(evolutions) >= 3 else evolutions
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Error analyzing performance: {e}"}


def get_current_metrics() -> Dict[str, Any]:
    """Get current performance metrics from real data"""
    try:
        analysis = analyze_performance_trends()
        if "error" in analysis:
            return analysis
        
        current_perf = analysis["current_performance"]
        trends = analysis["trends"]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "game_performance": {
                "win_rate": {
                    "current": current_perf["win_rate"],
                    "trend": trends["win_rate_trend"],
                    "target": 0.60
                },
                "average_score": {
                    "current": current_perf["average_score"],
                    "best": max(trends["score_progression"]) if trends["score_progression"] else 0
                },
                "evolution_cycle": current_perf["evolution_count"]
            },
            "strategic_metrics": {
                "games_analyzed": len(trends["latest_evolutions"]) * 3,  # Assuming 3 games per evolution
                "improvement_rate": "steady" if trends["win_rate_trend"] == "improving" else "needs_work"
            }
        }
    except Exception as e:
        return {"error": f"Error getting current metrics: {e}"}


# Web search placeholder (implement if needed)
def web_search_tool_call(query: str) -> str:
    """Placeholder for web search functionality"""
    return f"Web search not implemented. Query was: {query}" 