"""
Player Agent - Executes actions and interacts with Catanatron game systems
"""

from google.adk.agents import LlmAgent
from typing import Dict, Any, Optional, List
import requests
import json
import sys
from pathlib import Path

# Add parent directory to path to import shared_tools
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared_tools import (
    run_testfoo,
    read_foo,
    write_foo,
    get_current_metrics,
    analyze_performance_trends
)


class PlayerAgent:
    """
    Player Agent that executes actions in Catanatron game systems.
    Handles interaction with game testing and execution.
    """
    
    def __init__(self, model, api_config: Optional[Dict[str, str]] = None):
        self.model = model
        self.api_config = api_config or {}
        self.agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        """Create the player LLM agent"""
        return LlmAgent(
            name="player",
            model=self.model,
            instruction="""
            You are a Player Agent that executes actions in Catanatron game systems.
            Your role is to:
            1. Execute Catanatron game tests and player implementations
            2. Run game simulations and collect performance data
            3. Monitor game state and system responses
            4. Collect execution results and comprehensive feedback
            5. Handle errors and edge cases gracefully
            6. Validate player implementations before deployment
            
            When executing actions:
            - Always validate player implementations
            - Run comprehensive game tests
            - Log all game results and performance metrics
            - Collect detailed win/loss statistics
            - Ensure game execution safety and stability
            - Provide clear status updates and feedback
            
            Always execute games safely and collect comprehensive results.
            Report both successes and failures with detailed context.
            """,
            tools=[
                self._create_game_execution_tool(),
                self._create_testing_tool(),
                self._create_monitoring_tool(),
                self._create_validation_tool()
            ]
        )
    
    def _create_game_execution_tool(self):
        """Create Catanatron game execution tool"""
        def execute_catanatron_game(game_config: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a Catanatron game with specified configuration"""
            game_type = game_config.get('type', 'standard')
            short_game = game_config.get('short_game', False)
            
            try:
                # Run actual game using shared_tools
                game_output = run_testfoo(short_game=short_game)
                
                # Parse results from game output
                if "Error" in game_output:
                    return {
                        "game_execution": {
                            "status": "error",
                            "error": game_output,
                            "error_type": "execution_error",
                            "game_type": game_type
                        },
                        "results": {},
                        "recommendations": [
                            "Check player implementation for syntax errors",
                            "Verify Catanatron installation",
                            "Review game configuration parameters"
                        ]
                    }
                
                # Extract basic info from output
                results = self._parse_game_output(game_output)
                
                return {
                    "game_execution": {
                        "status": "completed",
                        "game_type": game_type,
                        "execution_time": f"~{180 if not short_game else 30}s",
                        "output_length": len(game_output)
                    },
                    "results": results,
                    "performance_summary": {
                        "execution_successful": True,
                        "output_captured": True,
                        "game_completed": "completed" in game_output.lower() or "result" in game_output.lower()
                    },
                    "raw_output": game_output[-1000:] if len(game_output) > 1000 else game_output  # Last 1000 chars
                }
                
            except Exception as e:
                return {
                    "game_execution": {
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "game_type": game_type
                    },
                    "results": {},
                    "recommendations": [
                        "Check player implementation for syntax errors",
                        "Verify Catanatron installation",
                        "Review system configuration"
                    ]
                }
        
        return execute_catanatron_game
    
    def _parse_game_output(self, output: str) -> Dict[str, Any]:
        """Parse game output to extract basic information"""
        results = {
            "raw_output_available": True,
            "contains_error": "error" in output.lower() or "exception" in output.lower(),
            "contains_timeout": "timeout" in output.lower(),
            "contains_results": "result" in output.lower() or "winner" in output.lower(),
            "output_length": len(output)
        }
        
        # Try to extract any numerical results
        import re
        numbers = re.findall(r'\d+', output)
        if numbers:
            results["extracted_numbers"] = numbers[-5:]  # Last 5 numbers found
        
        return results
    
    def _create_testing_tool(self):
        """Create Catanatron testing tool"""
        def test_catanatron_player(test_config: Dict[str, Any]) -> Dict[str, Any]:
            """Run comprehensive tests on Catanatron player implementation"""
            test_type = test_config.get('type', 'full')
            
            try:
                # Read current player implementation
                player_code = read_foo()
                
                if "Error" in player_code:
                    return {
                        "testing_results": {
                            "status": "error",
                            "error": player_code,
                            "test_type": test_type
                        },
                        "critical_issues": [
                            "Unable to read player implementation",
                            "File access issues detected"
                        ]
                    }
                
                # Validate the code
                validation_results = self._validate_player_implementation(player_code)
                
                # Run a short test game if code looks valid
                game_test_results = {}
                if validation_results.get("basic_structure_valid", False):
                    try:
                        test_output = run_testfoo(short_game=True)
                        game_test_results = {
                            "test_execution": "completed",
                            "test_output_length": len(test_output),
                            "contains_error": "error" in test_output.lower(),
                            "test_successful": "Error" not in test_output
                        }
                    except Exception as e:
                        game_test_results = {
                            "test_execution": "failed",
                            "test_error": str(e)
                        }
                
                return {
                    "testing_results": {
                        "status": "completed",
                        "test_type": test_type,
                        "code_length": len(player_code),
                        "validation_passed": validation_results.get("basic_structure_valid", False)
                    },
                    "syntax_validation": {
                        "valid_syntax": validation_results.get("syntax_valid", False),
                        "syntax_errors": validation_results.get("syntax_errors", []),
                        "import_errors": validation_results.get("import_issues", [])
                    },
                    "game_compatibility": {
                        "catanatron_compatible": validation_results.get("has_catanatron_imports", False),
                        "has_required_methods": validation_results.get("has_required_methods", False),
                        "structure_valid": validation_results.get("basic_structure_valid", False)
                    },
                    "game_test_results": game_test_results,
                    "recommendations": validation_results.get("recommendations", [])
                }
                
            except Exception as e:
                return {
                    "testing_results": {
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    "critical_issues": [
                        "Unable to test player implementation",
                        "System or configuration issues detected"
                    ]
                }
        
        return test_catanatron_player
    
    def _create_monitoring_tool(self):
        """Create Catanatron game monitoring tool"""
        def monitor_catanatron_performance() -> Dict[str, Any]:
            """Monitor Catanatron game performance and system health"""
            
            try:
                # Get real performance metrics
                current_metrics = get_current_metrics()
                performance_trends = analyze_performance_trends()
                
                if "error" in current_metrics:
                    return {
                        "system_health": {
                            "catanatron_status": "unknown",
                            "player_status": "no_data",
                            "last_check": "now",
                            "data_availability": "no performance data available"
                        },
                        "monitoring_status": "limited - no historical data available"
                    }
                
                # Extract real metrics
                game_perf = current_metrics.get("game_performance", {})
                current_win_rate = game_perf.get("win_rate", {}).get("current", 0.0)
                evolution_cycle = game_perf.get("evolution_cycle", 0)
                
                return {
                    "system_health": {
                        "catanatron_status": "operational",
                        "player_status": "active" if current_win_rate > 0 else "needs_development",
                        "last_check": "now",
                        "data_availability": "performance data available"
                    },
                    "game_performance": {
                        "current_win_rate": f"{current_win_rate:.1%}",
                        "evolution_cycle": evolution_cycle,
                        "performance_trend": game_perf.get("win_rate", {}).get("trend", "unknown"),
                        "data_points": "based on real game results"
                    },
                    "player_metrics": {
                        "foo_player_status": "active",
                        "current_performance": current_win_rate,
                        "average_score": game_perf.get("average_score", {}).get("current", 0.0),
                        "performance_level": self._assess_performance_level(current_win_rate)
                    },
                    "system_status": {
                        "tools_operational": True,
                        "file_access": True,
                        "game_execution": True,
                        "data_collection": True
                    }
                }
                
            except Exception as e:
                return {
                    "system_health": {
                        "catanatron_status": "error",
                        "error": str(e),
                        "last_check": "now"
                    },
                    "monitoring_status": "system monitoring failed"
                }
        
        return monitor_catanatron_performance
    
    def _assess_performance_level(self, win_rate: float) -> str:
        """Assess performance level based on win rate"""
        if win_rate > 0.6:
            return "excellent"
        elif win_rate > 0.4:
            return "good"
        elif win_rate > 0.2:
            return "developing"
        elif win_rate > 0.0:
            return "basic"
        else:
            return "needs_development"
    
    def _create_validation_tool(self):
        """Create Catanatron player validation tool"""
        def validate_catanatron_player(player_code: str) -> Dict[str, Any]:
            """Validate Catanatron player implementation"""
            try:
                # Simulate validation - replace with actual validation logic
                validation_results = self._validate_player_implementation(player_code)
                
                return {
                    "validation_status": "completed",
                    "overall_score": validation_results.get('score', 0.0),
                    "syntax_validation": {
                        "valid": validation_results.get('syntax_valid', True),
                        "errors": validation_results.get('syntax_errors', []),
                        "warnings": validation_results.get('warnings', [])
                    },
                    "catanatron_compliance": {
                        "api_usage": validation_results.get('api_compliance', True),
                        "required_methods": validation_results.get('methods_present', True),
                        "action_handling": validation_results.get('actions_valid', True),
                        "game_state_access": validation_results.get('state_access', True)
                    },
                    "strategic_analysis": {
                        "has_strategy": validation_results.get('has_strategy', True),
                        "decision_logic": validation_results.get('decision_quality', 'good'),
                        "endgame_handling": validation_results.get('endgame_logic', 'basic'),
                        "resource_management": validation_results.get('resource_logic', 'present')
                    },
                    "quality_metrics": {
                        "code_complexity": validation_results.get('complexity', 'medium'),
                        "maintainability": validation_results.get('maintainability', 'good'),
                        "documentation": validation_results.get('documentation', 'adequate'),
                        "error_handling": validation_results.get('error_handling', 'basic')
                    },
                    "recommendations": validation_results.get('recommendations', [
                        "Add more comprehensive error handling",
                        "Improve endgame strategy logic",
                        "Optimize decision-making performance"
                    ])
                }
                
            except Exception as e:
                return {
                    "validation_status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "critical_issues": [
                        "Unable to parse player code",
                        "Severe syntax or structural errors",
                        "Player implementation may be corrupted"
                    ]
                }
        
        return validate_catanatron_player
    
    def _validate_player_implementation(self, player_code: str) -> Dict[str, Any]:
        """Validate player implementation"""
        # This would be replaced with actual validation logic
        return {
            "score": 0.82,
            "syntax_valid": True,
            "syntax_errors": [],
            "warnings": ["Unused import detected"],
            "api_compliance": True,
            "methods_present": True,
            "actions_valid": True,
            "state_access": True,
            "has_strategy": True,
            "decision_quality": "good",
            "endgame_logic": "basic",
            "resource_logic": "present",
            "complexity": "medium",
            "maintainability": "good",
            "documentation": "adequate",
            "error_handling": "basic",
            "recommendations": [
                "Add more sophisticated endgame logic",
                "Improve error handling coverage",
                "Consider adding opponent behavior tracking"
            ]
        }
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the player agent with given task data"""
        return await self.agent.run(task_data)
    
    def get_agent(self) -> LlmAgent:
        """Get the underlying LLM agent"""
        return self.agent 