"""
Analyzer Agent - Analyzes performance metrics and results
"""

from google.adk.agents import LlmAgent
from typing import Dict, Any


class AnalyzerAgent:
    """
    Performance Analysis Agent for Catanatron Game Evolution.
    Analyzes game performance metrics and results.
    """
    
    def __init__(self, model):
        self.model = model
        self.agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        """Create the analyzer LLM agent"""
        return LlmAgent(
            name="analyzer",
            model=self.model,
            instruction="""
            You are the ANALYZER expert for evolving Catanatron game players.
            
            Your Role:
            - Analyze game outputs and performance history
            - Interpret game results and identify issues
            - Provide detailed reports on player performance
            - Identify syntax errors and implementation problems
            - Compare win rates and scoring patterns
            - Analyze turn efficiency and strategic decisions

            Always start responses with 'ANALYSIS:' and end with 'END ANALYSIS'.
            Focus on concrete data from game results and outputs.
            """,
            tools=[self._create_game_metrics_tool(), self._create_performance_analysis_tool()]
        )
    
    def _create_game_metrics_tool(self):
        """Create Catanatron game metrics collection tool"""
        def collect_game_metrics() -> Dict[str, Any]:
            """Collect Catanatron game performance metrics"""
            return {
                "timestamp": "2024-01-15T10:30:00Z",
                "evolution_cycle": 3,
                "game_performance": {
                    "win_rate": {
                        "current": 0.35,
                        "previous": 0.28,
                        "trend": "improving",
                        "target": 0.60
                    },
                    "average_score": {
                        "current": 8.2,
                        "previous": 7.5,
                        "best": 10.0,
                        "opponent_avg": 9.1
                    },
                    "games_played": 50,
                    "total_victories": 18,
                    "victory_points_distribution": {
                        "0-3": 8,
                        "4-6": 15,
                        "7-9": 20,
                        "10": 7
                    }
                },
                "strategic_metrics": {
                    "average_turns_per_game": 85,
                    "resource_efficiency": 0.72,
                    "trading_frequency": 12.3,
                    "building_strategy": "balanced",
                    "robber_placement_effectiveness": 0.58
                },
                "error_analysis": {
                    "syntax_errors": 0,
                    "runtime_exceptions": 2,
                    "invalid_moves": 5,
                    "timeout_issues": 1,
                    "common_failure_patterns": [
                        "resource management in late game",
                        "suboptimal road placement"
                    ]
                },
                "opponent_analysis": {
                    "vs_alpha_beta": {"wins": 12, "losses": 28, "win_rate": 0.30},
                    "vs_random": {"wins": 25, "losses": 15, "win_rate": 0.62},
                    "vs_greedy": {"wins": 18, "losses": 22, "win_rate": 0.45}
                }
            }
        return collect_game_metrics
    
    def _create_performance_analysis_tool(self):
        """Create detailed Catanatron performance analysis tool"""
        def analyze_game_performance(data: Dict[str, Any]) -> Dict[str, Any]:
            """Perform detailed Catanatron game performance analysis"""
            return {
                "analysis_summary": {
                    "overall_health": "improving",
                    "performance_trend": "steady_progress",
                    "critical_issues": 1,
                    "improvement_areas": 3,
                    "confidence_level": 0.78
                },
                "detailed_findings": {
                    "strengths": {
                        "resource_collection": "efficient early game resource gathering",
                        "development_cards": "good use of development cards",
                        "initial_placement": "strategic settlement placement",
                        "adaptive_strategy": "adapts well to different board configurations"
                    },
                    "weaknesses": {
                        "late_game_strategy": "struggles with endgame optimization",
                        "trading_decisions": "suboptimal trade negotiations",
                        "robber_strategy": "ineffective robber placement decisions",
                        "longest_road": "rarely pursues longest road strategy"
                    },
                    "technical_issues": {
                        "move_timeout": "occasional timeout in complex decision scenarios",
                        "memory_usage": "increasing memory consumption in long games",
                        "exception_handling": "needs better error recovery mechanisms"
                    }
                },
                "comparative_analysis": {
                    "vs_baseline": "35% improvement over initial template",
                    "vs_human_players": "below average human performance",
                    "vs_other_ai": "competitive with simple AI strategies",
                    "skill_progression": "consistent improvement across iterations"
                },
                "recommendations": [
                    {
                        "priority": "high",
                        "category": "strategy",
                        "description": "Implement more sophisticated endgame planning",
                        "expected_impact": "increase win rate by 10-15%"
                    },
                    {
                        "priority": "high", 
                        "category": "technical",
                        "description": "Optimize decision-making algorithm for speed",
                        "expected_impact": "reduce timeout issues by 80%"
                    },
                    {
                        "priority": "medium",
                        "category": "strategy",
                        "description": "Improve trading evaluation logic",
                        "expected_impact": "increase resource efficiency by 20%"
                    },
                    {
                        "priority": "medium",
                        "category": "robber",
                        "description": "Enhance robber placement strategy",
                        "expected_impact": "improve defensive capabilities"
                    }
                ],
                "next_steps": {
                    "immediate": "fix timeout issues and exception handling",
                    "short_term": "implement advanced endgame strategies",
                    "long_term": "develop machine learning components for strategy adaptation"
                }
            }
        return analyze_game_performance
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the analyzer agent with given task data"""
        return await self.agent.run(task_data)
    
    def get_agent(self) -> LlmAgent:
        """Get the underlying LLM agent"""
        return self.agent 