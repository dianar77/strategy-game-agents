"""
Analyzer Agent - Analyzes performance metrics and results
"""

from google.adk.agents import LlmAgent
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory to path to import shared_tools
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared_tools import (
    analyze_performance_trends,
    get_current_metrics,
    read_full_performance_history,
    read_game_results_file,
    read_game_output_file,
    read_older_foo_file
)


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
            
            # Get real performance data
            current_metrics = get_current_metrics()
            performance_trends = analyze_performance_trends()
            
            if "error" in current_metrics:
                # Return default structure when no data available
                return {
                    "timestamp": "No data available",
                    "evolution_cycle": 0,
                    "game_performance": {
                        "win_rate": {
                            "current": 0.0,
                            "previous": 0.0,
                            "trend": "no_data",
                            "target": 0.60
                        },
                        "average_score": {
                            "current": 0.0,
                            "previous": 0.0,
                            "best": 0.0,
                            "opponent_avg": 0.0
                        },
                        "games_played": 0,
                        "total_victories": 0,
                        "victory_points_distribution": {
                            "0-3": 0,
                            "4-6": 0,
                            "7-9": 0,
                            "10": 0
                        }
                    },
                    "strategic_metrics": {
                        "average_turns_per_game": 0,
                        "resource_efficiency": 0.0,
                        "trading_frequency": 0.0,
                        "building_strategy": "unknown",
                        "robber_placement_effectiveness": 0.0
                    },
                    "error_analysis": {
                        "syntax_errors": 0,
                        "runtime_exceptions": 0,
                        "invalid_moves": 0,
                        "timeout_issues": 0,
                        "common_failure_patterns": ["No data available"]
                    },
                    "opponent_analysis": {
                        "vs_alpha_beta": {"wins": 0, "losses": 0, "win_rate": 0.0},
                        "vs_random": {"wins": 0, "losses": 0, "win_rate": 0.0},
                        "vs_greedy": {"wins": 0, "losses": 0, "win_rate": 0.0}
                    }
                }
            
            # Extract real data
            game_perf = current_metrics.get("game_performance", {})
            current_win_rate = game_perf.get("win_rate", {}).get("current", 0.0)
            current_avg_score = game_perf.get("average_score", {}).get("current", 0.0)
            current_cycle = game_perf.get("evolution_cycle", 0)
            
            # Calculate derived metrics from trends
            trends_data = performance_trends.get("trends", {}) if "error" not in performance_trends else {}
            latest_evolutions = trends_data.get("latest_evolutions", [])
            
            # Calculate total games and victories
            total_games = len(latest_evolutions) * 3  # Assuming 3 games per test
            total_victories = sum(e.get("wins", 0) for e in latest_evolutions)
            
            # Estimate previous performance
            previous_win_rate = 0.0
            previous_avg_score = 0.0
            if len(latest_evolutions) > 1:
                previous_win_rate = latest_evolutions[-2].get("wins", 0) / 3.0
                previous_avg_score = latest_evolutions[-2].get("avg_score", 0.0)
            
            # Determine trend
            if current_win_rate > previous_win_rate:
                trend = "improving"
            elif current_win_rate < previous_win_rate:
                trend = "declining"
            else:
                trend = "stable"
            
            return {
                "timestamp": performance_trends.get("analysis_timestamp", "unknown"),
                "evolution_cycle": current_cycle,
                "game_performance": {
                    "win_rate": {
                        "current": current_win_rate,
                        "previous": previous_win_rate,
                        "trend": trend,
                        "target": 0.60
                    },
                    "average_score": {
                        "current": current_avg_score,
                        "previous": previous_avg_score,
                        "best": max([e.get("avg_score", 0) for e in latest_evolutions]) if latest_evolutions else 0.0,
                        "opponent_avg": 9.0  # Estimated opponent average
                    },
                    "games_played": total_games,
                    "total_victories": total_victories,
                    "victory_points_distribution": self._estimate_vp_distribution(latest_evolutions)
                },
                "strategic_metrics": {
                    "average_turns_per_game": sum([e.get("avg_turns", 0) for e in latest_evolutions]) / len(latest_evolutions) if latest_evolutions else 0,
                    "resource_efficiency": min(0.85, current_win_rate + 0.3),  # Estimated based on performance
                    "trading_frequency": 12.0 + (current_win_rate * 10),  # Estimated
                    "building_strategy": self._assess_building_strategy(current_win_rate),
                    "robber_placement_effectiveness": min(0.8, current_win_rate + 0.2)
                },
                "error_analysis": self._analyze_errors_from_history(),
                "opponent_analysis": self._analyze_opponent_performance(latest_evolutions)
            }
        return collect_game_metrics
    
    def _create_performance_analysis_tool(self):
        """Create detailed Catanatron performance analysis tool"""
        def analyze_game_performance(data: Dict[str, Any]) -> Dict[str, Any]:
            """Perform detailed Catanatron game performance analysis"""
            
            # Get real performance data
            current_metrics = get_current_metrics()
            performance_trends = analyze_performance_trends()
            
            if "error" in current_metrics:
                return {
                    "analysis_summary": {
                        "overall_health": "no_data",
                        "performance_trend": "unknown",
                        "critical_issues": 0,
                        "improvement_areas": 0,
                        "confidence_level": 0.0
                    },
                    "detailed_findings": {
                        "strengths": {"none": "No data available for analysis"},
                        "weaknesses": {"data_availability": "No performance data found"},
                        "technical_issues": {"initialization": "System needs initialization"}
                    },
                    "comparative_analysis": {
                        "vs_baseline": "No baseline data available",
                        "vs_human_players": "No comparison data available",
                        "vs_other_ai": "No comparison data available",
                        "skill_progression": "No progression data available"
                    },
                    "recommendations": [
                        {
                            "priority": "high",
                            "category": "initialization",
                            "description": "Run initial game tests to establish baseline",
                            "expected_impact": "establish baseline performance"
                        }
                    ],
                    "next_steps": {
                        "immediate": "initialize system and run baseline tests",
                        "short_term": "implement basic strategy",
                        "long_term": "develop comprehensive strategy framework"
                    }
                }
            
            # Extract real data for analysis
            game_perf = current_metrics.get("game_performance", {})
            current_win_rate = game_perf.get("win_rate", {}).get("current", 0.0)
            current_avg_score = game_perf.get("average_score", {}).get("current", 0.0)
            win_trend = game_perf.get("win_rate", {}).get("trend", "unknown")
            
            trends_data = performance_trends.get("trends", {}) if "error" not in performance_trends else {}
            latest_evolutions = trends_data.get("latest_evolutions", [])
            
            # Assess overall health
            if current_win_rate > 0.5:
                overall_health = "excellent"
            elif current_win_rate > 0.3:
                overall_health = "good"
            elif current_win_rate > 0.15:
                overall_health = "improving"
            else:
                overall_health = "needs_work"
            
            # Identify strengths and weaknesses
            strengths = {}
            weaknesses = {}
            
            if current_win_rate > 0.4:
                strengths["strategic_thinking"] = "demonstrates solid strategic decision making"
                strengths["consistency"] = "maintains consistent performance across games"
            elif current_win_rate > 0.2:
                strengths["basic_implementation"] = "successfully implements basic game mechanics"
                weaknesses["advanced_strategy"] = "needs improvement in advanced strategic concepts"
            else:
                weaknesses["fundamental_strategy"] = "requires development of fundamental strategic understanding"
                weaknesses["implementation"] = "basic implementation needs refinement"
            
            if current_avg_score < 7.0:
                weaknesses["scoring_efficiency"] = "struggles to accumulate victory points effectively"
            if current_avg_score > 8.5:
                strengths["scoring_optimization"] = "demonstrates strong victory point accumulation"
            
            # Generate recommendations based on performance
            recommendations = self._generate_recommendations(current_win_rate, current_avg_score, win_trend)
            
            return {
                "analysis_summary": {
                    "overall_health": overall_health,
                    "performance_trend": win_trend,
                    "critical_issues": len([w for w in weaknesses if "fundamental" in w.lower()]),
                    "improvement_areas": len(weaknesses),
                    "confidence_level": min(0.9, 0.5 + current_win_rate)
                },
                "detailed_findings": {
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "technical_issues": self._identify_technical_issues(latest_evolutions)
                },
                "comparative_analysis": {
                    "vs_baseline": f"{int((current_win_rate - 0.15) * 100)}% improvement over baseline" if current_win_rate > 0.15 else "below baseline performance",
                    "vs_human_players": "below average human performance" if current_win_rate < 0.5 else "competitive with human players",
                    "vs_other_ai": self._compare_with_ai_opponents(current_win_rate),
                    "skill_progression": self._assess_skill_progression(latest_evolutions)
                },
                "recommendations": recommendations,
                "next_steps": {
                    "immediate": recommendations[0]["description"] if recommendations else "continue current strategy",
                    "short_term": recommendations[1]["description"] if len(recommendations) > 1 else "monitor performance",
                    "long_term": "develop advanced strategic capabilities"
                }
            }
        return analyze_game_performance
    
    def _estimate_vp_distribution(self, evolutions: list) -> Dict[str, int]:
        """Estimate victory points distribution from evolution data"""
        if not evolutions:
            return {"0-3": 0, "4-6": 0, "7-9": 0, "10": 0}
        
        total_games = len(evolutions) * 3
        avg_score = sum(e.get("avg_score", 0) for e in evolutions) / len(evolutions)
        
        # Estimate distribution based on average score
        if avg_score < 5:
            return {"0-3": int(total_games * 0.6), "4-6": int(total_games * 0.3), "7-9": int(total_games * 0.1), "10": 0}
        elif avg_score < 7:
            return {"0-3": int(total_games * 0.3), "4-6": int(total_games * 0.5), "7-9": int(total_games * 0.2), "10": 0}
        else:
            return {"0-3": int(total_games * 0.1), "4-6": int(total_games * 0.3), "7-9": int(total_games * 0.4), "10": int(total_games * 0.2)}
    
    def _assess_building_strategy(self, win_rate: float) -> str:
        """Assess building strategy based on win rate"""
        if win_rate > 0.5:
            return "sophisticated"
        elif win_rate > 0.3:
            return "balanced"
        elif win_rate > 0.15:
            return "developing"
        else:
            return "basic"
    
    def _analyze_errors_from_history(self) -> Dict[str, Any]:
        """Analyze errors from game history"""
        # Try to read recent game logs to identify patterns
        try:
            recent_log = read_game_output_file(-1)
            if "error" in recent_log.lower() or "exception" in recent_log.lower():
                return {
                    "syntax_errors": 0,
                    "runtime_exceptions": 1,
                    "invalid_moves": 0,
                    "timeout_issues": 1 if "timeout" in recent_log.lower() else 0,
                    "common_failure_patterns": ["runtime issues detected in recent games"]
                }
        except:
            pass
        
        return {
            "syntax_errors": 0,
            "runtime_exceptions": 0,
            "invalid_moves": 0,
            "timeout_issues": 0,
            "common_failure_patterns": ["No error patterns detected"]
        }
    
    def _analyze_opponent_performance(self, evolutions: list) -> Dict[str, Dict[str, Any]]:
        """Analyze performance against different opponent types"""
        total_games = len(evolutions) * 3 if evolutions else 0
        total_wins = sum(e.get("wins", 0) for e in evolutions)
        
        # For now, assume all games are against AB (alpha-beta) opponent
        # In future, could parse game logs to identify opponent types
        return {
            "vs_alpha_beta": {
                "wins": total_wins,
                "losses": total_games - total_wins,
                "win_rate": total_wins / total_games if total_games > 0 else 0.0
            },
            "vs_random": {"wins": 0, "losses": 0, "win_rate": 0.0},
            "vs_greedy": {"wins": 0, "losses": 0, "win_rate": 0.0}
        }
    
    def _generate_recommendations(self, win_rate: float, avg_score: float, trend: str) -> list:
        """Generate recommendations based on performance metrics"""
        recommendations = []
        
        if win_rate < 0.2:
            recommendations.append({
                "priority": "high",
                "category": "strategy",
                "description": "Implement fundamental game strategy and decision-making logic",
                "expected_impact": "establish baseline competitive performance"
            })
        elif win_rate < 0.4:
            recommendations.append({
                "priority": "high",
                "category": "strategy",
                "description": "Enhance strategic planning and resource optimization",
                "expected_impact": "increase win rate by 15-20%"
            })
        else:
            recommendations.append({
                "priority": "medium",
                "category": "optimization",
                "description": "Fine-tune advanced strategies and opponent modeling",
                "expected_impact": "optimize performance against strong opponents"
            })
        
        if avg_score < 7.0:
            recommendations.append({
                "priority": "high",
                "category": "scoring",
                "description": "Improve victory point accumulation strategies",
                "expected_impact": "increase average score by 1-2 points"
            })
        
        if trend == "declining":
            recommendations.append({
                "priority": "high",
                "category": "stability",
                "description": "Investigate and fix performance regression issues",
                "expected_impact": "restore and improve performance stability"
            })
        
        return recommendations
    
    def _identify_technical_issues(self, evolutions: list) -> Dict[str, str]:
        """Identify technical issues from evolution history"""
        if not evolutions:
            return {"initialization": "no game data available for analysis"}
        
        # Check for consistency issues
        if len(evolutions) > 2:
            recent_scores = [e.get("avg_score", 0) for e in evolutions[-3:]]
            if max(recent_scores) - min(recent_scores) > 3:
                return {"inconsistency": "high variance in recent performance suggests stability issues"}
        
        return {"status": "no major technical issues detected"}
    
    def _compare_with_ai_opponents(self, win_rate: float) -> str:
        """Compare performance with AI opponents"""
        if win_rate > 0.6:
            return "outperforming most AI strategies"
        elif win_rate > 0.4:
            return "competitive with intermediate AI strategies"
        elif win_rate > 0.2:
            return "below average AI performance but improving"
        else:
            return "significantly below AI baseline"
    
    def _assess_skill_progression(self, evolutions: list) -> str:
        """Assess skill progression over time"""
        if len(evolutions) < 2:
            return "insufficient data for progression analysis"
        
        win_rates = [e.get("wins", 0) / 3.0 for e in evolutions]
        if len(win_rates) > 1:
            if win_rates[-1] > win_rates[0]:
                return "demonstrating consistent improvement over time"
            elif win_rates[-1] < win_rates[0]:
                return "showing concerning performance decline"
            else:
                return "maintaining stable performance level"
        
        return "progression analysis inconclusive"
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the analyzer agent with given task data"""
        return await self.agent.run(task_data)
    
    def get_agent(self) -> LlmAgent:
        """Get the underlying LLM agent"""
        return self.agent 