"""
Strategizer Agent - Plans improvement strategies and optimization approaches
"""

from google.adk.agents import LlmAgent
from google.adk.tools.google_search_tool import GoogleSearchTool
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
    read_foo
)


class StrategizerAgent:
    """
    Strategic Planning Agent for Catanatron game improvement.
    Plans improvement strategies and optimization approaches.
    """
    
    def __init__(self, model):
        self.model = model
        self.agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        """Create the strategizer LLM agent"""
        return LlmAgent(
            name="strategizer",
            model=self.model,
            instruction="""
            You are the STRATEGIZER expert for evolving Catanatron game players.
            
            Your Role:
            - Generate new strategic approaches for Catanatron gameplay
            - Analyze previous strategy effectiveness
            - Recommend strategic improvements based on game analysis
            - Develop comprehensive game plans and tactics
            - Search for new strategy ideas when needed

            Always start responses with 'STRATEGY:' and end with 'END STRATEGY'.
            Be creative and look for breakthrough approaches in:
            - Settlement and city placement
            - Resource management and trading
            - Development card usage
            - Robber placement strategies
            - Endgame optimization
            """,
            tools=[self._create_strategy_analysis_tool(), self._create_strategy_generation_tool()]
        )
    
    def _create_strategy_analysis_tool(self):
        """Create Catanatron strategy analysis tool"""
        def analyze_strategy_effectiveness(data: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze effectiveness of current Catanatron strategies"""
            
            # Get real performance data
            current_metrics = get_current_metrics()
            performance_trends = analyze_performance_trends()
            
            if "error" in current_metrics:
                return {
                    "strategy_evaluation": {
                        "overall_effectiveness": 0.0,
                        "strategy_category": "uninitialized",
                        "win_rate_by_strategy": {
                            "no_data": 0.0
                        }
                    },
                    "current_strengths": {
                        "early_game": {"effectiveness": 0.0, "key_tactics": ["no data available"]},
                        "mid_game": {"effectiveness": 0.0, "key_tactics": ["no data available"]},
                        "late_game": {"effectiveness": 0.0, "key_tactics": ["no data available"]}
                    },
                    "strategic_weaknesses": {
                        "initialization": {
                            "issue": "no performance data available",
                            "impact": "cannot assess strategy effectiveness",
                            "fix_priority": "high"
                        }
                    },
                    "opponent_adaptation": {
                        "status": "no opponent data available"
                    }
                }
            
            # Extract real performance data
            game_perf = current_metrics.get("game_performance", {})
            current_win_rate = game_perf.get("win_rate", {}).get("current", 0.0)
            win_trend = game_perf.get("win_rate", {}).get("trend", "unknown")
            
            # Determine strategy category based on performance
            if current_win_rate > 0.6:
                strategy_category = "advanced_adaptive"
            elif current_win_rate > 0.4:
                strategy_category = "balanced_competitive"
            elif current_win_rate > 0.2:
                strategy_category = "developing_strategy"
            else:
                strategy_category = "basic_implementation"
            
            overall_effectiveness = current_win_rate
            
            # Analyze phase effectiveness based on win rate
            early_game_eff = min(1.0, current_win_rate + 0.2)  # Usually strongest phase
            mid_game_eff = current_win_rate
            late_game_eff = max(0.0, current_win_rate - 0.1)  # Often weakest
            
            # Get recent game data for more specific analysis
            trends_data = performance_trends.get("trends", {}) if "error" not in performance_trends else {}
            latest_evolutions = trends_data.get("latest_evolutions", [])
            
            # Analyze strategy effectiveness patterns
            win_rate_by_strategy = self._analyze_strategy_patterns(latest_evolutions, current_win_rate)
            
            # Identify strengths and weaknesses
            strengths = self._identify_strategic_strengths(current_win_rate, win_trend)
            weaknesses = self._identify_strategic_weaknesses(current_win_rate, latest_evolutions)
            
            return {
                "strategy_evaluation": {
                    "overall_effectiveness": round(overall_effectiveness, 2),
                    "strategy_category": strategy_category,
                    "win_rate_by_strategy": win_rate_by_strategy
                },
                "current_strengths": {
                    "early_game": {
                        "effectiveness": round(early_game_eff, 2),
                        "key_tactics": strengths.get("early_game", ["basic implementation"])
                    },
                    "mid_game": {
                        "effectiveness": round(mid_game_eff, 2),
                        "key_tactics": strengths.get("mid_game", ["adaptive play"])
                    },
                    "late_game": {
                        "effectiveness": round(late_game_eff, 2),
                        "key_tactics": strengths.get("late_game", ["endgame focus"])
                    }
                },
                "strategic_weaknesses": weaknesses,
                "opponent_adaptation": self._analyze_opponent_adaptation(current_win_rate)
            }
        return analyze_strategy_effectiveness
    
    def _analyze_strategy_patterns(self, evolutions: list, current_win_rate: float) -> Dict[str, float]:
        """Analyze patterns in strategy effectiveness"""
        if not evolutions:
            return {"no_data": 0.0}
        
        # Simulate different strategy win rates based on overall performance
        # In a real implementation, this would parse game logs to identify which strategies were used
        base_rate = current_win_rate
        
        return {
            "resource_focus": round(base_rate + 0.1, 2),
            "trading_heavy": round(base_rate - 0.05, 2),
            "development_cards": round(base_rate + 0.05, 2),
            "aggressive_expansion": round(base_rate, 2),
            "defensive_play": round(base_rate - 0.1, 2)
        }
    
    def _identify_strategic_strengths(self, win_rate: float, trend: str) -> Dict[str, list]:
        """Identify strategic strengths based on performance"""
        strengths = {
            "early_game": [],
            "mid_game": [],
            "late_game": []
        }
        
        if win_rate > 0.4:
            strengths["early_game"].extend(["efficient resource collection", "strategic positioning"])
            strengths["mid_game"].extend(["balanced development", "adaptive trading"])
        
        if win_rate > 0.6:
            strengths["late_game"].extend(["sophisticated endgame", "victory optimization"])
        
        if trend == "improving":
            strengths["mid_game"].append("learning and adaptation")
        
        # Default if no strengths identified
        for phase in strengths:
            if not strengths[phase]:
                strengths[phase] = ["basic implementation"]
        
        return strengths
    
    def _identify_strategic_weaknesses(self, win_rate: float, evolutions: list) -> Dict[str, Dict[str, str]]:
        """Identify strategic weaknesses based on performance"""
        weaknesses = {}
        
        if win_rate < 0.3:
            weaknesses["fundamental_strategy"] = {
                "issue": "lacks basic strategic understanding",
                "impact": "low overall performance",
                "fix_priority": "high"
            }
        
        if win_rate < 0.5:
            weaknesses["endgame_planning"] = {
                "issue": "insufficient late-game strategy",
                "impact": "loses close games",
                "fix_priority": "high"
            }
        
        # Check for consistency issues
        if len(evolutions) > 2:
            win_rates = [e.get("wins", 0) / 3.0 for e in evolutions]
            if len(win_rates) > 1 and max(win_rates) - min(win_rates) > 0.3:
                weaknesses["consistency"] = {
                    "issue": "inconsistent performance across games",
                    "impact": "unpredictable results",
                    "fix_priority": "medium"
                }
        
        if not weaknesses:
            weaknesses["optimization"] = {
                "issue": "room for strategic optimization",
                "impact": "potential performance gains available",
                "fix_priority": "low"
            }
        
        return weaknesses
    
    def _analyze_opponent_adaptation(self, win_rate: float) -> Dict[str, str]:
        """Analyze adaptation to different opponent types"""
        if win_rate < 0.2:
            return {
                "vs_all_opponents": "needs fundamental strategy improvement",
                "adaptation_level": "basic"
            }
        elif win_rate < 0.4:
            return {
                "vs_aggressive_players": "developing defensive strategies",
                "vs_passive_players": "learning to exploit opportunities",
                "adaptation_level": "developing"
            }
        else:
            return {
                "vs_aggressive_players": "effective counter-strategies",
                "vs_trade_focused": "competitive trading approach",
                "vs_development_heavy": "balanced counter-play",
                "adaptation_level": "advanced"
            }
    
    def _create_strategy_generation_tool(self):
        """Create new strategy generation tool"""
        def generate_new_strategies() -> Dict[str, Any]:
            """Generate new strategic approaches for Catanatron"""
            
            # Get current performance to tailor recommendations
            current_metrics = get_current_metrics()
            
            if "error" in current_metrics:
                current_win_rate = 0.0
                strategy_level = "beginner"
            else:
                game_perf = current_metrics.get("game_performance", {})
                current_win_rate = game_perf.get("win_rate", {}).get("current", 0.0)
                
                if current_win_rate > 0.5:
                    strategy_level = "advanced"
                elif current_win_rate > 0.3:
                    strategy_level = "intermediate"
                else:
                    strategy_level = "beginner"
            
            # Generate strategies appropriate for current level
            new_strategies = self._generate_level_appropriate_strategies(strategy_level, current_win_rate)
            immediate_improvements = self._generate_immediate_improvements(strategy_level, current_win_rate)
            
            return {
                "new_strategy_concepts": new_strategies,
                "immediate_improvements": immediate_improvements,
                "long_term_strategic_goals": self._generate_long_term_goals(strategy_level),
                "implementation_roadmap": self._generate_implementation_roadmap(strategy_level)
            }
        return generate_new_strategies
    
    def _generate_level_appropriate_strategies(self, level: str, win_rate: float) -> list:
        """Generate strategies appropriate for current skill level"""
        if level == "beginner":
            return [
                {
                    "name": "Basic Resource Focus",
                    "description": "Focus on consistent resource production and basic building",
                    "implementation": {
                        "phase_1": "Prioritize high-yield resource positions",
                        "phase_2": "Build cities on best resource spots",
                        "phase_3": "Maintain balanced resource collection"
                    },
                    "expected_impact": "establish competitive baseline",
                    "risk_level": "low",
                    "complexity": "low"
                },
                {
                    "name": "Settlement Optimization",
                    "description": "Improve settlement placement strategy",
                    "implementation": {
                        "phase_1": "Analyze board for optimal positions",
                        "phase_2": "Prioritize diverse resource access",
                        "phase_3": "Block opponent expansion when possible"
                    },
                    "expected_impact": "10-15% performance improvement",
                    "risk_level": "low",
                    "complexity": "medium"
                }
            ]
        elif level == "intermediate":
            return [
                {
                    "name": "Adaptive Trading Strategy",
                    "description": "Implement intelligent trading based on resource needs",
                    "implementation": {
                        "phase_1": "Evaluate trade value accurately",
                        "phase_2": "Identify beneficial trading partners",
                        "phase_3": "Use trading to optimize resource flow"
                    },
                    "expected_impact": "15-20% resource efficiency improvement",
                    "risk_level": "medium",
                    "complexity": "high"
                },
                {
                    "name": "Development Card Optimization",
                    "description": "Strategic development card acquisition and timing",
                    "implementation": {
                        "phase_1": "Prioritize development cards when beneficial",
                        "phase_2": "Time card usage for maximum impact",
                        "phase_3": "Integrate cards into overall strategy"
                    },
                    "expected_impact": "10-15% strategic advantage",
                    "risk_level": "medium",
                    "complexity": "medium"
                }
            ]
        else:  # advanced
            return [
                {
                    "name": "Multi-Path Victory Strategy",
                    "description": "Dynamically pursue multiple victory conditions",
                    "implementation": {
                        "phase_1": "Evaluate all victory paths simultaneously",
                        "phase_2": "Maintain flexibility between paths",
                        "phase_3": "Optimize final push timing"
                    },
                    "expected_impact": "20-25% improvement in close games",
                    "risk_level": "high",
                    "complexity": "very_high"
                },
                {
                    "name": "Opponent Behavior Modeling",
                    "description": "Track and predict opponent strategies",
                    "implementation": {
                        "phase_1": "Implement opponent tracking system",
                        "phase_2": "Predict opponent moves",
                        "phase_3": "Counter opponent strategies"
                    },
                    "expected_impact": "15-20% win rate improvement",
                    "risk_level": "medium",
                    "complexity": "very_high"
                }
            ]
    
    def _generate_immediate_improvements(self, level: str, win_rate: float) -> list:
        """Generate immediate actionable improvements"""
        improvements = []
        
        if level == "beginner":
            improvements.extend([
                {
                    "category": "basic_strategy",
                    "improvement": "Implement consistent settlement placement logic",
                    "effort": "low",
                    "expected_gain": "establish baseline performance"
                },
                {
                    "category": "resource_management",
                    "improvement": "Add basic resource prioritization",
                    "effort": "medium",
                    "expected_gain": "improve resource efficiency"
                }
            ])
        elif level == "intermediate":
            improvements.extend([
                {
                    "category": "trading",
                    "improvement": "Enhance trade evaluation algorithm",
                    "effort": "medium",
                    "expected_gain": "12% resource efficiency"
                },
                {
                    "category": "endgame",
                    "improvement": "Add victory condition awareness",
                    "effort": "high",
                    "expected_gain": "15% win rate in close games"
                }
            ])
        else:  # advanced
            improvements.extend([
                {
                    "category": "optimization",
                    "improvement": "Implement Monte Carlo move evaluation",
                    "effort": "very_high",
                    "expected_gain": "10-15% strategic advantage"
                },
                {
                    "category": "adaptation",
                    "improvement": "Add opponent strategy recognition",
                    "effort": "high",
                    "expected_gain": "improved counter-play"
                }
            ])
        
        return improvements
    
    def _generate_long_term_goals(self, level: str) -> Dict[str, str]:
        """Generate long-term strategic goals"""
        if level == "beginner":
            return {
                "strategy_foundation": "Establish solid strategic fundamentals",
                "consistency": "Achieve consistent 30%+ win rate",
                "expansion": "Learn advanced building strategies"
            }
        elif level == "intermediate":
            return {
                "strategic_depth": "Develop sophisticated multi-phase strategies",
                "adaptation": "Implement opponent-aware decision making",
                "optimization": "Achieve 50%+ win rate against AI opponents"
            }
        else:
            return {
                "mastery": "Achieve expert-level strategic play",
                "innovation": "Develop novel strategic approaches",
                "dominance": "Consistently outperform human and AI opponents"
            }
    
    def _generate_implementation_roadmap(self, level: str) -> Dict[str, str]:
        """Generate implementation roadmap"""
        if level == "beginner":
            return {
                "week_1": "Implement basic settlement strategy",
                "week_2": "Add resource management logic",
                "week_3": "Develop building prioritization",
                "week_4": "Integrate endgame awareness"
            }
        elif level == "intermediate":
            return {
                "week_1": "Enhance trading evaluation system",
                "week_2": "Implement adaptive strategy selection",
                "week_3": "Add development card optimization",
                "week_4": "Develop robber placement strategy"
            }
        else:
            return {
                "week_1": "Implement advanced move evaluation",
                "week_2": "Add opponent behavior modeling",
                "week_3": "Develop multi-path victory optimization",
                "week_4": "Integrate machine learning components"
            }
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the strategizer agent with given task data"""
        return await self.agent.run(task_data)
    
    def get_agent(self) -> LlmAgent:
        """Get the underlying LLM agent"""
        return self.agent 