"""
Evolver Agent - Core evolution logic using Chain-of-Thought reasoning
"""

from google.adk.agents import LlmAgent
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add parent directory to path to import shared_tools
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared_tools import (
    analyze_performance_trends,
    get_current_metrics,
    read_full_performance_history,
    read_game_results_file,
    read_older_foo_file,
    run_testfoo,
    CURRENT_EVOLUTION
)


class EvolverAgent:
    """
    Core Evolution Agent for Catanatron players using Chain-of-Thought reasoning.
    Coordinates the evolution process across all agents.
    """
    
    def __init__(self, model):
        self.model = model
        self.agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        """Create the evolver LLM agent"""
        return LlmAgent(
            name="evolver",
            model=self.model,
            instruction="""
            You are the Core Evolution Agent for Catanatron game players using Chain-of-Thought reasoning.
            Your role is to:
            1. Coordinate the evolution process across all agents
            2. Use step-by-step reasoning to guide system improvements
            3. Make decisions about which improvements to implement
            4. Ensure coherent evolution strategy across iterations
            5. Balance multiple objectives and constraints
            
            Chain-of-Thought Process:
            - Step 1: Analyze current Catanatron player performance and game results
            - Step 2: Identify key problems and strategic opportunities
            - Step 3: Evaluate potential solutions and their trade-offs
            - Step 4: Develop comprehensive improvement plan
            - Step 5: Prioritize actions based on impact and feasibility
            - Step 6: Define success criteria and monitoring strategy
            
            Think through each step carefully and explain your reasoning.
            Use the following agents: Strategizer, Coder, Researcher, Analyzer.
            
            Always provide structured evolution plans with clear rationale.
            """,
            tools=[self._create_coordination_tool(), self._create_decision_tool(), self._create_evolution_tracking_tool()]
        )
    
    def _create_coordination_tool(self):
        """Create coordination tool for Catanatron agent orchestration"""
        def coordinate_catanatron_agents(task: str, agents: List[str]) -> Dict[str, Any]:
            """Coordinate multiple agents for Catanatron evolution tasks"""
            
            # Get real performance data
            current_metrics = get_current_metrics()
            performance_trends = analyze_performance_trends()
            
            # Extract real data or use defaults if error
            if "error" in current_metrics:
                win_rate = 0.15  # Default for new agent
                avg_score = 6.0
                evolution_cycle = 0
                weaknesses = ["initialization", "basic_strategy"]
                improvements = ["initial_setup"]
            else:
                game_perf = current_metrics.get("game_performance", {})
                win_rate = game_perf.get("win_rate", {}).get("current", 0.15)
                avg_score = game_perf.get("average_score", {}).get("current", 6.0)
                evolution_cycle = game_perf.get("evolution_cycle", 0)
                
                # Determine weaknesses based on performance
                weaknesses = []
                if win_rate < 0.3:
                    weaknesses.extend(["basic_strategy", "resource_management"])
                if win_rate < 0.5:
                    weaknesses.extend(["endgame_strategy", "trading_optimization"])
                if avg_score < 7.0:
                    weaknesses.append("scoring_efficiency")
                
                improvements = ["settlement_placement", "resource_management"] if evolution_cycle > 0 else ["initial_setup"]
            
            return {
                "coordination_id": f"catanatron_coord_{hash(task) % 10000}",
                "task": task,
                "assigned_agents": agents,
                "coordination_plan": {
                    "execution_strategy": self._determine_execution_strategy(task, agents),
                    "dependencies": self._analyze_catanatron_dependencies(agents),
                    "resource_allocation": self._allocate_catanatron_resources(agents),
                    "timeline": self._estimate_catanatron_timeline(task, agents),
                    "success_criteria": self._define_catanatron_success_criteria(task)
                },
                "catanatron_context": {
                    "current_win_rate": win_rate,
                    "target_win_rate": 0.60,
                    "current_avg_score": avg_score,
                    "main_weaknesses": weaknesses,
                    "recent_improvements": improvements,
                    "evolution_cycle": evolution_cycle,
                    "games_since_last_change": 50  # Standard test size
                },
                "risk_assessment": {
                    "coordination_risks": [
                        "strategy conflicts between agents",
                        "code integration issues",
                        "performance regression"
                    ],
                    "mitigation_strategies": [
                        "staged implementation with rollback",
                        "comprehensive testing after each change",
                        "incremental improvements rather than radical changes"
                    ],
                    "monitoring_points": [
                        "game win rate",
                        "performance metrics", 
                        "error rates",
                        "strategic coherence"
                    ]
                }
            }
        return coordinate_catanatron_agents
    
    def _create_decision_tool(self):
        """Create decision-making tool for Catanatron evolution choices"""
        def make_catanatron_evolution_decision(options: List[Dict[str, Any]], criteria: Dict[str, float]) -> Dict[str, Any]:
            """Make informed decisions about Catanatron evolution options"""
            
            # Get real performance data
            current_metrics = get_current_metrics()
            performance_trends = analyze_performance_trends()
            
            # Extract current performance or use defaults
            if "error" in current_metrics:
                current_performance = {
                    "win_rate": 0.15,
                    "average_score": 6.0,
                    "strategic_weaknesses": ["basic_strategy"],
                    "technical_issues": ["initialization"]
                }
                improvement_priorities = {
                    "high": ["basic_implementation"],
                    "medium": ["strategy_development"],
                    "low": ["optimization"]
                }
            else:
                game_perf = current_metrics.get("game_performance", {})
                current_performance = {
                    "win_rate": game_perf.get("win_rate", {}).get("current", 0.15),
                    "average_score": game_perf.get("average_score", {}).get("current", 6.0),
                    "strategic_weaknesses": self._identify_strategic_weaknesses(current_metrics),
                    "technical_issues": self._identify_technical_issues(performance_trends)
                }
                improvement_priorities = self._determine_improvement_priorities(current_performance)
            
            return {
                "decision_id": f"catanatron_decision_{hash(str(options)) % 10000}",
                "evolution_context": {
                    "current_performance": current_performance,
                    "improvement_priorities": improvement_priorities
                },
                "evaluation_criteria": criteria,
                "option_analysis": [
                    {
                        "option": option,
                        "catanatron_score": self._calculate_catanatron_score(option, criteria),
                        "strategic_impact": self._assess_strategic_impact(option),
                        "implementation_complexity": self._assess_implementation_complexity(option),
                        "risk_level": self._assess_catanatron_risk(option),
                        "expected_win_rate_improvement": self._estimate_win_rate_improvement(option)
                    }
                    for option in options
                ],
                "recommended_option": self._select_best_catanatron_option(options, criteria),
                "rationale": self._generate_decision_rationale(options, criteria),
                "confidence_level": self._calculate_confidence_level(options),
                "implementation_plan": self._create_implementation_plan(options),
                "success_metrics": {
                    "primary": "win_rate_improvement >= 0.05",
                    "secondary": ["reduced_error_rate", "improved_strategic_coherence"],
                    "timeline": "evaluate_after_50_games"
                }
            }
        return make_catanatron_evolution_decision
    
    def _create_evolution_tracking_tool(self):
        """Create tool for tracking Catanatron evolution progress"""
        def track_catanatron_evolution(cycle_data: Dict[str, Any]) -> Dict[str, Any]:
            """Track and analyze Catanatron evolution progress"""
            
            # Get real performance history
            performance_trends = analyze_performance_trends()
            
            if "error" in performance_trends:
                # No data available yet
                performance_history = {
                    "cycle_0": {"win_rate": 0.0, "avg_score": 0.0, "status": "no_data"}
                }
                improvement_trends = {
                    "win_rate_trend": "no_data",
                    "strategic_coherence": "unknown",
                    "error_reduction": "unknown",
                    "code_quality": "unknown"
                }
                implemented_strategies = ["none"]
                pending_strategies = ["basic_implementation", "strategy_development"]
                strategy_effectiveness = {}
                bugs_fixed = []
                remaining_issues = ["system_initialization"]
            else:
                # Build performance history from real data
                trends_data = performance_trends.get("trends", {})
                latest_evolutions = trends_data.get("latest_evolutions", [])
                
                performance_history = {}
                for i, evolution in enumerate(latest_evolutions):
                    cycle_key = f"cycle_{evolution['number']}"
                    performance_history[cycle_key] = {
                        "win_rate": evolution["wins"] / 3.0,  # Assuming 3 games per test
                        "avg_score": evolution["avg_score"],
                        "status": "completed"
                    }
                
                # Determine trends
                win_rates = [e["wins"] / 3.0 for e in latest_evolutions]
                if len(win_rates) > 1:
                    if win_rates[-1] > win_rates[0]:
                        win_trend = "positive, improving"
                    elif win_rates[-1] < win_rates[0]:
                        win_trend = "negative, declining"
                    else:
                        win_trend = "stable"
                else:
                    win_trend = "insufficient_data"
                
                improvement_trends = {
                    "win_rate_trend": win_trend,
                    "strategic_coherence": "improving" if trends_data.get("win_rate_trend") == "improving" else "needs_work",
                    "error_reduction": "steady_improvement",
                    "code_quality": "steady_improvement"
                }
                
                # Infer strategies based on performance
                current_win_rate = performance_trends.get("current_performance", {}).get("win_rate", 0)
                if current_win_rate > 0.4:
                    implemented_strategies = ["settlement_placement", "resource_management", "basic_strategy"]
                    pending_strategies = ["endgame_optimization", "advanced_trading"]
                elif current_win_rate > 0.2:
                    implemented_strategies = ["basic_strategy", "settlement_placement"]
                    pending_strategies = ["resource_management", "development_cards", "endgame_optimization"]
                else:
                    implemented_strategies = ["basic_implementation"]
                    pending_strategies = ["basic_strategy", "settlement_placement", "resource_management"]
                
                strategy_effectiveness = {strategy: min(0.85, current_win_rate + 0.3) for strategy in implemented_strategies}
                bugs_fixed = ["basic_implementation_errors"]
                remaining_issues = ["performance_optimization", "advanced_strategy_implementation"]
            
            return {
                "evolution_tracking": {
                    "current_cycle": cycle_data.get("cycle", CURRENT_EVOLUTION),
                    "performance_history": performance_history,
                    "improvement_trends": improvement_trends
                },
                "strategic_evolution": {
                    "implemented_strategies": implemented_strategies,
                    "pending_strategies": pending_strategies,
                    "strategy_effectiveness": strategy_effectiveness
                },
                "technical_evolution": {
                    "bugs_fixed": bugs_fixed,
                    "performance_improvements": ["system_optimization"],
                    "code_quality_improvements": ["error_handling", "documentation"],
                    "remaining_issues": remaining_issues
                },
                "next_cycle_recommendations": {
                    "priority_focus": pending_strategies[0] if pending_strategies else "optimization",
                    "secondary_focus": pending_strategies[1] if len(pending_strategies) > 1 else "testing",
                    "technical_focus": "performance_optimization",
                    "expected_impact": "10-15% win rate improvement"
                }
            }
        return track_catanatron_evolution
    
    def _determine_execution_strategy(self, task: str, agents: List[str]) -> str:
        """Determine optimal execution strategy for Catanatron tasks"""
        if "analysis" in task.lower():
            return "analyzer_first"
        elif "strategy" in task.lower():
            return "strategizer_led"
        elif "code" in task.lower():
            return "coder_focused"
        elif len(agents) > 2:
            return "parallel_with_coordination"
        else:
            return "sequential_with_feedback"
    
    def _analyze_catanatron_dependencies(self, agents: List[str]) -> Dict[str, List[str]]:
        """Analyze dependencies between Catanatron agents"""
        dependencies = {
            "analyzer": [],  # Can start independently
            "strategizer": ["analyzer"],  # Needs analysis first
            "researcher": [],  # Can start independently
            "coder": ["strategizer", "researcher"],  # Needs strategy and research
            "player": ["coder"]  # Needs code implementation
        }
        return {agent: dependencies.get(agent, []) for agent in agents}
    
    def _allocate_catanatron_resources(self, agents: List[str]) -> Dict[str, Dict[str, Any]]:
        """Allocate resources to Catanatron agents"""
        resource_map = {
            "analyzer": {"priority": "high", "timeout": "120s", "memory": "256MB"},
            "strategizer": {"priority": "high", "timeout": "180s", "memory": "512MB"},
            "researcher": {"priority": "medium", "timeout": "300s", "memory": "256MB"},
            "coder": {"priority": "high", "timeout": "240s", "memory": "512MB"},
            "player": {"priority": "medium", "timeout": "60s", "memory": "128MB"}
        }
        return {agent: resource_map.get(agent, {"priority": "normal", "timeout": "120s", "memory": "256MB"}) for agent in agents}
    
    def _estimate_catanatron_timeline(self, task: str, agents: List[str]) -> Dict[str, str]:
        """Estimate timeline for Catanatron evolution tasks"""
        base_times = {
            "analyzer": 60,
            "strategizer": 120,
            "researcher": 180,
            "coder": 240,
            "player": 30
        }
        
        total_time = sum(base_times.get(agent, 60) for agent in agents)
        
        return {
            "estimated_duration": f"{total_time}s",
            "phases": {
                "analysis": "60s",
                "strategy_development": "120s",
                "research": "180s",
                "implementation": "240s",
                "testing": "60s"
            },
            "parallel_optimization": f"{max(base_times.get(agent, 60) for agent in agents)}s"
        }
    
    def _define_catanatron_success_criteria(self, task: str) -> List[str]:
        """Define success criteria for Catanatron evolution tasks"""
        return [
            "win_rate_improvement >= 5%",
            "no_critical_bugs_introduced",
            "strategic_coherence_maintained",
            "performance_regression < 10%",
            "code_quality_improved"
        ]
    
    def _calculate_catanatron_score(self, option: Dict[str, Any], criteria: Dict[str, float]) -> float:
        """Calculate Catanatron-specific score for an option"""
        # Simulate scoring based on Catanatron-specific criteria
        base_score = 0.7
        
        if "endgame" in str(option).lower():
            base_score += 0.2
        if "trading" in str(option).lower():
            base_score += 0.15
        if "robber" in str(option).lower():
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _assess_strategic_impact(self, option: Dict[str, Any]) -> str:
        """Assess strategic impact of an option"""
        return "high" if "strategy" in str(option).lower() else "medium"
    
    def _assess_implementation_complexity(self, option: Dict[str, Any]) -> str:
        """Assess implementation complexity"""
        return "medium"  # Default complexity
    
    def _assess_catanatron_risk(self, option: Dict[str, Any]) -> str:
        """Assess risk level for Catanatron evolution"""
        return "medium"  # Default risk level
    
    def _estimate_win_rate_improvement(self, option: Dict[str, Any]) -> float:
        """Estimate potential win rate improvement"""
        return 0.08  # Default 8% improvement estimate
    
    def _select_best_catanatron_option(self, options: List[Dict[str, Any]], criteria: Dict[str, float]) -> Dict[str, Any]:
        """Select the best option for Catanatron evolution"""
        return max(options, key=lambda x: self._calculate_catanatron_score(x, criteria))
    
    def _generate_decision_rationale(self, options: List[Dict[str, Any]], criteria: Dict[str, float]) -> str:
        """Generate rationale for decision"""
        return "Selected based on highest potential impact on win rate and strategic coherence"
    
    def _calculate_confidence_level(self, options: List[Dict[str, Any]]) -> float:
        """Calculate confidence level in decision"""
        return 0.85  # Default confidence level
    
    def _create_implementation_plan(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create implementation plan for selected option"""
        return {
            "phase_1": "analysis_and_planning",
            "phase_2": "strategy_development",
            "phase_3": "code_implementation",
            "phase_4": "testing_and_validation",
            "rollback_plan": "maintain_previous_version"
        }
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the evolver agent with given task data"""
        return await self.agent.run(task_data)
    
    def get_agent(self) -> LlmAgent:
        """Get the underlying LLM agent"""
        return self.agent 

    def _identify_strategic_weaknesses(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify strategic weaknesses from performance metrics"""
        weaknesses = []
        game_perf = metrics.get("game_performance", {})
        win_rate = game_perf.get("win_rate", {}).get("current", 0)
        avg_score = game_perf.get("average_score", {}).get("current", 0)
        
        if win_rate < 0.3:
            weaknesses.extend(["basic_strategy", "resource_management"])
        if win_rate < 0.5:
            weaknesses.extend(["endgame_strategy", "trading_optimization"])
        if avg_score < 7.0:
            weaknesses.append("scoring_efficiency")
        
        return weaknesses or ["general_strategy"]
    
    def _identify_technical_issues(self, trends: Dict[str, Any]) -> List[str]:
        """Identify technical issues from performance trends"""
        if "error" in trends:
            return ["system_initialization", "data_access"]
        return ["timeout_handling", "exception_management"]
    
    def _determine_improvement_priorities(self, performance: Dict[str, Any]) -> Dict[str, List[str]]:
        """Determine improvement priorities based on performance"""
        win_rate = performance.get("win_rate", 0)
        
        if win_rate < 0.2:
            return {
                "high": ["basic_strategy_implementation", "critical_bug_fixes"],
                "medium": ["settlement_placement", "resource_management"],
                "low": ["optimization", "advanced_features"]
            }
        elif win_rate < 0.4:
            return {
                "high": ["endgame_strategy", "trading_optimization"],
                "medium": ["robber_strategy", "development_cards"],
                "low": ["code_cleanup", "documentation"]
            }
        else:
            return {
                "high": ["advanced_strategy", "performance_optimization"],
                "medium": ["opponent_modeling", "adaptive_strategy"],
                "low": ["code_refactoring", "testing"]
            } 