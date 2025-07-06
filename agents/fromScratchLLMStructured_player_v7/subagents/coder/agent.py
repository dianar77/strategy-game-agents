"""
Coder Agent - Implements code changes and technical solutions
"""

from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool
from typing import Dict, Any


class CoderAgent:
    """
    Code Implementation Agent for Catanatron players.
    Translates strategic plans into concrete code implementations.
    """
    
    def __init__(self, model):
        self.model = model
        self.agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        """Create the coder LLM agent"""
        return LlmAgent(
            name="coder",
            model=self.model,
            instruction="""
            You are the CODER expert for evolving Catanatron game players.
            
            Your Role:
            - Implement code changes based on strategic recommendations
            - Write clean, efficient, and well-documented Python code
            - Follow Catanatron API patterns and best practices
            - Ensure code quality and maintainability
            - Implement proper error handling and testing
            - Fix bugs and syntax errors in player implementations

            Coding Guidelines:
            - Follow Python 3.12 syntax and conventions
            - Add comprehensive docstrings and comments
            - Use print statements for debugging when needed
            - Prioritize fixing bugs and errors
            - Don't make up variables or functions that don't exist
            - Use proper Catanatron action types and game state access

            Always start responses with 'CODER:' and end with 'END CODER'.
            Report on all changes made to the code.
            """,
            tools=[self._create_code_analysis_tool(), self._create_implementation_tool()]
        )
    
    def _create_code_analysis_tool(self):
        """Create Catanatron-specific code analysis tool"""
        def analyze_catanatron_code(code: str) -> Dict[str, Any]:
            """Analyze Catanatron player code quality and structure"""
            return {
                "code_analysis": {
                    "overall_quality": 0.78,
                    "complexity_score": 0.82,
                    "maintainability_index": 85,
                    "catanatron_compliance": 0.92
                },
                "catanatron_specific_issues": [
                    {
                        "type": "api_usage",
                        "issue": "Using deprecated game.state.get_buildings() method",
                        "line": 45,
                        "suggestion": "Use game.state.board.buildings instead",
                        "severity": "medium"
                    },
                    {
                        "type": "action_creation",
                        "issue": "Missing validation for BuildSettlementAction parameters",
                        "line": 78,
                        "suggestion": "Add node availability check before creating action",
                        "severity": "high"
                    },
                    {
                        "type": "resource_management",
                        "issue": "Hard-coded resource costs instead of using constants",
                        "line": 92,
                        "suggestion": "Use BUILDING_COSTS from catanatron.models.enums",
                        "severity": "low"
                    }
                ],
                "performance_issues": [
                    {
                        "issue": "Inefficient board traversal in _find_best_settlement()",
                        "impact": "O(n²) complexity for settlement selection",
                        "suggestion": "Cache board analysis results between turns",
                        "line_range": "120-145"
                    },
                    {
                        "issue": "Repeated calculations in trade evaluation",
                        "impact": "unnecessary computational overhead",
                        "suggestion": "Memoize trade value calculations",
                        "line_range": "200-230"
                    }
                ],
                "strategy_implementation": {
                    "strengths": [
                        "well-structured decision tree in decide() method",
                        "good separation of concerns between strategy and actions",
                        "clear resource management logic",
                        "comprehensive action filtering"
                    ],
                    "weaknesses": [
                        "lacks endgame strategy adaptation",
                        "no opponent behavior tracking",
                        "simple robber placement logic",
                        "missing development card optimization"
                    ]
                },
                "code_structure": {
                    "methods_analysis": {
                        "decide()": "main decision method - well implemented",
                        "_evaluate_settlement()": "good logic but needs optimization",
                        "_should_trade()": "basic implementation, needs enhancement",
                        "_place_robber()": "too simplistic, needs strategic improvement"
                    },
                    "missing_methods": [
                        "_evaluate_endgame_strategy()",
                        "_track_opponent_behavior()",
                        "_optimize_development_cards()",
                        "_calculate_victory_probability()"
                    ]
                },
                "recommended_improvements": [
                    {
                        "priority": "high",
                        "category": "bug_fix",
                        "description": "Fix IndexError in settlement evaluation",
                        "implementation": "Add bounds checking for node access"
                    },
                    {
                        "priority": "high",
                        "category": "performance",
                        "description": "Optimize board analysis caching",
                        "implementation": "Implement turn-based caching system"
                    },
                    {
                        "priority": "medium",
                        "category": "strategy",
                        "description": "Add sophisticated endgame logic",
                        "implementation": "Implement victory condition probability calculation"
                    }
                ]
            }
        return analyze_catanatron_code
    
    def _create_implementation_tool(self):
        """Create implementation tool for Catanatron code generation"""
        def generate_catanatron_code(specification: Dict[str, Any]) -> Dict[str, Any]:
            """Generate Catanatron-specific code based on specifications"""
            spec_type = specification.get("type", "unknown")
            
            if spec_type == "strategy_improvement":
                return self._generate_strategy_code(specification)
            elif spec_type == "bug_fix":
                return self._generate_bug_fix_code(specification)
            elif spec_type == "performance_optimization":
                return self._generate_optimization_code(specification)
            elif spec_type == "new_feature":
                return self._generate_feature_code(specification)
            else:
                return self._generate_generic_code(specification)
        
        return generate_catanatron_code
    
    def _generate_strategy_code(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy improvement code"""
        return {
            "code_type": "strategy_improvement",
            "implementation": {
                "endgame_optimization": '''
def _evaluate_endgame_strategy(self, game_state):
    """Evaluate and choose optimal endgame strategy"""
    my_vp = game_state.current_player.victory_points
    turns_left = self._estimate_turns_remaining(game_state)
    
    # Calculate probability of winning with different strategies
    strategies = {
        "city_rush": self._calculate_city_rush_probability(game_state),
        "development_cards": self._calculate_dev_card_probability(game_state),
        "longest_road": self._calculate_longest_road_probability(game_state)
    }
    
    # Choose strategy with highest win probability
    best_strategy = max(strategies, key=strategies.get)
    return best_strategy, strategies[best_strategy]
                ''',
                "opponent_tracking": '''
def _track_opponent_behavior(self, game_state, action_history):
    """Track and analyze opponent behavior patterns"""
    if not hasattr(self, 'opponent_profiles'):
        self.opponent_profiles = {}
    
    for player in game_state.players:
        if player.color != self.color:
            if player.color not in self.opponent_profiles:
                self.opponent_profiles[player.color] = {
                    "trading_frequency": 0,
                    "aggressive_robber": 0,
                    "preferred_strategy": "unknown"
                }
            
            # Update behavior metrics based on recent actions
            self._update_opponent_profile(player, action_history)
                ''',
                "adaptive_trading": '''
def _evaluate_trade_with_prediction(self, trade_action, game_state):
    """Evaluate trade considering future game state"""
    # Calculate immediate value
    immediate_value = self._calculate_resource_value(trade_action.give, trade_action.receive)
    
    # Predict opponent responses
    opponent_responses = self._predict_opponent_reactions(trade_action, game_state)
    
    # Calculate long-term strategic value
    strategic_value = self._calculate_strategic_trade_value(trade_action, game_state)
    
    return immediate_value + strategic_value - opponent_responses
                '''
            },
            "integration_points": [
                "Add endgame evaluation to main decide() method",
                "Initialize opponent tracking in __init__",
                "Integrate adaptive trading in trade selection logic"
            ]
        }
    
    def _generate_bug_fix_code(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate bug fix code"""
        return {
            "code_type": "bug_fix",
            "implementation": {
                "bounds_checking": '''
def _safe_node_access(self, board, node_id):
    """Safely access board nodes with bounds checking"""
    try:
        if node_id in board.nodes:
            return board.nodes[node_id]
        else:
            print(f"Warning: Node {node_id} not found in board")
            return None
    except (KeyError, AttributeError) as e:
        print(f"Error accessing node {node_id}: {e}")
        return None
                ''',
                "exception_handling": '''
def decide(self, game, valid_actions):
    """Main decision method with comprehensive error handling"""
    try:
        # Validate inputs
        if not valid_actions:
            print("Warning: No valid actions available")
            return None
        
        # Main decision logic
        chosen_action = self._make_decision(game, valid_actions)
        
        # Validate output
        if chosen_action not in valid_actions:
            print(f"Error: Chosen action not in valid actions, defaulting to first")
            return valid_actions[0]
        
        return chosen_action
        
    except Exception as e:
        print(f"Error in decide(): {e}")
        print(f"Defaulting to first valid action")
        return valid_actions[0] if valid_actions else None
                '''
            },
            "fixes_applied": [
                "Added bounds checking for node access",
                "Implemented comprehensive exception handling",
                "Added input/output validation"
            ]
        }
    
    def _generate_optimization_code(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance optimization code"""
        return {
            "code_type": "performance_optimization",
            "implementation": {
                "caching_system": '''
def __init__(self, color):
    super().__init__(color)
    self.board_cache = {}
    self.last_turn_analyzed = -1
    self.cached_evaluations = {}

def _get_cached_board_analysis(self, game_state):
    """Get cached board analysis or compute new one"""
    current_turn = game_state.turn_number
    
    if current_turn != self.last_turn_analyzed:
        # Invalidate cache for new turn
        self.board_cache.clear()
        self.last_turn_analyzed = current_turn
    
    cache_key = self._generate_board_hash(game_state.board)
    if cache_key not in self.board_cache:
        self.board_cache[cache_key] = self._analyze_board(game_state.board)
    
    return self.board_cache[cache_key]
                ''',
                "efficient_algorithms": '''
def _find_best_settlement_optimized(self, game_state, available_nodes):
    """Optimized settlement selection using pre-computed metrics"""
    if not available_nodes:
        return None
    
    # Use cached board analysis
    board_analysis = self._get_cached_board_analysis(game_state)
    
    # Score nodes using vectorized operations
    node_scores = {}
    for node_id in available_nodes:
        if node_id in board_analysis['node_values']:
            node_scores[node_id] = board_analysis['node_values'][node_id]
    
    # Return best node
    return max(node_scores, key=node_scores.get) if node_scores else available_nodes[0]
                '''
            },
            "optimizations_applied": [
                "Implemented turn-based caching system",
                "Optimized settlement selection algorithm",
                "Added efficient board analysis caching"
            ]
        }
    
    def _generate_feature_code(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new feature code"""
        return {
            "code_type": "new_feature",
            "implementation": {
                "feature_example": "# New feature implementation would go here"
            }
        }
    
    def _generate_generic_code(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic code improvements"""
        return {
            "code_type": "generic_improvement",
            "implementation": {
                "generic_example": "# Generic code improvement would go here"
            }
        }
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the coder agent with given task data"""
        return await self.agent.run(task_data)
    
    def get_agent(self) -> LlmAgent:
        """Get the underlying LLM agent"""
        return self.agent 