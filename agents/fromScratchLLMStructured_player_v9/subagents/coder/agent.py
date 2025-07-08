"""
Coder Agent - Implements code changes and technical solutions
"""

from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory to path to import shared_tools
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared_tools import (
    read_foo,
    write_foo,
    read_local_file,
    run_testfoo
)


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
        def analyze_catanatron_code(code: str = None) -> Dict[str, Any]:
            """Analyze Catanatron player code quality and structure"""
            
            # Read current code if not provided
            if code is None:
                code = read_foo()
            
            if "Error" in code:
                return {
                    "error": code,
                    "code_analysis": {
                        "overall_quality": 0.0,
                        "complexity_score": 0.0,
                        "maintainability_index": 0,
                        "catanatron_compliance": 0.0
                    }
                }
            
            # Analyze the actual code
            analysis = self._perform_real_code_analysis(code)
            return analysis
        return analyze_catanatron_code
    
    def _perform_real_code_analysis(self, code: str) -> Dict[str, Any]:
        """Perform actual analysis of the code"""
        lines = code.split('\n')
        total_lines = len(lines)
        
        # Count different types of code elements
        class_count = len([line for line in lines if line.strip().startswith('class ')])
        method_count = len([line for line in lines if line.strip().startswith('def ')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        docstring_lines = len([line for line in lines if '"""' in line or "'''" in line])
        
        # Check for common Catanatron imports and patterns
        has_catanatron_imports = any('catanatron' in line for line in lines)
        has_action_imports = any('Action' in line for line in lines)
        has_player_base = any('Player' in line for line in lines)
        
        # Identify potential issues
        issues = []
        performance_issues = []
        
        # Check for basic implementation issues
        if not has_catanatron_imports:
            issues.append({
                "type": "imports",
                "issue": "Missing Catanatron imports",
                "line": 1,
                "suggestion": "Add required Catanatron imports",
                "severity": "high"
            })
        
        if method_count == 0:
            issues.append({
                "type": "implementation",
                "issue": "No methods defined",
                "line": 1,
                "suggestion": "Implement required player methods",
                "severity": "high"
            })
        
        # Check for specific method implementations
        code_lower = code.lower()
        required_methods = ['decide', 'action', 'play']
        missing_methods = []
        
        for method in required_methods:
            if f'def {method}' not in code_lower:
                missing_methods.append(method)
        
        if missing_methods:
            issues.append({
                "type": "required_methods",
                "issue": f"Missing required methods: {', '.join(missing_methods)}",
                "line": 1,
                "suggestion": "Implement missing required methods",
                "severity": "high"
            })
        
        # Check for performance issues
        if total_lines > 500:
            performance_issues.append({
                "issue": "Large file size may impact performance",
                "impact": "potential slow loading",
                "suggestion": "Consider breaking into smaller modules",
                "line_range": f"1-{total_lines}"
            })
        
        # Calculate quality metrics
        code_complexity = min(1.0, method_count / 10.0)  # Assume 10 methods is reasonably complex
        documentation_ratio = (comment_lines + docstring_lines) / max(total_lines, 1)
        compliance_score = 0.7 if has_catanatron_imports and has_player_base else 0.3
        
        overall_quality = (code_complexity + documentation_ratio + compliance_score) / 3
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if has_catanatron_imports:
            strengths.append("proper Catanatron imports")
        if documentation_ratio > 0.1:
            strengths.append("includes documentation")
        if class_count > 0:
            strengths.append("object-oriented structure")
        
        if not has_action_imports:
            weaknesses.append("missing action type imports")
        if method_count < 3:
            weaknesses.append("insufficient method implementation")
        if documentation_ratio < 0.05:
            weaknesses.append("lacks adequate documentation")
        
        # Generate recommendations
        recommendations = []
        
        if issues:
            recommendations.append({
                "priority": "high",
                "category": "bug_fix",
                "description": "Fix critical implementation issues",
                "implementation": "Address missing imports and methods"
            })
        
        if performance_issues:
            recommendations.append({
                "priority": "medium",
                "category": "performance",
                "description": "Optimize code structure",
                "implementation": "Refactor for better performance"
            })
        
        if documentation_ratio < 0.1:
            recommendations.append({
                "priority": "low",
                "category": "documentation",
                "description": "Add comprehensive documentation",
                "implementation": "Add docstrings and comments"
            })
        
        return {
            "code_analysis": {
                "overall_quality": round(overall_quality, 2),
                "complexity_score": round(code_complexity, 2),
                "maintainability_index": int(overall_quality * 100),
                "catanatron_compliance": round(compliance_score, 2),
                "total_lines": total_lines,
                "methods_count": method_count,
                "classes_count": class_count,
                "documentation_ratio": round(documentation_ratio, 2)
            },
            "catanatron_specific_issues": issues,
            "performance_issues": performance_issues,
            "strategy_implementation": {
                "strengths": strengths,
                "weaknesses": weaknesses
            },
            "code_structure": {
                "has_required_imports": has_catanatron_imports,
                "has_player_class": has_player_base,
                "has_action_handling": has_action_imports,
                "missing_methods": missing_methods
            },
            "recommended_improvements": recommendations
        }
    
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
        strategy_name = spec.get("strategy", "general")
        
        code_templates = {
            "endgame_optimization": '''
def _evaluate_endgame_strategy(self, game_state):
    """Evaluate and choose optimal endgame strategy"""
    my_vp = self._get_victory_points(game_state)
    
    # Simple endgame logic based on current position
    if my_vp >= 8:
        return "rush_to_victory"
    elif my_vp >= 6:
        return "balanced_approach"
    else:
        return "catch_up_strategy"
            ''',
            
            "resource_management": '''
def _optimize_resource_usage(self, game_state):
    """Optimize resource allocation and usage"""
    my_resources = self._get_my_resources(game_state)
    
    # Prioritize based on current needs
    priorities = []
    if my_resources.get("wood", 0) >= 4 and my_resources.get("brick", 0) >= 4:
        priorities.append("build_roads")
    if my_resources.get("wheat", 0) >= 2 and my_resources.get("ore", 0) >= 3:
        priorities.append("build_city")
    
    return priorities
            ''',
            
            "trading_strategy": '''
def _evaluate_trade_opportunity(self, game_state, trade_offer):
    """Evaluate whether a trade opportunity is beneficial"""
    my_resources = self._get_my_resources(game_state)
    
    # Simple trade evaluation
    giving = trade_offer.get("giving", {})
    receiving = trade_offer.get("receiving", {})
    
    # Check if we can afford the trade
    can_afford = all(my_resources.get(resource, 0) >= amount 
                    for resource, amount in giving.items())
    
    return can_afford and len(receiving) > 0
            '''
        }
        
        if strategy_name in code_templates:
            implementation = code_templates[strategy_name]
        else:
            implementation = code_templates["resource_management"]  # Default
        
        return {
            "code_type": "strategy_improvement",
            "strategy_name": strategy_name,
            "implementation": implementation,
            "integration_notes": "Add this method to your player class and call from decide() method",
            "test_suggestion": "Test with run_testfoo() after integration"
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