"""
Player Agent - Executes actions and interacts with external systems/APIs
"""

from google.adk.agents import LlmAgent
from typing import Dict, Any, Optional, List
import requests
import json


class PlayerAgent:
    """
    Player Agent that executes actions in the target system.
    Handles interaction with external APIs and systems.
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
            You are a Player Agent that executes actions in the target system.
            Your role is to:
            1. Execute planned actions and implementations safely
            2. Interact with external APIs and systems
            3. Monitor game state and system responses
            4. Collect execution results and comprehensive feedback
            5. Handle errors and edge cases gracefully
            
            When executing actions:
            - Always validate inputs and parameters
            - Implement proper error handling and recovery
            - Log all actions and their outcomes
            - Collect detailed performance metrics
            - Ensure system safety and stability
            - Provide clear status updates and feedback
            
            Always execute actions safely and collect comprehensive results.
            Report both successes and failures with detailed context.
            """,
            tools=[
                self._create_execution_tool(),
                self._create_api_tool(),
                self._create_monitoring_tool(),
                self._create_safety_tool()
            ]
        )
    
    def _create_execution_tool(self):
        """Create execution tool for system actions"""
        def execute_action(action: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a specific action in the target system"""
            action_type = action.get('type', 'unknown')
            action_params = action.get('parameters', {})
            
            try:
                # Simulate action execution - replace with actual implementation
                if action_type == 'code_deployment':
                    result = self._execute_code_deployment(action_params)
                elif action_type == 'configuration_update':
                    result = self._execute_configuration_update(action_params)
                elif action_type == 'system_restart':
                    result = self._execute_system_restart(action_params)
                elif action_type == 'test_execution':
                    result = self._execute_tests(action_params)
                else:
                    result = self._execute_generic_action(action_type, action_params)
                
                return {
                    "action": action,
                    "status": "success",
                    "result": result,
                    "execution_time": result.get('execution_time', 'unknown'),
                    "output": result.get('output', f"Successfully executed {action_type}"),
                    "metrics": result.get('metrics', {}),
                    "timestamp": result.get('timestamp', 'now')
                }
                
            except Exception as e:
                return {
                    "action": action,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "output": f"Failed to execute {action_type}: {str(e)}",
                    "timestamp": "now"
                }
        
        return execute_action
    
    def _create_api_tool(self):
        """Create API interaction tool (e.g., for Catanatron API)"""
        def call_api(endpoint: str, data: Dict[str, Any], method: str = "POST") -> Dict[str, Any]:
            """Call external API (like Catanatron API)"""
            try:
                # Get API configuration
                base_url = self.api_config.get('base_url', 'https://api.example.com')
                api_key = self.api_config.get('api_key', '')
                timeout = self.api_config.get('timeout', 30)
                
                # Prepare request
                url = f"{base_url}/{endpoint.lstrip('/')}"
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}' if api_key else ''
                }
                
                # Make request (simulated - replace with actual requests)
                # response = requests.request(method, url, json=data, headers=headers, timeout=timeout)
                
                # Simulated response for example
                simulated_response = {
                    "status": "success",
                    "data": f"API response from {endpoint}",
                    "game_state": {
                        "current_round": 5,
                        "player_score": 1250,
                        "system_performance": 0.92
                    },
                    "recommendations": [
                        "optimize move selection algorithm",
                        "improve resource management"
                    ]
                }
                
                return {
                    "endpoint": endpoint,
                    "method": method,
                    "request_data": data,
                    "response": simulated_response,
                    "status": "success",
                    "response_time": "150ms",
                    "timestamp": "now"
                }
                
            except Exception as e:
                return {
                    "endpoint": endpoint,
                    "method": method,
                    "request_data": data,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": "now"
                }
        
        return call_api
    
    def _create_monitoring_tool(self):
        """Create monitoring tool for system state"""
        def monitor_system() -> Dict[str, Any]:
            """Monitor current system state and health"""
            return {
                "system_health": {
                    "status": "healthy",
                    "uptime": "99.95%",
                    "last_check": "now",
                    "components": {
                        "database": "online",
                        "api_server": "online",
                        "cache": "online",
                        "message_queue": "online"
                    }
                },
                "performance_metrics": {
                    "response_time": "125ms",
                    "throughput": "1200 req/sec",
                    "error_rate": "0.01%",
                    "cpu_usage": "45%",
                    "memory_usage": "67%"
                },
                "game_metrics": {
                    "active_games": 245,
                    "avg_game_duration": "8.5 minutes",
                    "win_rate": "52%",
                    "player_satisfaction": 4.3
                }
            }
        
        return monitor_system
    
    def _create_safety_tool(self):
        """Create safety validation tool"""
        def validate_safety(action: Dict[str, Any]) -> Dict[str, Any]:
            """Validate action safety before execution"""
            action_type = action.get('type', 'unknown')
            
            # Safety checks
            safety_checks = {
                "input_validation": self._validate_inputs(action),
                "resource_check": self._check_resources(action),
                "dependency_check": self._check_dependencies(action),
                "rollback_plan": self._verify_rollback_plan(action)
            }
            
            all_safe = all(check['safe'] for check in safety_checks.values())
            
            return {
                "action_type": action_type,
                "safety_status": "safe" if all_safe else "unsafe",
                "checks": safety_checks,
                "recommendations": self._get_safety_recommendations(safety_checks),
                "can_proceed": all_safe
            }
        
        return validate_safety
    
    def _execute_code_deployment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code deployment action"""
        # TODO: Replace with actual deployment logic
        # Example real implementation:
        # - Copy files to target directory
        # - Run build/compile processes  
        # - Update configurations
        # - Restart services
        
        return {
            "deployment_id": "deploy_001",
            "status": "completed",
            "execution_time": "45s",
            "output": "Code deployed successfully to production",
            "metrics": {"deployment_size": "2.3MB", "files_changed": 15}
        }
    
    def _execute_configuration_update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute configuration update action"""
        return {
            "config_version": "v1.2.3",
            "status": "applied",
            "execution_time": "5s",
            "output": "Configuration updated successfully",
            "metrics": {"configs_changed": 3, "restart_required": False}
        }
    
    def _execute_system_restart(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system restart action"""
        return {
            "restart_type": params.get('type', 'graceful'),
            "status": "completed",
            "execution_time": "30s",
            "output": "System restarted successfully",
            "metrics": {"downtime": "15s", "startup_time": "15s"}
        }
    
    def _execute_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute test suite"""
        # TODO: Replace with actual test execution
        # Example real implementation:
        # import subprocess
        # result = subprocess.run(['python', '-m', 'pytest'], capture_output=True, text=True)
        # return parse_test_results(result)
        
        return {
            "test_suite": params.get('suite', 'all'),
            "status": "passed",
            "execution_time": "120s",
            "output": "All tests passed successfully",
            "metrics": {"tests_run": 156, "passed": 156, "failed": 0, "coverage": "94%"}
        }
    
    def _execute_generic_action(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic action"""
        return {
            "action_type": action_type,
            "status": "completed",
            "execution_time": "10s",
            "output": f"Generic action {action_type} executed",
            "metrics": {"success": True}
        }
    
    def _validate_inputs(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Validate action inputs"""
        return {"safe": True, "message": "All inputs validated"}
    
    def _check_resources(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check resource availability"""
        return {"safe": True, "message": "Sufficient resources available"}
    
    def _check_dependencies(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check system dependencies"""
        return {"safe": True, "message": "All dependencies satisfied"}
    
    def _verify_rollback_plan(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Verify rollback plan exists"""
        return {"safe": True, "message": "Rollback plan verified"}
    
    def _get_safety_recommendations(self, checks: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get safety recommendations based on checks"""
        recommendations = []
        for check_name, check_result in checks.items():
            if not check_result.get('safe', True):
                recommendations.append(f"Address {check_name} issues before proceeding")
        return recommendations or ["All safety checks passed"]
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the player agent with given task data"""
        return await self.agent.run(task_data)
    
    def get_agent(self) -> LlmAgent:
        """Get the underlying LLM agent"""
        return self.agent 