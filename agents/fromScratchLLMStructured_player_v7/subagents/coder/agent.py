"""
Coder Agent - Implements code changes and technical solutions
"""

from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool
from typing import Dict, Any


class CoderAgent:
    """
    Code Implementation Agent.
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
            You are a Code Implementation Agent.
            Your role is to:
            1. Translate strategic plans into concrete code implementations
            2. Write clean, efficient, and well-documented code
            3. Follow best practices and design patterns
            4. Ensure code quality and maintainability
            5. Implement proper error handling and testing
            
            When writing code:
            - Use clear variable and function names
            - Add comprehensive docstrings and comments
            - Follow SOLID principles and clean code practices
            - Include error handling and edge cases
            - Write testable, modular code
            - Consider performance and scalability
            
            Always provide working, tested code solutions with explanations.
            """,
            tools=[self._create_code_analysis_tool()]
        )
    
    def _create_code_analysis_tool(self):
        """Create code analysis tool"""
        def analyze_code(code: str) -> Dict[str, Any]:
            """Analyze code quality and structure"""
            # This would connect to actual code analysis tools
            return {
                "quality_score": 0.78,
                "complexity_score": 0.82,
                "maintainability_index": 85,
                "issues": [
                    "function complexity too high in process_data()",
                    "missing unit tests for core functionality",
                    "potential memory leak in loop iteration"
                ],
                "suggestions": [
                    "refactor large functions into smaller units",
                    "add comprehensive unit test coverage",
                    "implement proper resource cleanup",
                    "add type hints for better code clarity"
                ],
                "strengths": [
                    "good error handling",
                    "clear documentation",
                    "consistent naming conventions"
                ]
            }
        return analyze_code
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the coder agent with given task data"""
        return await self.agent.run(task_data)
    
    def get_agent(self) -> LlmAgent:
        """Get the underlying LLM agent"""
        return self.agent 