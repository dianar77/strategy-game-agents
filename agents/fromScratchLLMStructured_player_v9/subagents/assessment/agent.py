"""
Assessment Agent - Evaluates evolution progress and controls loop termination
"""

from google.adk.agents import LlmAgent
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory to path to import shared_tools
sys.path.append(str(Path(__file__).parent.parent.parent))
from evolution_tools import assess_evolution_progress, exit_evolution_loop


class AssessmentAgent:
    """
    Evolution Assessment Agent for controlling iteration loops.
    Evaluates evolution results and determines when to terminate loops.
    """
    
    def __init__(self, model):
        self.model = model
        self.agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        """Create the assessment LLM agent"""
        return LlmAgent(
            name="EvolutionAssessor",
            model=self.model,
            instruction="""You are an Evolution Assessment Agent responsible for evaluating 
            the progress of an iterative evolution process and determining when to terminate the loop.

            ## YOUR TASK
            Analyze the evolution result from the current iteration and decide whether to:
            1. Continue with more iterations, OR
            2. Terminate the loop because goals have been sufficiently met

            ## EVALUATION PROCESS
            1. Use assess_evolution_progress tool to analyze the current result
            2. Consider the iteration number and cumulative progress
            3. Evaluate these criteria for termination:
               - **Goal Achievement**: Has the primary improvement goal been met?
               - **Quality Threshold**: Is the implementation sufficiently robust?
               - **Diminishing Returns**: Are further iterations unlikely to yield significant improvements?
               - **Iteration Limit**: Are we approaching reasonable computational limits?
               - **Error Patterns**: Are we stuck in repetitive failure cycles?

            ## TERMINATION CONDITIONS
            Call exit_evolution_loop if ANY of these conditions are met:
            - Implementation demonstrates clear achievement of stated goals
            - Code quality and performance meet high standards  
            - Last 2-3 iterations show minimal meaningful improvement
            - Iteration count reaches 7 or higher (computational efficiency)
            - Persistent errors indicate architectural issues requiring redesign

            ## CONTINUATION CONDITIONS  
            Continue if:
            - Clear improvement trajectory is evident
            - Recent iterations show meaningful progress
            - Implementation has obvious gaps or issues
            - Iteration count is below 5 and progress is being made

            ## INPUT DATA
            **Current Evolution Result:**
            {evolution_result}
            
            **Current Iteration:** {iteration_number}
            
            **Target System:** {target_system}
            
            **Improvement Goal:** {improvement_goal}

            ## OUTPUT INSTRUCTIONS
            - If continuing: Provide specific next improvement goal and rationale
            - If terminating: Call exit_evolution_loop with clear reasoning
            - Be decisive and practical in your assessment

            Always start responses with 'ASSESSMENT:' and end with 'END ASSESSMENT'.
            """,
            description="Assesses evolution progress and controls loop termination based on improvement criteria",
            tools=[assess_evolution_progress, exit_evolution_loop],
            output_key="assessment_decision",
        )
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the assessment agent with the provided task data"""
        return await self.agent.run(task_data)
    
    def get_agent(self) -> LlmAgent:
        """Get the underlying LLM agent"""
        return self.agent 