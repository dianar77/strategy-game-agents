"""
Evolution Tools for Loop Agent

This module provides tools for controlling the evolution loop process.
"""

from typing import Any, Dict
from google.adk.tools.tool_context import ToolContext


def assess_evolution_progress(result: str, iteration: int, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to assess evolution progress and determine if goals have been met.
    
    Args:
        result: The evolution result to assess
        iteration: Current iteration number
        tool_context: Context for accessing and updating session state
        
    Returns:
        Dict[str, Any]: Assessment results including recommendations
    """
    print(f"\n----------- EVOLUTION ASSESSMENT (Iteration {iteration}) -----------")
    print(f"Assessing evolution result...")
    print("----------------------------------------------------------------\n")
    
    # Store assessment data in session state
    tool_context.state[f"iteration_{iteration}_result"] = result
    tool_context.state["last_iteration"] = iteration
    
    # Simple assessment criteria - in practice this could be more sophisticated
    assessment = {
        "iteration": iteration,
        "result_length": len(result) if isinstance(result, str) else 0,
        "has_implementation": "implementation" in str(result).lower(),
        "has_assessment": "assessment" in str(result).lower(),
        "status": "evaluated"
    }
    
    return assessment


def exit_evolution_loop(reason: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Call this function when the evolution process should be terminated.
    This can happen when goals are met, sufficient improvements achieved,
    or when continuing would not be beneficial.
    
    Args:
        reason: Explanation for why the loop should exit
        tool_context: Context for tool execution
        
    Returns:
        Dict with termination details
    """
    print("\n----------- EXIT EVOLUTION LOOP TRIGGERED -----------")
    print(f"Termination reason: {reason}")
    print("Evolution loop will exit now")
    print("----------------------------------------------------\n")
    
    # Set escalate flag to terminate the loop
    tool_context.actions.escalate = True
    
    # Store termination info in state
    tool_context.state["loop_terminated"] = True
    tool_context.state["termination_reason"] = reason
    
    return {
        "terminated": True,
        "reason": reason,
        "message": f"Evolution loop terminated: {reason}"
    } 