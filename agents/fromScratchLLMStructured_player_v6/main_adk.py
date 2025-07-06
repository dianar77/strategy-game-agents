#!/usr/bin/env python3
"""
Main entry point for the ADK-based Creator Agent.
This replaces the LangGraph implementation with Google ADK.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .creator_agent import CreatorAgent


def main():
    """Main function to run the ADK Creator Agent."""
    
    print("Starting ADK-based Catanatron Creator Agent...")
    print("=" * 50)
    
    # Check for required environment variables
    required_env_vars = [
        'GOOGLE_API_KEY',  # For Google Gemini API
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following environment variables:")
        for var in missing_vars:
            if var == 'GOOGLE_API_KEY':
                print(f"  {var}: Get this from Google AI Studio (https://aistudio.google.com/app/apikey)")
        print("\nExample:")
        print(f"  export GOOGLE_API_KEY='your-api-key-here'")
        return 1
    
    try:
        # Create and run the ADK creator agent
        creator = CreatorAgent()
        creator.run_evolution_cycle()
        
        print("\n" + "=" * 50)
        print("ADK Creator Agent completed successfully!")
        print(f"Results saved in: {creator.run_dir}")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nEvolution interrupted by user.")
        return 1
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 