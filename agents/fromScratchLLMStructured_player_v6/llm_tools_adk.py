import os
from datetime import datetime
import google.generativeai as genai
from google.genai import types
import time
import json


class LLM:
    """
    ADK-compatible LLM class for foo_player.py
    
    This version uses Google Gemini instead of LangChain models.
    It's designed to work with the ADK agent system while maintaining
    the same interface as the original LLM class.
    """
    
    def __init__(self):
        """
        Initialize the LLM with Google Gemini model.
        
        This replaces the LangChain-based initialization with
        Google Gemini configuration for ADK compatibility.
        """
        # Configure Google Gemini API
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Configure the Gemini client
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model_name = "gemini-2.0-flash"
        self.model = genai.GenerativeModel(self.model_name)
        
        # Set up logging directory
        self.save_dir = f"agents/fromScratchLLMStructured_player_v6/runs_adk/game_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        
        # Configuration for generation
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Safety settings to allow game-related content
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

    def query_llm(self, prompt: str) -> str:
        """
        Query the LLM and return its response.
        
        This method maintains the same interface as the original LangChain
        version but uses Google Gemini instead. It includes retry logic
        and logging for consistency.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: The LLM's response, stripped of whitespace
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Generate response using Gemini
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                # Extract the response text
                response_text = response.text if response.text else ""
                
                # Log the interaction
                self._log_interaction(prompt, response_text)
                
                return response_text.strip()
                
            except Exception as e:
                error_msg = str(e)
                
                # Handle rate limiting
                if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    if attempt < max_retries - 1:
                        print(f"Rate limit hit, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        return f"LLM query error: Rate limit exceeded after {max_retries} attempts"
                
                # Handle other API errors
                elif "api" in error_msg.lower() or "auth" in error_msg.lower():
                    return f"LLM query error: API/Authentication issue - {error_msg}"
                
                # Handle content filtering
                elif "safety" in error_msg.lower() or "content" in error_msg.lower():
                    return f"LLM query error: Content filtered - {error_msg}"
                
                # Generic error handling
                else:
                    if attempt < max_retries - 1:
                        print(f"Error occurred, retrying: {error_msg}")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return f"LLM query error: {error_msg}"
        
        return "LLM query error: Maximum retries exceeded"

    def _log_interaction(self, prompt: str, response: str):
        """
        Log the LLM interaction to a file.
        
        This maintains the same logging behavior as the original
        LangChain version for consistency and debugging.
        
        Args:
            prompt (str): The prompt sent to the LLM
            response (str): The response received from the LLM
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(self.save_dir, f"{self.model_name}_{timestamp}.txt")
            
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                log_file.write(f"Prompt:\n{prompt}\n\n")
                log_file.write(f"{'='*40}\n\n")
                log_file.write(f"Response:\n{response}\n\n")
                log_file.write(f"{'='*80}\n\n")
                
        except Exception as e:
            print(f"Warning: Could not log LLM interaction: {e}")

    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            dict: Model information including name and configuration
        """
        return {
            "model_name": self.model_name,
            "generation_config": self.generation_config,
            "save_dir": self.save_dir,
            "api_type": "Google Gemini"
        }

    def test_connection(self) -> bool:
        """
        Test the connection to the LLM service.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            test_response = self.query_llm("Hello, this is a connection test.")
            return len(test_response) > 0 and "error" not in test_response.lower()
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


# Compatibility function for easy switching
def create_llm() -> LLM:
    """
    Factory function to create an LLM instance.
    
    This provides a consistent interface for creating LLM instances
    regardless of whether using LangChain or ADK version.
    
    Returns:
        LLM: Configured LLM instance
    """
    return LLM()


# For backward compatibility with existing code
def get_llm_instance() -> LLM:
    """
    Get a singleton LLM instance.
    
    This can be useful for sharing LLM instances across multiple
    parts of the application to avoid repeated initialization.
    
    Returns:
        LLM: Shared LLM instance
    """
    if not hasattr(get_llm_instance, '_instance'):
        get_llm_instance._instance = LLM()
    return get_llm_instance._instance


if __name__ == "__main__":
    """
    Test script to verify the ADK LLM functionality.
    
    This can be run directly to test the LLM configuration:
    python llm_tools_adk.py
    """
    print("Testing ADK LLM Tools...")
    
    try:
        llm = LLM()
        print(f"✓ LLM initialized successfully")
        print(f"✓ Model: {llm.model_name}")
        print(f"✓ Save directory: {llm.save_dir}")
        
        # Test connection
        if llm.test_connection():
            print("✓ Connection test passed")
        else:
            print("✗ Connection test failed")
            
        # Test a simple query
        print("\nTesting simple query...")
        response = llm.query_llm("What is 2+2?")
        print(f"Response: {response}")
        
        if "4" in response:
            print("✓ Simple query test passed")
        else:
            print("✗ Simple query test failed")
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        exit(1)
    
    print("\nADK LLM Tools test completed successfully!") 