import os
from agents.base_llm import AzureOpenAILLM, OpenRouterLLM

class LLM:
    def __init__(self):
        # Try different LLM providers in order of preference
        self.base_llm = None
        self.use_real_llm = False
        
        # Try OpenRouter first (has free models)
        try:
            if os.getenv("OPENROUTER_API_KEY"):
                self.base_llm = OpenRouterLLM(model_name="deepseek/deepseek-r1:free")
                self.use_real_llm = True
                print("Using OpenRouter LLM (deepseek-r1 free model)")
                return
        except Exception as e:
            print(f"OpenRouter setup failed: {e}")
        
        # Try Azure OpenAI as fallback
        try:
            if os.getenv("AZURE_OPENAI_API_KEY"):
                self.base_llm = AzureOpenAILLM()
                self.use_real_llm = True
                print("Using Azure OpenAI LLM")
                return
        except Exception as e:
            print(f"Azure OpenAI setup failed: {e}")
        
        # No API keys available, use mock
        print("Warning: No LLM API keys found. Using mock LLM responses.")
        print("To use real LLM, set either:")
        print("  - OPENROUTER_API_KEY (free models available)")
        print("  - AZURE_OPENAI_API_KEY")

    def query_llm(self, prompt):
        """Query the LLM with a prompt and return the response."""
        if self.use_real_llm:
            try:
                response = self.base_llm.query(prompt)
                return response
            except Exception as e:
                print(f"Error querying LLM: {e}")
                return "Error: Could not query LLM"
        else:
            # Mock response for testing
            return "Mock LLM response: I would recommend taking the first available action for now." 