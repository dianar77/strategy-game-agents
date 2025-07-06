"""
Configuration settings for the Agent Evolver system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # External API configuration (can be customized as needed)
    EXTERNAL_API_BASE_URL = os.getenv('EXTERNAL_API_BASE_URL', 'http://localhost:8000')
    EXTERNAL_API_KEY = os.getenv('EXTERNAL_API_KEY', 'default-key')
    EXTERNAL_API_TIMEOUT = int(os.getenv('EXTERNAL_API_TIMEOUT', '30'))
    
    # Ollama configuration
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:8b') 