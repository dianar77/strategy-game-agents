import os
from datetime import datetime
#from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
import time
import httpx  # Ensure httpx is imported to catch HTTPStatusError
from langchain_aws import ChatBedrockConverse
from base_llm import MistralLLM, BaseLLM, AnthropicLLM, AzureOpenAILLM



class LLM:
    def __init__(self):
        # Initialize the LLM with the desired model and parameters
        # For example, using OpenAI's GPT-3.5-turbo

        # self.llm = AzureChatOpenAI(
        #     model="gpt-4o-mini",
        #     azure_endpoint="https://gpt-amayuelas.openai.azure.com/",
        #     api_version = "2024-12-01-preview"
        # )
        # self.model_name = "gpt-4o-mini"

        #self.llm = AzureOpenAILLM(model_name="gpt-4o")
        #self.llm = MistralLLM(model_name="mistral-large-latest")
        self.llm = AnthropicLLM()

        self.model_name = self.llm.model


        self.save_dir = f"agents/fromScratchLLMStructured_player_v5_M/runs/game_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Set the environment variable to disable tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    
    def query_llm(self, prompt):
        # Use the LLM to generate a response based on the prompt

        # Create a message
        #msg = HumanMessage(content=prompt)

        # Message list
        #messages = [msg]

        # Invoke the model with a list of messages 
        #response = self.llm.invoke(messages).content
        response = self.llm.query(prompt)

        log_path = os.path.join(self.save_dir, f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        with open(log_path, "a") as log_file:
            log_file.write(f"Prompt:\n{prompt}\n\n{'='*40}\n\nResponse:\n{response}")

        return response.strip()


# class LLM:
#     def __init__(self):
#         # Initialize the LLM with the desired model and parameters
#         # For example, using OpenAI's GPT-3.5-turbo
#         # rate_limiter = InMemoryRateLimiter(
#         #     requests_per_second=0.1,    # Adjust based on your API tier
#         #     check_every_n_seconds=0.1,
#         #     max_bucket_size=10        # Allows for burst handling
#         # )
#         self.llm = ChatMistralAI(
#             model="mistral-large-latest",
#             temperature=0,
#             max_retries=2,
#             #rate_limiter=rate_limiter,
#         )
#         self.model_name = "mistral-large-latest"
#         self.save_dir = f"agents/fromScratchLLMStructured_player_v2/runs/game_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

#         # Set the environment variable to disable tracing
#         os.environ["LANGCHAIN_TRACING_V2"] = "false"


    
#     def query_llm(self, prompt):
#         # Use the LLM to generate a response based on the prompt

#         # Create a message
#         msg = HumanMessage(content=prompt)

#         # Message list
#         messages = [msg]

#         # Invoke the model with a list of messages 

#         while True:
#             try:
#                 response = self.llm.invoke(messages).content

#                 log_path = os.path.join(self.save_dir, f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
#                 if not os.path.exists(self.save_dir):
#                     os.makedirs(self.save_dir, exist_ok=True)
#                 with open(log_path, "a") as log_file:
#                     log_file.write(f"Prompt:\n{prompt}\n\n{'='*40}\n\nResponse:\n{response}")

#                 return response.strip()
                
#             except httpx.HTTPStatusError as e:
#                 if e.response.status_code == 429:
#                     #print("Rate limit exceeded. Retrying after a short delay...")
#                     time.sleep(1)  # Add a small delay for rate limiting
#                     continue
#                 else:
#                     return f"LLM query error: {e.response.status_code} - {e.response.text}"
#             except Exception as e:
#                 return f"LLM query error: {e}"
        

# class LLM:
#     def __init__(self):
#         # Initialize the LLM with the desired model and parameters
#         # For example, using OpenAI's GPT-3.5-turbo
#         # rate_limiter = InMemoryRateLimiter(
#         #     requests_per_second=0.1,    # Adjust based on your API tier
#         #     check_every_n_seconds=0.1,
#         #     max_bucket_size=10        # Allows for burst handling
#         # )
#         self.model_name = "claude-3.7"
#         self.llm = ChatBedrockConverse(
#             aws_access_key_id = os.environ["AWS_ACESS_KEY"],
#             aws_secret_access_key = os.environ["AWS_SECRET_KEY"],
#             region_name = "us-east-2",
#             provider = "anthropic",
#             model_id="arn:aws:bedrock:us-east-2:288380904485:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
#         )
#         self.save_dir = f"agents/fromScratchLLMStructured_player_v2/runs/game_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

#         # Set the environment variable to disable tracing
#         os.environ["LANGCHAIN_TRACING_V2"] = "false"


    
#     def query_llm(self, prompt):
#         # Use the LLM to generate a response based on the prompt

#         # Create a message
#         msg = HumanMessage(content=prompt)

#         # Message list
#         messages = [msg]

#         # Invoke the model with a list of messages 

#         while True:
#             try:
#                 response = self.llm.invoke(messages).content

#                 log_path = os.path.join(self.save_dir, f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
#                 if not os.path.exists(self.save_dir):
#                     os.makedirs(self.save_dir, exist_ok=True)
#                 with open(log_path, "a") as log_file:
#                     log_file.write(f"Prompt:\n{prompt}\n\n{'='*40}\n\nResponse:\n{response}")

#                 return response.strip()
                
#             except httpx.HTTPStatusError as e:
#                 if e.response.status_code == 429:
#                     #print("Rate limit exceeded. Retrying after a short delay...")
#                     time.sleep(1)  # Add a small delay for rate limiting
#                     continue
#                 else:
#                     return f"LLM query error: {e.response.status_code} - {e.response.text}"
#             except Exception as e:
#                 return f"LLM query error: {e}"

# class LLM:
#     run_dir = None

#     def __init__(self):
#         self.llm = AzureChatOpenAI(
#             model="gpt-4o",
#             azure_endpoint="https://gpt-amayuelas.openai.azure.com/",
#             api_version="2024-12-01-preview"
#         )
#         self.model_name = "gpt-4o"
#         if LLM.run_dir is None:
#             # Initialize the base runs directory
#             base_dir = os.path.dirname(os.path.abspath(__file__))
#             runs_dir = os.path.join(base_dir, "runs")
#             os.makedirs(runs_dir, exist_ok=True)
#             LLM.run_dir = os.path.join(runs_dir, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
#             os.makedirs(LLM.run_dir, exist_ok=True)

#     def query_llm(self, prompt: str) -> str:
#         """Query the LLM and return its response."""
#         try:
#             msg = HumanMessage(content=prompt)
#             messages = self.llm.invoke([msg])
#             response = "\n".join(m.content for m in messages['messages'])
#             log_path = os.path.join(LLM.run_dir, f"llm_log_{self.model_name}.txt")
#             with open(log_path, "a") as log_file:
#                 log_file.write(f"Prompt:\n{prompt}\n\nResponse:\n{response}\n{'='*40}\n")
#             return response.strip()
#         except Exception as e:
#             print(f"LLM query error: {e}")
#             return ""