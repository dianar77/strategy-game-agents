import os
from datetime import datetime
from langchain.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage


class LLM:
    def __init__(self):
        # Initialize the LLM with the desired model and parameters
        # For example, using OpenAI's GPT-3.5-turbo

        self.llm = AzureChatOpenAI(
            model="gpt-4o",
            azure_endpoint="https://gpt-amayuelas.openai.azure.com/",
            api_version = "2024-12-01-preview"
        )
        self.model_name = "gpt-4o-mini"
        self.save_dir = f"agents/fromScratchLLM_player_v2/runs/game_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    
    def query_llm(self, prompt):
        # Use the LLM to generate a response based on the prompt

        # Create a message
        msg = HumanMessage(content=prompt)

        # Message list
        messages = [msg]

        # Invoke the model with a list of messages 
        response = self.llm.invoke(messages).content

        log_path = os.path.join(self.save_dir, f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        with open(log_path, "a") as log_file:
            log_file.write(f"Prompt:\n{prompt}\n\n{'='*40}\n\nResponse:\n{response}")

        return response.strip()


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