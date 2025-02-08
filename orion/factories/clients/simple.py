from typing import List, Dict, Union
from orion.llm_clients import OpenAIClient
from orion.config import config
from dataclasses import dataclass

class AutoClientFactory:
    def __init__(self):
        pass
    
    def create(self, task_description: str):
        # Choose a model based on the task description
        pass
    
DEFAULT_CLIENT_FACTORY = AutoClientFactory()