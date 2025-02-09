from orion.utils import logger
from dataclasses import dataclass
from dotenv import load_dotenv
import os

@dataclass
class Config:
    GEMINI_API_KEY: str
    OPENAI_API_KEY: str
    
def load_config():
    load_dotenv()
    config = Config(
        GEMINI_API_KEY=os.getenv("GEMINI_API_KEY"),
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    )
    return config
    
config = load_config()