from dataclasses import dataclass
from orion.config import config

@dataclass
class ModelInfo:
    model_name: str
    description: str
    provider: str
    api_url: str = None
    api_key: str = None

DEFAULT_SUPPORTED_MODELS_LIST = [
    ModelInfo(
        "gpt-4o", 
        "Large model, Great for most questions and fairly accurate. Offers a very high level of intelligence and strong performance, with higher cost per token.",
        "OpenAI",
        api_key=config.OPENAI_API_KEY,
    ),
    ModelInfo(
        "gpt-4o-mini",
        "Faster and good enough for most questions than gpt-4o but less accurate. Offers intelligence not quite on the level of the larger model, but it's faster and less expensive per token.",
        "OpenAI",
    ),
    ModelInfo(
        "gemini-2.0-flash",
        "Faster, better and more accurate than gpt-4o, and may be cheaper as well, but google model and less tested",
        "Google",
        api_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=config.GEMINI_API_KEY,
    ),
    ModelInfo(
        "o1",
        "Advanced reasoning model, better than gpt-4o and gemini-2.0-flash, is slower to return a result, and uses more tokens to \"think,\" but is capable of advanced reasoning, coding, and multi-step planning.",
        "OpenAI",
        api_key=config.OPENAI_API_KEY,
    ),
    ModelInfo(
        "o1-mini",
        "Faster and good enough for most questions than o1 but less accurate. Is also capable of advanced reasoning, coding, and multi-step planning, but is faster and less expensive per token.",
        "OpenAI",
        api_key=config.OPENAI_API_KEY,
    ),
]

DEFAULT_SUPPORTED_MODELS = {model.model_name: model for model in DEFAULT_SUPPORTED_MODELS_LIST}