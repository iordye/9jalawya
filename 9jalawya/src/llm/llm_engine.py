from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.logging_config import get_logger
from src.config.settings import Settings
from functools import lru_cache

logger = get_logger(__name__)

settings = Settings()

@lru_cache(maxsize=1)
def model():
    llm = ChatGoogleGenerativeAI(
        model=settings.llm.model,
        google_api_key=settings.llm.google_api_key,
        max_output_tokens=settings.llm.max_output_tokens,
        temperature=settings.llm.temperature,
    )
    return llm