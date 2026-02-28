import os
from langchain_openai import ChatOpenAI

def get_llm(streaming: bool = True):
    model_name = os.getenv("OPENROUTER_MODEL")
    url = os.getenv("base_url")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    return ChatOpenAI(
        model=model_name,
        base_url=url,
        api_key=api_key,
        temperature=0.1,
        streaming=streaming,
    )
