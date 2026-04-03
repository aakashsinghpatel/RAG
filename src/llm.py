from langchain_ollama import ChatOllama
import config


def load_llm():
    llm = ChatOllama(
        model=config.LLM_MODEL,
        temperature=0.1
    )
    return llm