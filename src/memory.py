import config
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

_memory_store = {}

def get_memory(session_id: str = "default"):
    if session_id not in _memory_store:
        _memory_store[session_id] = InMemoryChatMessageHistory()
    return _memory_store[session_id]

def add_to_memory(session_id: str, role: str, message: str):
    memory = get_memory(session_id)
    # Use new message objects instead of raw role/message
    if role.lower() == "user":
        memory.add_message(HumanMessage(content=message))
    elif role.lower() == "ai":
        memory.add_message(AIMessage(content=message))
    else:
        raise ValueError(f"Unknown role '{role}' for memory.")
     # Keep only last N messages
    MAX_MESSAGES = config.MEMORY_K
    if len(memory.messages) > MAX_MESSAGES:
        memory.messages = memory.messages[-MAX_MESSAGES:]

def get_history(session_id: str):
    memory = get_memory(session_id)
    return memory.messages  # list of BaseMessage objects