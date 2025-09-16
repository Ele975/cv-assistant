import gradio as gr
from uuid import uuid4

from agent import build_agent
from retriever import build_retriever_from_cv
from models import get_llm_summarization

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    CombinedMemory
)

def get_hybrid_memory():
    # use llm to summarize older conversation
    llm = get_llm_summarization()
    # keep last 5 messages
    window_memory =ConversationBufferWindowMemory (
        k = 5,
        return_messages = True
    )
    #summarize old messages
    summary_memory = ConversationSummaryMemory(
        llm = llm,
        return_messages = True
    )

    #combine memories
    hybrid_memory = CombinedMemory(memories = [window_memory, summary_memory])
    return hybrid_memory



def get_session_history(session_id : str):
    return get_hybrid_memory()

def setup_agent(cv_file):
    retriever = build_retriever_from_cv(cv_file)
    agent = build_agent(retriever)
