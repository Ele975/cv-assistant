from tools import make_tool_retriever, make_tool_search_engine
from langchain.agents import create_react_agent, Tool
from models import get_llm

def build_agent(retriever):
    tools = [
        make_tool_retriever(retriever),
        make_tool_search_engine()]
    llm = get_llm()

    agent = create_react_agent(llm, tools)
    return agent