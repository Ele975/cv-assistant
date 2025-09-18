from tools import make_tool_retriever, make_tool_search_engine
from langchain.agents import initialize_agent, AgentType
from models import get_llm

def build_agent(retriever):
    tools = [
        make_tool_retriever(retriever),
        make_tool_search_engine()]
    
    llm = get_llm()

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True
    )
    return agent