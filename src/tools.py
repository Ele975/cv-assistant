from langchain.tools import Tool
from retriever import get_retriever
from models import get_search_engine

def make_tool_retriever(retriever):
    retriever_tool = Tool(
        name = "CVRetriever"
        func = lambda q: "\n\n".join([d.page_content for d in retriever.get_relevant_documents(q)]),
        description = "Use this tool to answer questions from the uploaded document."
    )
    return retriever_tool


def make_tool_search_engine():
    search_engine = get_search_engine()
    tavily_search_tool = Tool (
        name ="TavilySearch",
        func = lambda q: search_engine.run(q),
        description = "Use this tool to search the web for up-to-date or external information."
    )
    return tavily_search_tool
