import getpass
import os
import sys
from dotenv import load_dotenv

from langsmith import Client

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

# load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    print("Failed import from .env file.")
    sys.exit(1)

# set up langsmith

# redundant, but inserted to for double safety


def get_langsmith():
    if not os.environ["LANGSMITH_API_KEY"]:
        print("Missing LangSmith key.")
        sys.exit(1)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    client = Client()
    project_name = "cv-assistant"
    session = client.create_project(project_name=project_name)
    print(f"LangSmith activated with project name {project_name}.")


def get_llm_summarization():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OpenAI key.")
        sys.exit(1)
    # smaller model for summarization for long-term memory
    model = "gpt-3.5-turbo"
    llm = ChatOpenAI(model=model, temperature=0, max_tokens=None)
    return llm   
    
def get_llm():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OpenAI key.")
        sys.exit(1)

    # large model for main agent for reasoning
    model = "gpt-4o"
    llm = ChatOpenAI(model=model, temperature=0.5, max_tokens=None)
    return llm


def get_retriever():
    model = "text-embedding-3-large"
    embedding_model = OpenAIEmbeddings(model=model)
    return embedding_model


def get_search_engine():
    if not os.environ.get["TAVILY_API_KEY"]:
        print("Missing Tavily key.")
        sys.exit(1)

    search_engine = TavilySearchResults(max_results=5, topic="general")
    return search_engine
