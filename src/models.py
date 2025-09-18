import getpass
import os
import sys
from dotenv import load_dotenv

from langsmith import Client

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
import torch
from transformers import pipeline
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.embeddings import HuggingFaceEmbeddings

# load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    print("Failed import from .env file.")
    sys.exit(1)

# set up langsmith

# redundant, but inserted to for double safety


def get_langsmith():
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("Missing LangSmith key.")
        sys.exit(1)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    client = Client()
    project_name = "cv-assistant"
    session = client.create_project(project_name=project_name)
    print(f"LangSmith activated with project name {project_name}.")


def get_llm_summarization():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",          # M1 acceleration
        torch_dtype=torch.float16   # efficient on Apple silicon
    )  

    summarizer = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.0,
    do_sample=False
    )

    return HuggingFacePipeline(pipeline=summarizer)
    
def get_llm():
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN.")
    
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=token
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.5
    )

    return HuggingFacePipeline(pipeline=generator)


def get_retriever():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name
    )

    return embedding_model


def get_search_engine():
    if not os.environ.get("TAVILY_API_KEY"):
        print("Missing Tavily key.")
        sys.exit(1)

    search_engine = TavilySearchResults(max_results=5, topic="general")
    return search_engine
