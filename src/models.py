import getpass
import os
import sys
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
import torch
from transformers import pipeline, BitsAndBytesConfig
from langchain_tavily import TavilySearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    print("Failed import from .env file.")
    sys.exit(1)


def get_llm_summarization():
    # # model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # model_path = "models_downloaded/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map="auto",
    #     dtype=torch.float16  
    # )  

    # summarizer = pipeline(
    # "text-generation",
    # model=model,
    # tokenizer=tokenizer,
    # max_new_tokens=256,
    # temperature=0.0,
    # do_sample=False
    # )

    # return HuggingFacePipeline(pipeline=summarizer)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OpenAI key.")
        sys.exit(1)

    return ChatOpenAI(
        model = "gpt-4-turbo",
        temperature = 0.0,
        max_tokens = 256
    )
    
def get_llm():
    # token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    # if not token:
    #     raise ValueError("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN.")
    
    # # model_path = "models_downloaded/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a"
    # model_path = "models_downloaded/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"

    # # model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map="auto",
    #     dtype=torch.float16
    # )

    # generator = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=512,
    #     temperature=0.5
    # )

    # return HuggingFacePipeline(pipeline=generator)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OpenAI key.")
        sys.exit(1)

    return ChatOpenAI(
        model = "gpt-4-turbo",
        temperature = 0.0,
        max_tokens = 256
    )

def get_retriever():
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_path = "models_downloaded/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

    embedding_model = HuggingFaceEmbeddings(
        model_name = model_path
    )

    return embedding_model


def get_search_engine():
    if not os.environ.get("TAVILY_API_KEY"):
        print("Missing Tavily key.")
        sys.exit(1)

    search_engine = TavilySearch(max_results=5, topic="general")
    return search_engine
