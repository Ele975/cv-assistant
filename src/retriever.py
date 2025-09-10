from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from models import get_retriever

def get_documents(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    return docs

def splitting(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=200, add_start_index = True
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def embedding():
    embeddings = get_retriever()
    # get dimension of sample query
    dim = len(embeddings.embed_query('Hello world.'))
    index = faiss.IndexFlatL2(dim)

    vector_store = FAISS (
        embedding_function = embeddings,
        index = index,
        docstore = InMemoryDocstore(),
        index_to_docstore_id={}
    )

    return vector_store


def store_embeddings(chunks):
