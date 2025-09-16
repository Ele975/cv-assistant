import faiss
from uuid import uuid4

from models import get_retriever

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_documents(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    # return Document objects
    return docs

def splitting(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=200, add_start_index = True
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def get_empty_vector_store():
    # this entire implementation can be replaced with 'vector_store = FAISS.from_documents(chunks, embeddings)'
    # not here because we need the vector store without for now passing any doc
    embedding_model = get_retriever()
    # get dimension of sample query
    dim = len(embedding_model.embed_query('Hello world.'))
    index = faiss.IndexFlatL2(dim)

    vector_store = FAISS (
        embedding_function = embedding_model,
        index = index,
        docstore = InMemoryDocstore(),
        index_to_docstore_id={}
    )

    return vector_store


def build_vector_store_from_file(filepath):
    docs = get_documents(filepath)
    chunks = splitting(docs)
    vector_store = get_empty_vector_store()
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids = uuids)

    return vector_store


def build_retriever_from_cv(filepath):
    vector_store = build_vector_store_from_file(filepath)
    return vector_store.as_retriever()
    