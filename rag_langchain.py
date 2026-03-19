from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

from langchain.schema import Document
import os

def load_documents(folder_path="data"):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".md"):
            path = os.path.join(folder_path, file)
            print("Loading:", path)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append(Document(page_content=text))
            except Exception as e:
                print("Error loading file:", path)
                print(e)

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 3})