#taking the necessary import 
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

#---------------------------
#1)
#---------------------------
def load_pdf_files(data):
    loader=PyPDFLoader(data)
    documents=loader.load()
    return documents

#---------------------------
#2)
#only the page_content and the source is important 
#----------------------------
def filter_to_minimal_docs(docs:List[Document])->List[Document]:
    minimal_docs:List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
            page_content=doc.page_content,
            metadata={"source":src}
        ))
    return minimal_docs

#---------------------------------
#3)
#-------------------------------- 
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts_chunks=text_splitter.split_documents(minimal_docs)
    return texts_chunks

#----------------------------------4
#4)
#----------------------------------

def download_embeddings():
    # Repo ID for the model we discussed (80MB)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
  

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        
    )
    return embeddings