from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os 
from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests

app=FastAPI()

load_dotenv()
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

# --- LAZY LOADING SETUP START ---
# 1. Create a global variable starting as None
rag_chain = None

# 2. Wrap your exact AI logic in a function
def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        print("Downloading embeddings and initializing models...")
        
        #loading the embedding model
        embeddings=download_embeddings()
        index_name="medical-chatbot"

        #making my retriever
        docsearch=PineconeVectorStore.from_existing_index(
            embedding=embeddings,
            index_name=index_name
        )

        retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

        chatmodel=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        prompt=ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            ("human","{input}"),
        ])

        #making the chains
        questions_answer_chain=create_stuff_documents_chain(chatmodel,prompt)
        rag_chain=create_retrieval_chain(retriever,questions_answer_chain)
        
    return rag_chain
# --- LAZY LOADING SETUP END ---


#defining where to look for the html and css files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request:Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/msg")
def chat(msg:str=Form(...)):
    # 3. Call the function here. It will only download on the very first message!
    chain = get_rag_chain()
    
    response=chain.invoke({"input":msg})
    print("Response : ",response["answer"] )
    return str(response["answer"])