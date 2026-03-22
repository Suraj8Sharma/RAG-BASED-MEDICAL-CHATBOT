from dotenv import load_dotenv
import os 
from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
# 1. Load your .env file
load_dotenv()

# 2. Safely retrieve the key
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# 3. Check if it's missing to avoid the "str | None" error
if pinecone_api_key is None:
    raise ValueError("PINECONE_API_KEY is not set in your .env file or the file wasn't found.")

# 4. Set the environment variable as a string
os.environ["PINECONE_API_KEY"] = pinecone_api_key

extracted_data=load_pdf_files(data="/data/Medical_book.pdf")
filter_data=filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

embeddings=download_embeddings()

pc=Pinecone(api_key=pinecone_api_key)


index_name="medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,#embedding function dimensions 
        metric="cosine",#for cosine similarity
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )
#pointing to the index in the PINECONE
index=pc.Index(index_name)
# 5. Initialize the Vector Store
vs = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)