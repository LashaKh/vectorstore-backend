from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI App
app = FastAPI()

# Enable CORS for external API access (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Load OpenAI Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# Function to ensure collection exists in Qdrant
def ensure_collection_exists(collection_name):
    collections = qdrant_client.get_collections()
    collection_names = [c.name for c in collections.collections]

    if collection_name not in collection_names:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)  # Ensure correct embedding size
        )

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), collection_name: str = Form(...)):
    try:
        # Ensure collection exists before proceeding
        ensure_collection_exists(collection_name)

        # Save file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Determine file type and load content
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            return {"error": "Unsupported file format. Only PDF, TXT, and DOCX are supported."}

        # Load text from document
        documents = loader.load()

        # Chunk text into 2500-character pieces with 150-character overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        # Convert text chunks into vector embeddings
        vectors = [
            embedding_model.embed_documents([chunk.page_content])[0] for chunk in chunks
        ]

        # Prepare points for Qdrant
        payloads = [{"text": chunk.page_content} for chunk in chunks]
        points = [
            models.PointStruct(id=i, vector=vectors[i], payload=payloads[i])
            for i in range(len(chunks))
        ]

        # Upload vectors to Qdrant
        qdrant_client.upsert(collection_name=collection_name, points=points)

        return {"message": f"File '{file.filename}' uploaded and vectorized successfully!"}

    except Exception as e:
        return {"error": str(e)}
