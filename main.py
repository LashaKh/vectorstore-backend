from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os
import asyncio
import concurrent.futures
import tempfile

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

# Ensure Qdrant Collection Exists
def ensure_collection_exists(collection_name):
    collections = qdrant_client.get_collections()
    collection_names = [c.name for c in collections.collections]

    if collection_name not in collection_names:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

# Process Embeddings Asynchronously (Parallel Processing)
async def embed_text_chunks(chunks):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        vectors = await loop.run_in_executor(pool, embedding_model.embed_documents, [chunk.page_content for chunk in chunks])
    return vectors

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), collection_name: str = Form(...)):
    try:
        # Ensure Collection Exists
        ensure_collection_exists(collection_name)

        # ✅ Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(file.filename)[-1]) as temp_file:
            temp_file.write(await file.read())  # Read file into temp
            temp_file.flush()  # Ensure data is written to disk

            # ✅ Load document based on file type
            if file.filename.endswith(".pdf"):
                loader = PyPDFLoader(temp_file.name)
            elif file.filename.endswith(".txt"):
                loader = TextLoader(temp_file.name)
            elif file.filename.endswith(".docx"):
                loader = Docx2txtLoader(temp_file.name)
            else:
                return {"error": "Unsupported file format. Only PDF, TXT, and DOCX are supported."}

            # Extract text content
            documents = loader.load()

        # ✅ Optimized Chunking: 2500-character chunks with 100-character overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # ✅ Process Embeddings in Parallel
        vectors = await embed_text_chunks(chunks)

        # ✅ Prepare Qdrant Payload
        payloads = [{"text": chunk.page_content} for chunk in chunks]
        points = [models.PointStruct(id=i, vector=vectors[i], payload=payloads[i]) for i in range(len(chunks))]

        # ✅ Batch Insertions into Qdrant
        qdrant_client.upsert(collection_name=collection_name, points=points)

        return {"message": f"File '{file.filename}' uploaded and vectorized successfully!"}

    except Exception as e:
        return {"error": str(e)}
