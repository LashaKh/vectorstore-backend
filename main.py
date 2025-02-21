from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uuid
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Qdrant Client Setup
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # Change for deployment
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ✅ OpenAI Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# ✅ Ensure Collection Exists
def ensure_collection_exists(collection_name):
    """Creates Qdrant collection if it doesn't exist."""
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    
    if collection_name not in existing_collections:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

# ✅ Process and Vectorize File with Batching for Large Payloads
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), collection_name: str = Form(...)):
    try:
        ensure_collection_exists(collection_name)  # ✅ Ensure collection exists in Qdrant

        # ✅ Stream file to disk to avoid memory overflow
        temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_path, "wb") as f:
            for chunk in iter(lambda: file.file.read(1024 * 1024), b""):  # 1MB chunks
                f.write(chunk)

        # ✅ Load Document
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif file.filename.endswith(".txt"):
            loader = TextLoader(temp_path)
        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        else:
            return {"error": "Unsupported file type"}

        documents = loader.load()
        os.remove(temp_path)  # ✅ Cleanup temp file

        # ✅ Chunk Text (2000 Characters, 150 Overlap)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        # ✅ Parallel Processing for Faster Vectorization
        with ThreadPoolExecutor(max_workers=5) as executor:
            embeddings = list(executor.map(lambda chunk: embedding_model.embed_query(chunk.page_content), chunks))

        # ✅ **Batch Processing to Avoid Qdrant Payload Limit**
        BATCH_SIZE = 500  # Adjust based on performance needs
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_points = [
                PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": chunk.page_content})
                for embedding, chunk in zip(embeddings[i:i+BATCH_SIZE], chunks[i:i+BATCH_SIZE])
            ]
            qdrant_client.upsert(collection_name=collection_name, points=batch_points)

        return {"message": "File processed successfully", "num_chunks": len(chunks)}

    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}
