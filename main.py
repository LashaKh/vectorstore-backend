from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# 🌍 **Enable CORS (Cross-Origin Resource Sharing)**
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://silly-madeleine-17b9cd.netlify.app"],  # ✅ Update with your Netlify frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # ✅ Allows all headers (Content-Type, Authorization, etc.)
)

# ✅ Load environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Initialize Qdrant Client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ✅ Define Text Chunking Config
CHUNK_SIZE = 1500
OVERLAP = 150

# ✅ File Upload and Vectorization
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), collection_name: str = Form(...)):
    try:
        # Save the file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load document based on file type
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            return {"error": "Unsupported file type"}

        docs = loader.load()

        # ✅ Chunk the text into 1500-character chunks with 150-character overlap
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
        chunks = text_splitter.split_documents(docs)

        # ✅ Convert chunks into embeddings
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
        vectors = embedding_model.embed_documents([chunk.page_content for chunk in chunks])

        # ✅ Store vectors in Qdrant
        points = [{"id": i, "vector": vec, "payload": {"text": chunks[i].page_content}} for i, vec in enumerate(vectors)]
        qdrant_client.upsert(collection_name=collection_name, points=points)

        # ✅ Clean up temporary file
        os.remove(file_path)

        return {"message": "File processed and vectorized successfully", "chunks": len(chunks)}

    except Exception as e:
        return {"error": str(e)}

