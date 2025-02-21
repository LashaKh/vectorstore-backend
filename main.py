import os
import uvicorn
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# Initialize Text Splitter (1500-character chunks, 150-character overlap)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)


@app.get("/")
async def home():
    """ Root endpoint to check if the API is running """
    return {"message": "🚀 Qdrant Vector Store API is running!"}


@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form(...)
):
    """ Uploads, chunks, and vectorizes documents before storing in Qdrant """

    try:
        # 🔍 Check if collection exists
        existing_collections = qdrant_client.get_collections()
        existing_collection_names = {col.name for col in existing_collections.collections}

        if collection_name not in existing_collection_names:
            # 🛠️ Create the collection if it does not exist
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )

        # 🔄 Save file to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 📝 Load file using LangChain loaders
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        elif file.filename.endswith(".txt"):
            loader = TextLoader(temp_file_path)
        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
        else:
            return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

        docs = loader.load()

        # 🔢 Chunk text before vectorizing
        all_chunks = []
        for doc in docs:
            chunks = text_splitter.split_text(doc.page_content)
            all_chunks.extend(chunks)

        # 🔢 Convert each chunk into a vector
        points = []
        for i, chunk in enumerate(all_chunks):
            embedding = embedding_model.embed_query(chunk)
            points.append(
                models.PointStruct(id=i, vector=embedding, payload={"text": chunk})
            )

        # 🔄 Insert into Qdrant
        qdrant_client.upsert(collection_name=collection_name, points=points)

        # 🗑️ Cleanup temp file
        os.remove(temp_file_path)

        return {
            "message": f"✅ File processed and stored in `{collection_name}`",
            "chunks_stored": len(all_chunks),
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
