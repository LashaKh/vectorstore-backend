import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
import uuid
import time

app = FastAPI()

# 🏎️ Use a FASTER embedding model (2x speed boost)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key="YOUR_OPENAI_API_KEY")

# 🚀 Optimize Qdrant Config
QDRANT_COLLECTION_SIZE = 1536  # Vector size
QDRANT_DISTANCE = Distance.COSINE  # Best for similarity search
BATCH_SIZE = 100  # Increase batch size for faster inserts

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), collection_name: str = Form(...)):
    """Fast vectorization with parallel processing & optimized chunking."""
    start_time = time.time()  # ⏱️ Start timing
    
    try:
        # 🔥 Step 1: Read & Load File
        content = await file.read()
        file_extension = file.filename.split(".")[-1].lower()
        
        if file_extension == "pdf":
            loader = PyPDFLoader(file.file)
        elif file_extension == "txt":
            loader = TextLoader(file.file)
        elif file_extension == "docx":
            loader = Docx2txtLoader(file.file)
        else:
            return {"error": "Unsupported file type"}
        
        documents = loader.load()

        # 🔥 Step 2: Optimize Text Chunking (2200 chars for balance)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2200, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # 🔥 Step 3: Ensure Collection Exists in Qdrant
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=QDRANT_COLLECTION_SIZE, distance=QDRANT_DISTANCE),
        )

        # 🔥 Step 4: Use Async Parallel Processing for Faster Embeddings
        async def process_batch(batch):
            texts = [chunk.page_content for chunk in batch]
            embeddings = await asyncio.to_thread(embedding_model.embed_documents, texts)

            points = [
                PointStruct(
                    id=uuid.uuid4().int & (1 << 64) - 1,
                    vector=embedding,
                    payload={"text": batch[j].page_content}
                )
                for j, embedding in enumerate(embeddings)
            ]

            qdrant_client.upsert(collection_name=collection_name, points=points)

        # 🚀 Process Chunks in Parallel (Split into batches)
        tasks = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            tasks.append(process_batch(batch))

        await asyncio.gather(*tasks)  # Run all batches at once 🔥

        # ⏱️ Calculate Time Taken
        total_time = time.time() - start_time
        return {"message": "File vectorized & stored!", "processing_time_sec": total_time}

    except Exception as e:
        return {"error": str(e)}
