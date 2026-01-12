import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

app = FastAPI()

# ---------------- LLM (OLLAMA) ----------------
llm = Ollama(
    model="mistral:latest",
    temperature=0.7  # Higher temp = faster inference
)

# ---------------- GLOBAL STORAGE ----------------
vector_stores = {}
latest_uploaded_file = None

# ---------------- CACHE EMBEDDINGS MODEL (Load once) ----------------
# Using faster, lightweight model for better performance
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fastest model
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 256}  # Higher batch = faster
)

# ---------------- DOCUMENT LOADING ----------------
def load_document(file_path: str):
    _, ext = os.path.splitext(file_path)

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type")

# ---------------- CHUNKING ----------------
def chunk_data(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# ---------------- EMBEDDINGS + CHROMA (OPTIMIZED) ----------------
def create_embeddings(chunks, filename):
    # Use persistent storage for faster retrieval
    persist_directory = f"./chroma_db/{filename.replace('.', '_')}"
    
    return Chroma.from_documents(
        chunks, 
        embeddings_model,  # Use cached model
        persist_directory=persist_directory
    )

# ---------------- QUESTION ANSWERING (RAG) ----------------
def ask_question(question: str, vector_store):
    # Create retriever with semantic search
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Use RetrievalQA for fast answers
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    
    # Get the response
    response = qa_chain({"query": question})
    return response.get("result", "No answer generated")

# ---------------- FILE UPLOAD ----------------
@app.post("/upload/")
async def upload_file(files: list[UploadFile] = File(...)):
    global latest_uploaded_file

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file = files[0]
        os.makedirs("files", exist_ok=True)
        path = f"files/{file.filename}"

        with open(path, "wb") as f:
            f.write(file.file.read())

        docs = load_document(path)
        print(f"📄 Loaded {len(docs)} pages/sections from {file.filename}")
        
        chunks = chunk_data(docs)
        print(f"✂️ Created {len(chunks)} chunks from document")
        
        vector_store = create_embeddings(chunks, file.filename)
        print(f"✅ Embedded all {len(chunks)} chunks into ChromaDB")

        vector_stores[file.filename] = vector_store
        latest_uploaded_file = file.filename

        return {"message": f"File processed successfully: {len(docs)} pages, {len(chunks)} chunks embedded"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- ASK QUESTION ----------------
@app.post("/ask/")
async def ask(question: str):
    if latest_uploaded_file is None:
        raise HTTPException(status_code=404, detail="No file uploaded")

    vector_store = vector_stores.get(latest_uploaded_file)
    if vector_store is None:
        raise HTTPException(status_code=404, detail="Vector store not found")

    answer = ask_question(question, vector_store)
    return {"result": answer}

# ---------------- ASK QUESTION ACROSS ALL DOCUMENTS ----------------
@app.post("/ask_all/")
async def ask_all(question: str):
    if not vector_stores:
        raise HTTPException(status_code=404, detail="No files uploaded")

    all_results = []
    
    for filename, vector_store in vector_stores.items():
        answer = ask_question(question, vector_store)
        all_results.append({
            "filename": filename,
            "result": answer
        })
    
    return {"results": all_results}

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
