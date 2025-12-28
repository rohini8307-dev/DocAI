import os
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException

from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

app = FastAPI()

# ---------------- LLM (OLLAMA) ----------------
llm = Ollama(
    model="mistral:latest",
    temperature=0.2
)

# ---------------- GLOBAL STORAGE ----------------
vector_stores = {}
latest_uploaded_file = None

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
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# ---------------- EMBEDDINGS + CHROMA ----------------
def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
    )
    return Chroma.from_documents(chunks, embeddings)

# ---------------- QUESTION ANSWERING (RAG) ----------------
def ask_question(question: str, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
        You are a document-based assistant.
        Answer ONLY using the provided context.

        Context:
        {context}

        Question:
        {input}
        """
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    response = rag_chain.invoke({"input": question})
    return response["answer"]

# ---------------- FILE UPLOAD ----------------
@app.post("/upload/")
async def upload_file(file: UploadFile):
    global latest_uploaded_file

    try:
        os.makedirs("files", exist_ok=True)
        path = f"files/{file.filename}"

        with open(path, "wb") as f:
            f.write(file.file.read())

        docs = load_document(path)
        chunks = chunk_data(docs)
        vector_store = create_embeddings(chunks)

        vector_stores[file.filename] = vector_store
        latest_uploaded_file = file.filename

        return {"message": "File processed successfully"}

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
    return {"answer": answer}

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
