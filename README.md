# DocAI
## ABSTRACT

DocAI is an AI-powered document-based chatbot that allows users to upload documents and ask questions based on the content of those documents. The system supports uploading multiple documents at the same time and enables users to query information across all uploaded files.

Once the documents are uploaded, the text content is extracted and split into smaller, meaningful chunks. Each chunk is converted into a vector embedding using a Sentence Transformer model, which captures the semantic meaning of the text. These embeddings are stored in a vector database for efficient similarity-based retrieval.

When a user submits a question, the question is processed in the same way and converted into a vector embedding using the same Sentence Transformer model. A semantic similarity search is then performed between the question embedding and the stored document embeddings to identify the top k relevant document chunks.

The retrieved document content is provided to a Large Language Model (LLM) to generate the final response. This approach, known as Retrieval-Augmented Generation (RAG), ensures that the answers are accurate, context-aware, and generated strictly from the uploaded documents, avoiding LLM hallucination.

## PIPELINE

1. User Uploads Document
    - Users upload documents via the Streamlit frontend.
    - The files are sent to the FastAPI backend for processing.

2. Document Processing
    - Text is extracted and split into manageable chunks.
    - Sentence Transformers convert each chunk into vector embeddings.

3. Vector Storage
    - Embeddings are stored in ChromaDB for fast and efficient semantic retrieval.

4. Query Processing
    - Users submit a natural language question.
    - FastAPI converts the question into an embedding and retrieves the most relevant chunks from ChromaDB using semantic search.

5. LLM Response Generation
    - Retrieved chunks are passed to Mistral-7B (or another LLM).
    - Retrieval-Augmented Generation (RAG) ensures that the answer is based strictly on the uploaded content, avoiding hallucinations.

6. Answer Sent Back to User
    - The generated response is returned to the Streamlit UI for display.

## TECHNICAL ARCHITECTURE
  
   <img width="1585" height="211" alt="┌───────────────┐ │ User Upload │ │ Documents │ └───────┬───────┘ │ ▼ ┌───────────────┐ │ Text Extraction│ │   Chunking │ └───────┬───────┘ │ ▼ ┌───────────────┐ │ Sentence │ │ Transformers │ │ Em" src="https://github.com/user-attachments/assets/8df828f3-e24f-47c9-bdb7-fb1c2c516106" />

## TECH STACK
    - Python
    - FastAPI
    - Streamlit
    - Sentence Transformer from HuggingFace (MiniLM-L6-v2)
    - ChromaDB
    - Mistral 
    - Ollama
    - LangChain

## DEMO

<img width="1810" height="884" alt="Screenshot 2025-12-26 100310" src="https://github.com/user-attachments/assets/2800f6b6-94bc-4d12-8856-ac118b40d052" />
<img width="1866" height="889" alt="Screenshot 2025-12-26 100330" src="https://github.com/user-attachments/assets/6de33376-c2da-4f45-a710-385270c362d3" />
<img width="1829" height="891" alt="Screenshot 2025-12-26 100253" src="https://github.com/user-attachments/assets/df352c34-3aa1-4839-b388-0baa25af78f0" />

