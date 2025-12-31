# DocAI
##ABSTRACT
DocAI is an AI-powered document-based chatbot that allows users to upload documents and ask questions based on the content of those documents. The system supports uploading multiple documents at the same time and enables users to query information across all uploaded files.

Once the documents are uploaded, the text content is extracted and split into smaller, meaningful chunks. Each chunk is converted into a vector embedding using a Sentence Transformer model, which captures the semantic meaning of the text. These embeddings are stored in a vector database for efficient similarity-based retrieval.

When a user submits a question, the question is processed in the same way and converted into a vector embedding using the same Sentence Transformer model. A semantic similarity search is then performed between the question embedding and the stored document embeddings to identify the top k relevant document chunks.

The retrieved document content is provided to a Large Language Model (LLM) to generate the final response. This approach, known as Retrieval-Augmented Generation (RAG), ensures that the answers are accurate, context-aware, and generated strictly from the uploaded documents, avoiding LLM hallucination.
