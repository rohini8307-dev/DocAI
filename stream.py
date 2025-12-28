import streamlit as st
import requests
import json

FASTAPI_URL = "http://127.0.0.1:8000"

st.title("Doc.AI")

# ---------------- File Upload Section ----------------
st.header("Upload a File")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "doc"])

if st.button("Upload File"):
    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
        response = requests.post(f"{FASTAPI_URL}/upload/", files=files)

        if response.status_code == 200:
            st.success("File uploaded and processed successfully!")
        else:
            try:
                st.error(f"Error: {response.json().get('detail')}")
            except:
                st.error("Error: Unable to parse response from server.")
    else:
        st.warning("Please upload a file before clicking upload.")

# ---------------- Ask Question Section ----------------
st.header("Ask a Question")
question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if question:
        # Append instructions to the question
        question_text = (
            question + 
            ". Do not answer if the content is not present in the documents. "
            "Say the question is irrelevant to the document. Don't say anything other than that specific line."
        )
        
        # Call the FastAPI endpoint
        response = requests.post(f"{FASTAPI_URL}/ask/?question={question_text}")
        
        if response.status_code == 200:
            # Get the answer safely
            answer_data_raw = response.json().get('answer', {})
            
            # If it's a JSON string, parse it
            try:
                answer_data = json.loads(answer_data_raw)
            except (TypeError, json.JSONDecodeError):
                # Already a dict or plain string
                answer_data = answer_data_raw

            # Extract result and source documents safely
            if isinstance(answer_data, dict):
                result = answer_data.get('result', '')
                source_documents = answer_data.get('source_documents', [])
            else:
                result = str(answer_data)
                source_documents = []

            # Display the answer
            st.subheader("Response")
            st.success(result)

            # Display source documents if answer is relevant
            if "irrelevant" not in result.lower() and source_documents:
                st.subheader("Source Documents")
                for doc in source_documents:
                    page_number = doc.get("metadata", {}).get("page", "Unknown")
                    content = doc.get("page_content", "")
                    source_file = doc.get("metadata", {}).get("source", "Unknown")
                    
                    st.subheader(f"Page {page_number}")
                    st.write(content)
                    st.subheader(f"📜 File: {source_file}")
        else:
            try:
                st.error(f"Error: {response.json().get('detail')}")
            except:
                st.error("Error: Unable to parse response from server.")
    else:
        st.warning("Please enter a question before clicking Get Answer.")
