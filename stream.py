import streamlit as st
import requests
import json

FASTAPI_URL = "http://127.0.0.1:8000"

st.title("Doc.AI ")
st.caption("Fast RAG-powered document Q&A")

st.header("Upload a File")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "doc"])

if st.button("Upload File", type="primary"):
    if uploaded_file:
        with st.spinner("Processing document... This may take a moment"):
            files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
            response = requests.post(f"{FASTAPI_URL}/upload/", files=files, timeout=120)

            if response.status_code == 200:
                st.success("File uploaded and processed successfully!")
            else:
                try:
                    st.error(f"Error: {response.json().get('detail')}")
                except:
                    st.error("Error: Unable to parse response from server.")
    else:
        st.warning("Please upload a file before clicking upload.")

st.header("Ask a Question")
question = st.text_input("Enter your question", placeholder="What is this document about?")

if st.button("Get Answer", type="primary"):
    if question:
        with st.spinner("Searching through your documents..."):
            response = requests.post(f"{FASTAPI_URL}/ask/?question={question}", timeout=60)
        
        if response.status_code == 200:
            answer = response.json().get('answer', {}).get('result', 'No answer received')
            
            st.subheader("Response")
            st.write(answer)
        else:
            try:
                st.error(f"Error: {response.json().get('detail')}")
            except:
                st.error("Error: Unable to parse response from server.")
    else:
        st.warning("Please enter a question before clicking Get Answer.")

