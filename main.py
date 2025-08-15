# RAG-Based Chatbot for Internal Documentation
# Requirements: Install the following libraries
# pip install streamlit langchain-community langchain-huggingface faiss-cpu sentence-transformers transformers pypdf2

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import tempfile
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM (using a small seq2seq model for generation)
model_name = "google/flan-t5-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)
except Exception as e:
    st.error(f"Error initializing language model: {e}")
    st.stop()

# Directory for storing documents
DOC_DIR = "internal_docs"
if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR)

# Function to index documents
@st.cache_resource
def create_or_update_vectorstore():
    try:
        documents = []
        for file in os.listdir(DOC_DIR):
            file_path = os.path.join(DOC_DIR, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue
            docs = loader.load()
            documents.extend(docs)
        
        if not documents:
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error indexing documents: {e}")
        return None

# Streamlit App
st.title("RAG-Based Chatbot for Internal Documentation")

# Sidebar for uploading new documents
with st.sidebar:
    st.header("Upload New Document")
    uploaded_file = st.file_uploader("Choose a file (PDF or TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Move to DOC_DIR
            dest_path = os.path.join(DOC_DIR, uploaded_file.name)
            os.rename(tmp_file_path, dest_path)
            st.success(f"Document '{uploaded_file.name}' uploaded and indexed successfully!")
            # Clear cache to reindex
            create_or_update_vectorstore.clear()
        except Exception as e:
            st.error(f"Error uploading document: {e}")

# Main chat interface
vectorstore = create_or_update_vectorstore()

if vectorstore is None:
    st.warning("No documents indexed yet. Upload some documents to get started.")
else:
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the internal docs:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = qa_chain({"query": prompt})
                        response = result["result"]
                        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
                        st.markdown(response)
                        st.markdown(f"**Sources:** {', '.join(sources)}")
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
    except Exception as e:
        st.error(f"Error initializing QA chain: {e}")

# Instructions for running
if __name__ == "__main__":
    st.info("Please run this app using: `streamlit run main.py`")