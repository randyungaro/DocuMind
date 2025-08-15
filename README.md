RAG-Based Chatbot for Internal Documentation
Overview
This is a Retrieval-Augmented Generation (RAG) chatbot designed to answer queries based on internal company documentation (e.g., PDFs and text files). It uses LangChain, Hugging Face Transformers, and FAISS to retrieve relevant document chunks and generate accurate responses. The app includes a Streamlit-based web interface for uploading documents and interacting with the chatbot.
Features

Document Indexing: Upload PDF or TXT files to automatically index and store in a FAISS vector database.
Query Processing: Ask questions about internal documents, and the chatbot retrieves relevant information using RAG.
Chat Interface: User-friendly Streamlit UI with chat history and source citation for transparency.
Dynamic Updates: New documents are automatically indexed upon upload, updating the knowledge base.

Requirements

Python 3.7+
Required packages:pip install streamlit langchain-community langchain-huggingface faiss-cpu sentence-transformers transformers pypdf2



Installation

Clone the repository:git clone https://github.com/yourusername/your-repo.git
cd your-repo


Create and activate a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Run the Streamlit app:streamlit run main.py


Open the provided URL (e.g., http://localhost:8501) in your browser.

Usage

Upload Documents:
Use the sidebar to upload PDF or TXT files.
Uploaded files are saved to the internal_docs folder and indexed automatically.


Ask Questions:
Enter queries in the chat input (e.g., "What are the steps for onboarding from cookbook.pdf?").
The chatbot responds with answers and cites source documents.


Prompting Tips:
Be specific: e.g., "Summarize the leave policy from the employee handbook."
Request a brief reasoning summary for complex queries: e.g., "List server setup steps and explain your reasoning first."
Avoid contradictory instructions to ensure accurate responses.



Project Structure
your-repo/
├── main.py               # Main Streamlit app and RAG pipeline
├── internal_docs/        # Folder for storing uploaded documents
├── requirements.txt      # List of dependencies
└── README.md             # This file

Example Prompt
From cookbook.pdf, list the steps to configure a new server. Start with a brief bullet-point summary of how you found the answer.

Troubleshooting

ModuleNotFoundError: Ensure all dependencies are installed (pip install langchain-huggingface).
Streamlit Errors: Run the app with streamlit run main.py, not python main.py.
Model Performance: For better results, consider using a larger model like google/flan-t5-large (update main.py accordingly).
Document Indexing Issues: Verify that internal_docs folder exists and contains valid PDF or TXT files.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.