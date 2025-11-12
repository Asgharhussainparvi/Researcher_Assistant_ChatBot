🧠 Research Assistant Chatbot using Groq API (RAG-based)

This project implements a Retrieval-Augmented Generation (RAG) chatbot powered by Groq API and LangChain, designed to answer questions by synthesizing information from multiple research paper PDFs.

It automatically loads all PDFs from a folder, processes and embeds their text, stores them in a FAISS vector database, and uses Groq’s LLaMA-3 model to generate context-aware answers.

🚀 Features

📂 Load multiple research article PDFs from a folder

🧩 Chunk and embed text using HuggingFace embeddings (all-MiniLM-L6-v2)

⚡ Fast and scalable vector retrieval via FAISS

🗣️ Natural language Q&A using Groq LLaMA-3.1-8B-Instant

🧠 Context-aware and citation-backed answers

💬 Interactive command-line chat interface

🏗️ Project Structure
📦 research-assistant-chatbot
├── papers/                    # Folder containing all your research PDFs
├── .env                       # Contains your Groq API key
├── requirements.txt            # Python dependencies
├── simple_rag_chatbot.py                    # Main chatbot script
└── README.md                   # Project documentation

⚙️ Installation
1. Clone the repository
git clone https://github.com/<your-username>/research-assistant-chatbot.git
cd research-assistant-chatbot

2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate     # for Linux/Mac
venv\Scripts\activate        # for Windows

3. Install dependencies
pip install -r requirements.txt


Example requirements.txt:

langchain
langchain_groq
langchain_huggingface
langchain_community
python-dotenv
faiss-cpu
PyPDF2

4. Set up environment variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here


You can obtain the API key from 👉 https://console.groq.com/

📘 Usage

Place all your research PDFs inside the papers/ folder.

Run the chatbot:

python main.py


Once loaded, you’ll see:

RAG Chatbot Ready! You can now ask questions about your PDFs.
Type 'exit' to quit.


Start asking questions like:

You: What is the main contribution of the second paper?
You: Explain the methodology used in the dataset section.

🧩 How It Works

PDF Loading: All PDFs are read and converted to text.

Text Chunking: Large text is split into manageable 1000-character chunks with overlaps.

Embedding: Each chunk is converted into vector form using HuggingFace embeddings.

Vector Store: FAISS stores these vectors for efficient similarity search.

Retrieval: Relevant chunks are retrieved based on user query.

Generation: Groq’s LLaMA-3 model synthesizes a concise and factual response using the context.

🧠 Example Output
You: What is the objective of the proposed model in these papers?
Thinking...

Answer:
The main objective of the proposed models across the papers is to improve contextual understanding and performance in domain-specific text classification tasks using lightweight transformer architectures.

============================================================

🧑‍💻 Technologies Used

Python 3.10+

LangChain

Groq API (LLaMA-3.1-8B-Instant)

FAISS

HuggingFace Sentence Embeddings

PyPDFLoader

📌 Future Enhancements

Add a Streamlit web interface

Integrate citation retrieval and document references

Enable long-term memory for multi-session chat

Support additional document types (e.g., Word, HTML, CSV)

🧑‍🔬 Author

Asghar Hussain
AI Researcher | Software Engineering Student | AI Intern
🔗 GitHub

💡 Passionate about AI, RAG systems, and scientific research automation.