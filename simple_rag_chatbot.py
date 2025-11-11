import os
import sys

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def setup_api_key():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in .env file.")
        print("Please get your API key from https://console.groq.com/ and add it to a .env file.")
        sys.exit(1)
    print("GROQ API Key found.")

def get_pdf_text_chunks_from_folder(folder_path):
    """
    Loads and splits all PDFs inside the folder into text chunks.
    """
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files. Processing them...\n")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        try:
            print(f"Loading: {pdf_file}")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"✓ Processed '{pdf_file}' ({len(chunks)} chunks)")
        except Exception as e:
            print(f"⚠️ Skipping '{pdf_file}' due to error: {e}")

    print(f"\nTotal chunks created from all PDFs: {len(all_chunks)}")
    return all_chunks

def create_vector_store(chunks, embeddings):
    print("Creating FAISS vector store...")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        sys.exit(1)

def create_rag_chain(vector_store, llm):
    template = """
    Use the following context to answer the question at the end.
    If you don't know the answer, say you don't know.
    Keep the answer short and factual.
    
    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    return qa_chain

def main():
    PDF_PATH = "papers"

    print("Starting RAG Chatbot with multiple PDFs...\n")
    setup_api_key()

    try:
        print("Loading HuggingFace embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("Embedding model loaded.")
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)
    except Exception as e:
        print(f"Error initializing models: {e}")
        sys.exit(1)

    chunks = get_pdf_text_chunks_from_folder(PDF_PATH)
    if not chunks:
        print("No text chunks extracted from any PDF. Exiting.")
        sys.exit(1)

    vector_store = create_vector_store(chunks, embeddings)
    rag_chain = create_rag_chain(vector_store, llm)

    print("\n" + "="*60)
    print("RAG Chatbot Ready! You can now ask questions about your PDFs.")
    print("Type 'exit' to quit.")
    print("="*60 + "\n")

    while True:
        user_question = input("You: ")
        if user_question.lower() == 'exit':
            print("Goodbye!")
            break
        if not user_question.strip():
            continue
        print("Thinking...")
        try:
            result = rag_chain.invoke({"query": user_question})
            print("\nAnswer:")
            print(result["result"])
        except Exception as e:
            print(f"Error during query processing: {e}")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
