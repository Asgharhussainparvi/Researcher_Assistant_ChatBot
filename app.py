# app.py  (new file — wrapper for Gradio)
import os
import gradio as gr
import traceback

# Import your original script as a module. Make sure your original file is named rag_bot.py
import simple_rag_chatbot as rag_bot

# We'll lazy-initialize the RAG chain so Space build can complete quickly
RAG_CHAIN = None

def ensure_chain():
    global RAG_CHAIN
    if RAG_CHAIN is not None:
        return RAG_CHAIN

    try:
        # Ensure API key is present (uses your function)
        rag_bot.setup_api_key()

        # Use the same embedding / llm classes you already import in rag_bot
        embeddings = rag_bot.HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        llm = rag_bot.ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

        # Use your PDF loader + splitter function
        chunks = rag_bot.get_pdf_text_chunks_from_folder("papers")
        if not chunks:
            raise ValueError("No chunks found from papers/ — ensure PDFs exist in 'papers/' in the repo.")

        vector_store = rag_bot.create_vector_store(chunks, embeddings)
        RAG_CHAIN = rag_bot.create_rag_chain(vector_store, llm)

        return RAG_CHAIN
    except Exception as e:
        # raise to let gradio show the error on UI
        raise RuntimeError(f"Failed to initialize RAG chain: {e}\n{traceback.format_exc()}")

def answer(question, history):
    if not question or not question.strip():
        return history, "Please type a question."

    try:
        chain = ensure_chain()
        # reuse the chain invoke call you used in CLI
        result = chain.invoke({"query": question})
        answer_text = result.get("result") if isinstance(result, dict) else str(result)
    except Exception as e:
        answer_text = f"Error while processing question: {e}\n{traceback.format_exc()}"

    # append to chat history (simple list of tuples)
    history = history or []
    history.append(("You", question))
    history.append(("Bot", answer_text))
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot — PDFs (Gradio UI)")
    chat = gr.Chatbot()
    state = gr.State([])
    txt = gr.Textbox(show_label=False, placeholder="Ask about the PDFs and press Enter")

    txt.submit(fn=answer, inputs=[txt, state], outputs=[chat, txt])
    gr.Markdown("Make sure your PDFs are in the `papers/` folder and your `GROQ_API_KEY` is set in Space Secrets.")

if __name__ == "__main__":
    # create a public share link (works even if localhost is not accessible)
    demo.launch(share=True)

