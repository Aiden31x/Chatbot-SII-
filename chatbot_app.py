import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
import os
import json
import datetime

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create the FAISS index
if os.path.exists("sii_qa_index"):
    db = FAISS.load_local("sii_qa_index", embedding_model, allow_dangerous_deserialization=True)
else:
    with open('faq_data.json', 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    documents = [Document(page_content=f"Q: {q}\nA: {a}") for q, a in qa_pairs.items()]
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local("sii_qa_index")

# Set up the retriever
retriever = db.as_retriever()

# Initialize the LLM
llm = ChatOllama(model="mistral_prompt")

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Chat function with timestamp
def styled_chat(query, history):
    time = datetime.datetime.now().strftime("%-I:%M:%S %p")
    result = rag_chain.invoke({"query": query})
    history.append((f"{time}\n{query}", f"{time}\n{result['result']}"))
    return history, history, ""

# CSS for styling the chat interface
custom_css = """
body {
    background-color: white;
}
#chat-header {
    background-color: #062e63;
    color: white;
    padding: 10px;
    font-weight: bold;
    font-size: 16px;
    text-align: center;
    border-radius: 6px 6px 0 0;
}
#chat-header small {
    color: white;
}
#send-button {
    font-size: 20px;
    background-color: #ff8c00 !important;
    color: white !important;
    border-radius: 6px !important;
    height: 100%;
    margin-left: 4px;
    padding: 0 16px;
}
.gradio-container {
    max-width: 380px !important;
    margin: auto;
    border: 1px solid #ccc;
    border-radius: 6px;
    background-color: white;
}
.gradio-chatbot .message.user {
    background-color: #5e5ef5;
    color: white;
    padding: 8px;
    border-radius: 14px;
    margin: 4px;
    align-self: flex-end;
    max-width: 75%;
    white-space: pre-wrap;
}
.gradio-chatbot .message.bot {
    background-color: #f1f1f1;
    color: black;
    padding: 8px;
    border-radius: 14px;
    margin: 4px;
    align-self: flex-start;
    max-width: 75%;
    white-space: pre-wrap;
}
.loading-wrap {
    display: none !important;
}
textarea {
    border: none !important;
    border-top: 1px solid #ccc !important;
    border-radius: 0px !important;
    padding: 8px;
}
button {
    background-color: #ff8c00 !important;
    color: white !important;
    border-radius: 0px !important;
    padding: 8px;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("<div id='chat-header'>Study in India Assistant<br><small>Your guide to studying in India</small></div>")
    
    chatbot = gr.Chatbot(label="", elem_classes="chatbot-container", height=370)
    
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask about admissions, visa, etc...", show_label=False, lines=2, scale=8)
        send_btn = gr.Button("➤", elem_id="send-button", scale=1)
    clear = gr.Button("Clear Chat")
    state = gr.State([])
    msg.submit(styled_chat, [msg, state], [chatbot, state, msg])
    send_btn.click(styled_chat, [msg, state], [chatbot, state, msg])
    clear.click(lambda: ([], [], ""), None, [chatbot, state, msg])
    gr.HTML(
        """
        <script>
        const observer = new MutationObserver(() => {
            const chat = document.querySelector('.gradio-chatbot');
            if (chat) {
                chat.scrollTop = 0;
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
        </script>
        """
    )

# Launch the interface
demo.launch()