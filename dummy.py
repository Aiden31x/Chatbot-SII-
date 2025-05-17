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

# Create a dictionary for university contact information
university_contacts = {}

# Try to load university contact data from the JSON file
# Try to load university contact data from the JSON file
try:
    with open('faq_data.json', 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        for question, answer in qa_data.items():
            if "university" in question.lower() and ("contact" in question.lower() or "email" in question.lower()):
                university_contacts[question] = answer
except Exception as e:
    print(f"Failed to load university contact data: {e}")



# Set up the retriever
retriever = db.as_retriever()

# Initialize the LLM
llm = ChatOllama(model="mistral_prompt")

# Set up the retriever with appropriate parameters to limit results
retriever = db.as_retriever(search_kwargs={"k": 1})  # Only retrieve the top 1 most relevant document

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Global variable to track the last query
last_query_options = []
button_labels = ["", "", "", ""]

# Chat function with timestamp and quick replies
def styled_chat(query, history):
    global last_query_options, university_contacts
    
    time = datetime.datetime.now().strftime("%-I:%M:%S %p")
    
    # Check if this is a greeting/first message
    if any(keyword in query.lower() for keyword in ["hello", "hi", "hey", "start", "help", "assist", "what can you do", "options", "menu"]) and not history:
        # Special welcome message for first greeting
        response_text = "Hi, I am Study in India Assistant. How may I help you?"
        quick_replies = ["How to apply", "Contact details", "Available scholarships", "Required documents"]
    else:
        # Check if query is about a specific university
        university_mentioned = None
        contact_info = None
        
        # First, look for any university name in the query
        for uni_question in university_contacts.keys():
            # Extract university name from the question
            potential_uni_name = uni_question.lower().split("university")[0].strip() + " university"
            if potential_uni_name.lower() in query.lower():
                university_mentioned = potential_uni_name
                contact_info = university_contacts[uni_question]
                break
        
        # Regular flow for other queries
        result = rag_chain.invoke({"query": query})
        response_text = result['result']
        
        # Clean up the response
        # Remove embedded navigation options from the original JSON data
        response_text = response_text.split('\n\n 👉')[0].strip()
        response_text = response_text.replace("Q:", "").replace("A:", "").strip()
        
        # If a university was mentioned and we have contact info, append it to the response
        if university_mentioned and contact_info:
            contact_info = contact_info.split('\n\n 👉')[0].strip()  # Clean up contact info
            response_text += f"\n\n📧 **Contact Information for {university_mentioned.title()}:**\n{contact_info}"
        
        # Add quick reply buttons based on certain keywords in the query
        quick_replies = []
        
        if any(keyword in query.lower() for keyword in ["hello", "hi", "hey", "start", "help", "assist", "what can you do", "options", "menu"]):
            quick_replies = ["How to apply", "Contact details", "Available scholarships", "Required documents"]
        elif any(keyword in query.lower() for keyword in ["apply", "application", "admission"]):
            quick_replies = ["What documents do I need?", "Application timeline", "Required documents", "Application fee"]
        elif any(keyword in query.lower() for keyword in ["document", "upload", "certificate"]):
            quick_replies = ["Document formats accepted", "How to attest documents", "Deadline for submission"]
        elif any(keyword in query.lower() for keyword in ["contact", "reach", "connect"]):
            quick_replies = ["Email contacts", "Phone numbers", "Chat support", "Office locations"]
        elif any(keyword in query.lower() for keyword in ["scholarship", "financial", "fee", "waiver"]):
            quick_replies = ["Am I eligible for a scholarship?", "How to apply for scholarships", "Scholarship amount", "Deadlines"]
        # If asking about a university
        elif "university" in query.lower():
            quick_replies = ["How to apply", "Contact details", "Available scholarships", "Required documents"]
    
    # Store options for quick reply handling
    last_query_options = quick_replies
    
    # Add quick reply options as simple text with emoji bullets
    if quick_replies:
        response_text += "\n\n📋 **Quick options:**\n"
        for option in quick_replies:
            response_text += f"• {option}\n"
    
    history.append((f"{time}\n{query}", f"{time}\n{response_text}"))
    return history, history, "", gr.update(visible=bool(quick_replies))

# Function to handle button clicks and add to chat
def handle_button_click(btn_value, history):
    global last_query_options, university_contacts
    
    # Create a new chat message with the button text
    time = datetime.datetime.now().strftime("%-I:%M:%S %p")
    
    # More precise query to improve retrieval
    precise_query = btn_value
    
    # For specific common buttons, use more precise queries
    if btn_value == "Required documents":
        precise_query = "What documents do I need?"
    elif btn_value == "How to apply":
        precise_query = "How do I apply for the SII program?"
    elif btn_value == "Contact details":
        precise_query = "What are the contact details for Study in India?"
    elif btn_value == "Available scholarships":
        precise_query = "Tell me about scholarships"
    elif btn_value == "Am I eligible for a scholarship?":
        precise_query = "Am I eligible for a scholarship?"
    
    result = rag_chain.invoke({"query": precise_query})
    response_text = result['result']
    
    # Clean up the response
    # Remove embedded navigation options from the original JSON data
    response_text = response_text.split('\n\n 👉')[0].strip()
    response_text = response_text.replace("Q:", "").replace("A:", "").strip()
    
    # Determine new quick replies based on the button clicked
    quick_replies = []
    
    if btn_value == "How to apply":
        quick_replies = ["What documents do I need?", "Application timeline", "Application fee"]
    elif btn_value == "Contact details":
        quick_replies = ["Email contacts", "Phone numbers", "Office locations"]
    elif btn_value == "Available scholarships" or "scholarship" in btn_value.lower():
        quick_replies = ["Am I eligible for a scholarship?", "How to apply for scholarships", "What costs are not covered?"]
    elif any(keyword in btn_value.lower() for keyword in ["document", "required"]):
        quick_replies = ["Document formats accepted", "Can I apply without a passport?", "Can I upload documents later?"]
        
    # Special handling for "Email contacts" - show a list of universities
    if btn_value == "Email contacts":
        university_names = []
        for uni_question in university_contacts.keys():
            potential_uni_name = uni_question.lower().split("university")[0].strip() + " university"
            university_names.append(potential_uni_name.title())
            
        if university_names:
            response_text += "\n\nHere are some universities you can ask about:\n"
            for uni in university_names[:5]:  # Show only top 5 to avoid clutter
                response_text += f"• {uni}\n"
            response_text += "\nAsk me about any specific university for contact details."
            
            # Update quick replies to include some universities
            quick_replies = university_names[:4]  # Limit to 4 for the UI buttons
    
    # Add quick reply options to response
    if quick_replies:
        response_text += "\n\n📋 **Quick options:**\n"
        for option in quick_replies:
            response_text += f"• {option}\n"
    
    # Store new options
    last_query_options = quick_replies
    
    # Add both the button click as a user message and the response
    history.append((f"{time}\n{btn_value}", f"{time}\n{response_text}"))
    return history, history, gr.update(visible=bool(quick_replies))
    
    if any(keyword in btn_value.lower() for keyword in ["how to apply"]):
        quick_replies = ["Where do I upload documents?", "Application timeline", "Required documents"]
    elif any(keyword in btn_value.lower() for keyword in ["contact details"]):
        quick_replies = ["Email contacts", "Phone numbers", "Office locations"]
    elif any(keyword in btn_value.lower() for keyword in ["scholarship", "financial"]):
        quick_replies = ["Scholarship eligibility", "How to apply for scholarships", "Scholarship amount"]
    elif any(keyword in btn_value.lower() for keyword in ["document", "upload"]):
        quick_replies = ["Document formats accepted", "How to attest documents", "Deadline for submission"]
    
    # Add quick reply options to response
    if quick_replies:
        response_text += "\n\n📋 **Quick options:**\n"
        for option in quick_replies:
            response_text += f"• {option}\n"
    
    # Store new options
    last_query_options = quick_replies
    
    # Add both the button click as a user message and the response
    history.append((f"{time}\n{btn_value}", f"{time}\n{response_text}"))
    return history, history, gr.update(visible=bool(quick_replies))

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
    white-space: pre-wrap;
}
.gradio-chatbot .message.bot {
    background-color: #f1f1f1;
    color: black;
    padding: 8px;
    border-radius: 14px;
    margin: 4px;
    white-space: pre-wrap;
}
textarea {
    border: none !important;
    border-top: 1px solid #ccc !important;
    border-radius: 0px !important;
    padding: 8px;
}
.quick-reply-row button {
    background-color: #e9f5ff !important;
    color: #062e63 !important;
    border: 1px solid #062e63 !important;
    margin: 2px !important;
    padding: 6px 12px !important;
    border-radius: 16px !important;
    font-size: 12px !important;
    cursor: pointer !important;
    transition: all 0.3s;
}
.quick-reply-row button:hover {
    background-color: #062e63 !important;
    color: white !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("<div id='chat-header'>Study in India Assistant<br><small>Your guide to studying in India</small></div>")
    
    chatbot = gr.Chatbot(label="", height=370)
    state = gr.State([])
    
    # Quick reply buttons (initially hidden)
    with gr.Column(visible=False) as quick_reply_section:
        with gr.Row(elem_classes="quick-reply-row"):
            btn1 = gr.Button("", size="sm")
            btn2 = gr.Button("", size="sm")
        with gr.Row(elem_classes="quick-reply-row"):
            btn3 = gr.Button("", size="sm")
            btn4 = gr.Button("", size="sm")
    
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask about admissions, visa, etc...", show_label=False, lines=2, scale=8)
        send_btn = gr.Button("➤", elem_id="send-button", scale=1)
    
    clear = gr.Button("Clear Chat")
    
    # Update button labels when new options are available
    def update_buttons():
        global last_query_options, button_labels
        button_labels = ["", "", "", ""]
        if last_query_options:
            for i, option in enumerate(last_query_options[:4]):
                button_labels[i] = option
        return [gr.update(value=btn) for btn in button_labels]
    
    # Set up event handlers
    msg.submit(styled_chat, [msg, state], [chatbot, state, msg, quick_reply_section])
    send_btn.click(styled_chat, [msg, state], [chatbot, state, msg, quick_reply_section])
    clear.click(lambda: ([], [], "", gr.update(visible=False)), None, [chatbot, state, msg, quick_reply_section])
    
    # Set up quick reply button handlers with proper button text passing
    btn1.click(lambda history: handle_button_click(button_labels[0], history) if button_labels[0] else (history, history, gr.update()), 
               [state], [chatbot, state, quick_reply_section])
    btn2.click(lambda history: handle_button_click(button_labels[1], history) if button_labels[1] else (history, history, gr.update()), 
               [state], [chatbot, state, quick_reply_section])
    btn3.click(lambda history: handle_button_click(button_labels[2], history) if button_labels[2] else (history, history, gr.update()), 
               [state], [chatbot, state, quick_reply_section])
    btn4.click(lambda history: handle_button_click(button_labels[3], history) if button_labels[3] else (history, history, gr.update()), 
               [state], [chatbot, state, quick_reply_section])
    
    # Update button labels after each message
    chatbot.change(update_buttons, None, [btn1, btn2, btn3, btn4])

# Launch the interface
demo.launch()