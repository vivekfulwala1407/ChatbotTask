"""
Simple Streamlit Chatbot UI for RAG Query Engine.
"""
import streamlit as st
import requests

# Example queries to show
EXAMPLE_QUERIES = [
    "Which outlet has the highest sales in Surat?",
    "What is the best product in terms of rating and sales?",
    "How many products are there in category 'beverages'?",
]

# Page configuration
st.set_page_config(
    page_title="Café Chatbot",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state keys safely
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0  # Counter to force input refresh
if "auto_submit" not in st.session_state:
    st.session_state.auto_submit = False
if "auto_query" not in st.session_state:
    st.session_state.auto_query = ""

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-container { padding: 20px; border-radius: 10px; margin: 10px 0; }
    .user-message { background-color: #e3f2fd; padding: 15px; border-radius: 10px;
                    margin: 10px 0; border-left: 4px solid #2196F3; }
    .bot-response { background-color: #f5f5f5; padding: 15px; border-radius: 10px;
                   margin: 10px 0; border-left: 4px solid #4CAF50; }
    .title { color: #d4a574; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value="http://127.0.0.1:8000",
                            help="Base URL of the RAG API server")
    
    st.divider()
    st.markdown("### 📊 About")
    st.info("This chatbot answers questions about café outlets and products using a RAG system.")
    
    st.divider()
    st.markdown("### 💡 Example Queries")
    st.caption("Click any example to automatically send it:")
    
    # Display example queries in sidebar only
    for i, query in enumerate(EXAMPLE_QUERIES):
        if st.button(f"{query}", key=f"example_{i}", 
                     use_container_width=True,
                     help=f"Click to send this query: {query}"):
            # Set auto-submit flag and query
            st.session_state.auto_submit = True
            st.session_state.auto_query = query
            st.rerun()

# Title
st.markdown("<h1 class='title'>☕ Café Chatbot</h1>", unsafe_allow_html=True)
st.markdown("---")

# Conversation
st.subheader("💬 Conversation")
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-response"><b>Bot:</b> {msg["content"]}</div>', unsafe_allow_html=True)

# Input area
st.markdown("---")
st.subheader("📝 Ask a Question")

# Create a form
with st.form(key="question_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Show placeholder or auto_query if set
        placeholder_text = f"e.g., {EXAMPLE_QUERIES[0]}"
        
        # The clear_on_submit=True in the form will handle clearing
        user_input = st.text_input(
            "Your question:",
            placeholder=placeholder_text,
            label_visibility="collapsed",
            key=f"input_{st.session_state.input_key}"  # Dynamic key to force refresh
        )
    
    with col2:
        submit_button = st.form_submit_button("Send ✈️", use_container_width=True)
    
    # Process auto-submit from sidebar example
    if st.session_state.auto_submit and st.session_state.auto_query:
        q = st.session_state.auto_query
        
        # Append user message
        st.session_state.messages.append({"role": "user", "content": q})
        
        # Call API and append bot response
        with st.spinner("🔄 Processing your question..."):
            try:
                response = requests.post(f"{api_url}/chat", json={"question": q, "k": 5}, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    bot_answer = data.get("answer", "No response received.")
                    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
                else:
                    err = f"API Error {response.status_code}: {response.text}"
                    st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {e}"})
        
        # Reset auto-submit flag and increment input key
        st.session_state.auto_submit = False
        st.session_state.auto_query = ""
        st.session_state.input_key += 1
        
        # Rerun to refresh the conversation
        st.rerun()
    
    # Process manual form submission
    elif submit_button and user_input and user_input.strip():
        q = user_input.strip()
        
        # Append user message
        st.session_state.messages.append({"role": "user", "content": q})
        
        # Call API and append bot response
        with st.spinner("🔄 Processing your question..."):
            try:
                response = requests.post(f"{api_url}/chat", json={"question": q, "k": 5}, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    bot_answer = data.get("answer", "No response received.")
                    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
                else:
                    err = f"API Error {response.status_code}: {response.text}"
                    st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {e}"})
        
        # Increment the input key to force a refresh of the input field
        st.session_state.input_key += 1
        
        # Rerun to refresh the conversation
        st.rerun()

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray; font-size:12px;'>Powered by RAG Engine | Sentence Transformers | Groq API</div>", unsafe_allow_html=True)

with st.sidebar:
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.input_key += 1  # Also refresh input
        st.session_state.auto_submit = False
        st.session_state.auto_query = ""
        st.rerun()