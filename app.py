import streamlit as st
from rag_agent import ask_college_bot

# ==========================================
# PAGE CONFIGURATION (Must be first Streamlit command)
# ==========================================
st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to mimic the ChatGPT dark-theme sidebar feel
st.markdown("""
<style>
    /* Add a slight border to the sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid #333;
    }
    /* Style the generic buttons to look like the 'New Chat' button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #555;
        background-color: transparent;
    }
    .stButton>button:hover {
        border-color: #888;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
# This keeps the chat history visible on the screen
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# SIDEBAR (ChatGPT Style)
# ==========================================
with st.sidebar:
    # New Chat Button
    if st.button("📝 New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fake history list just to mimic the UI (Streamlit session state forgets this on refresh unless linked to a DB)
    st.caption("Recent")
    st.markdown("💬 Faculty Schedules\n\n💬 Final Year Projects\n\n💬 CSE Timetables")
    
    # Bottom spacer
    st.markdown("<br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("👤 **Aryan Pandey**")

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================
st.title("What are you working on?")

# 1. Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Chat Input Box
if prompt := st.chat_input("Ask anything about Faculty Schedules, Projects, or your teachers..."):
    
    # Append user question to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call your existing backend function
                response = ask_college_bot(prompt)
                st.markdown(response)
                # Append assistant response to UI
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Sorry, I ran into an issue connecting to the database: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})