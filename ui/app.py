import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.title("📚 RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask something:")

if st.button("Send"):
    if user_input:
        response = requests.post(API_URL, json={"question": user_input})
        answer = response.json()["answer"]

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", answer))

for i in range(0, len(st.session_state.chat_history), 2):
    with st.container():
        col1, col2 = st.columns([1, 8])

        # User
        if i < len(st.session_state.chat_history):
            role, text = st.session_state.chat_history[i]
            st.markdown(f"🧑 **You:** {text}")

        # Bot
        if i + 1 < len(st.session_state.chat_history):
            role, text = st.session_state.chat_history[i + 1]
            st.markdown(f"🤖 **Bot:** {text}")

        st.divider()