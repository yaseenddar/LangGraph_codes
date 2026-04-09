import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

# ------------------ INIT ------------------

if "chats" not in st.session_state:
    st.session_state["chats"] = {}

if "current_chat" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state["current_chat"] = new_id
    st.session_state["chats"][new_id] = []

# ------------------ CONFIG ------------------

CONFIG = {
    "configurable": {
        "thread_id": st.session_state["current_chat"]
    }
}

# ------------------ SIDEBAR ------------------

with st.sidebar:
    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state["chats"][new_id] = []
        st.session_state["current_chat"] = new_id
        st.rerun()

    st.markdown("## 💬 Chats")

    for chat_id in st.session_state["chats"].keys():
        if st.button(f"Chat {chat_id[:6]}", key=chat_id):
            st.session_state["current_chat"] = chat_id
            st.rerun()

# ------------------ CURRENT CHAT ------------------

messages = st.session_state["chats"][st.session_state["current_chat"]]

# render messages
for message in messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# ------------------ INPUT ------------------

user_input = st.chat_input("Type your message")

if user_input:

    # store user message
    messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.text(user_input)

    # call model
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG
    )

    ai_message = response["messages"][-1].content

    # store AI message
    messages.append({
        "role": "assistant",
        "content": ai_message
    })

    with st.chat_message("assistant"):
        st.text(ai_message)