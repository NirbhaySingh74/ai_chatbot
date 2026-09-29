from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot
import streamlit as st

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 AI Chatbot")
st.caption("Powered by LangGraph + Groq")


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}


for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])


user_input = st.chat_input("Type your message...")


if user_input:

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Save user message
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # -------------------------
    # Generate AI Response
    # -------------------------

    with st.chat_message("assistant"):

        # Show loader while LLM is responding
        with st.spinner("Thinking..."):

            response = chatbot.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG
            )

            # Get latest AI message
            ai_message = response["messages"][-1].content

        # Display AI response
        st.write(ai_message)

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })
