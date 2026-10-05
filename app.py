from langchain_core.messages import HumanMessage, ToolMessage
from langgraph_backend import chatbot
import rag
import streamlit as st

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 AI Chatbot")
st.caption("Powered by LangGraph + Groq · RAG over your documents")


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}


# -------------------------
# Sidebar: Knowledge Base
# -------------------------

with st.sidebar:

    st.header("📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Add to knowledge base", disabled=not uploaded_files, use_container_width=True):

        with st.spinner("Indexing documents..."):

            total_chunks = 0

            for file in uploaded_files:
                try:
                    docs = rag.load_file(file.name, file.getvalue())
                    total_chunks += rag.add_documents(docs)
                except Exception as e:
                    st.error(f"Failed to index {file.name}: {e}")

        st.success(f"Indexed {total_chunks} chunks")

    sources = rag.list_sources()

    if sources:
        st.subheader("Indexed documents")

        for source in sources:
            st.markdown(f"- {source}")

        if st.button("Clear knowledge base", use_container_width=True):
            rag.clear_index()
            st.rerun()
    else:
        st.info("No documents yet. Upload some to chat with them.")


def render_sources(docs):

    if not docs:
        return

    with st.expander(f"📄 Sources ({len(docs)})"):

        for i, doc in enumerate(docs, start=1):
            source = doc["source"]
            page = doc.get("page")
            label = f"{source} — page {page + 1}" if page is not None else source

            st.markdown(f"**[{i}] {label}**")
            st.caption(doc["content"][:500] + ("..." if len(doc["content"]) > 500 else ""))


for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])
        render_sources(message.get("sources"))


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

            # Collect the chunks retrieved during this turn
            retrieved = []

            for msg in reversed(response["messages"]):
                if isinstance(msg, HumanMessage):
                    break
                if isinstance(msg, ToolMessage) and msg.artifact:
                    retrieved = [
                        {
                            "source": d.metadata.get("source", "unknown"),
                            "page": d.metadata.get("page"),
                            "content": d.page_content
                        }
                        for d in msg.artifact
                    ] + retrieved

        # Display AI response
        st.write(ai_message)
        render_sources(retrieved)

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message,
        "sources": retrieved
    })
