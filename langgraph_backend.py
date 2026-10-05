from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import rag

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)


@tool(response_format="content_and_artifact")
def search_documents(query: str):
    """Search the user's uploaded documents for information relevant to the query.
    Use this whenever the question might be answered by the uploaded documents."""

    docs = rag.search(query)

    if not docs:
        return "No relevant documents found.", []

    return rag.format_docs(docs), docs


tools = [search_documents]

llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """You are a helpful AI assistant.

The user can upload documents (PDF, TXT, MD). Use the `search_documents` tool \
to look up information in them whenever a question could relate to their content. \
When you answer using retrieved content, ground your answer in it and cite the \
sources like [1], [2]. If the documents don't contain the answer, say so and \
answer from general knowledge if you can.

Currently indexed documents: {sources}"""


def chatbot_node(state: MessagesState):

    sources = rag.list_sources()
    system = SystemMessage(
        content=SYSTEM_PROMPT.format(sources=", ".join(sources) if sources else "none")
    )

    model = llm_with_tools if sources else llm

    response = model.invoke([system] + state["messages"])

    return {
        "messages": [response]
    }


graph = StateGraph(MessagesState)

graph.add_node("chatbot", chatbot_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chatbot")
# If the model asked to search the documents, run the tool, otherwise finish
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")

checkpointer = InMemorySaver()

chatbot = graph.compile(
    checkpointer=checkpointer
)
