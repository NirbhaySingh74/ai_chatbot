from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)


def chatbot_node(state: MessagesState):

    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


graph = StateGraph(MessagesState)

graph.add_node("chatbot", chatbot_node)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

checkpointer = InMemorySaver()

chatbot = graph.compile(
    checkpointer=checkpointer
)
