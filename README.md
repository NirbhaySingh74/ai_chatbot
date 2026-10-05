# 🤖 AI Chatbot (LangGraph + Groq + RAG)

A Streamlit chatbot built with LangGraph and Groq that can answer questions about your own documents using Retrieval-Augmented Generation (RAG).

## How RAG works here

1. **Upload** PDF, TXT or Markdown files from the sidebar and click **Add to knowledge base**.
2. The documents are split into overlapping chunks (`RecursiveCharacterTextSplitter`), embedded locally with [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`, no API key needed) and stored in a **FAISS** index saved to `vector_store/`, so it persists across restarts.
3. The LLM has a `search_documents` tool. LangGraph routes between the chatbot and tool nodes:

   ```
   START → chatbot ──(tool call?)──► tools ──► chatbot → END
                    └──(no)──────────────────────────► END
   ```

   The model decides when it needs your documents, so normal chit-chat still works. Answers cite sources like `[1]`, and the retrieved chunks show up in a **Sources** expander under each answer.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI: chat plus the knowledge-base sidebar |
| `langgraph_backend.py` | LangGraph agent with the retrieval tool |
| `rag.py` | Loading, chunking, embedding, FAISS storage and search |

## Setup

```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
streamlit run app.py
```

The first time you index a document, the embedding model (~70 MB) is downloaded once.

### Optional settings (env vars)

- `RAG_INDEX_DIR`: where the FAISS index is stored (default `vector_store`)
- `RAG_EMBEDDING_MODEL`: any FastEmbed-supported model (default `BAAI/bge-small-en-v1.5`). If you change it, clear the knowledge base and re-index.
