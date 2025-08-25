import streamlit as st
from pathlib import Path

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import TextNode, NodeWithScore


# ============== Cached RAG Index ==================
@st.cache_resource
def build_index(pdf_path: Path):
    """Build and cache the index for a given PDF path."""
    pdf_loader = PyMuPDFReader()
    documents = pdf_loader.load(file_path=pdf_path)

    file_name = pdf_path.stem
    for doc in documents:
        doc.metadata = {"file_name": file_name}

    # Always include page 1 (often title/introduction) as high-score context
    title_text = documents[0].text.split("\f")[0]
    title_node = TextNode(text=title_text, metadata={"file_name": file_name})
    title_node_with_score = NodeWithScore(node=title_node, score=1.0)

    # Embeddings + chunking
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)

    # Vector index + retriever
    index = VectorStoreIndex(nodes)
    retriever = index.as_retriever(similarity_top_k=10)

    # Inject title page node into every retrieval result
    class CustomRetriever:
        def __init__(self, base_retriever, inject_node=None):
            self.retriever = base_retriever
            self.inject_node = inject_node

        def retrieve(self, query):
            chunks = self.retriever.retrieve(query)
            return [self.inject_node] + chunks if self.inject_node else chunks

    return CustomRetriever(retriever, inject_node=title_node_with_score)


def answer_question(custom_retriever, question: str) -> str:
    """Run the RAG query over the cached index."""
    llm = Ollama(model="phi", request_timeout=180)
    query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=get_response_synthesizer(llm=llm),
    )
    response = query_engine.query(question)
    return str(response)


# ============== Streamlit UI (Chat) ==================
st.set_page_config(page_title="PDF Q&A with Local LLM", layout="centered")
st.title("📄 RAG-based Personal Library (Chat)")

# Initialize session state for chat & file tracking
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}
if "active_file" not in st.session_state:
    st.session_state.active_file = None

# Sidebar: upload + controls
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    st.caption("Tip: use **/clear** in chat to reset history.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.success("Chat cleared.")

# Handle file save & index building
retriever = None
if uploaded_file:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    save_path = data_dir / uploaded_file.name

    # Save file to disk (overwrite if same name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Reset chat if the file changed
    if st.session_state.active_file != str(save_path):
        st.session_state.active_file = str(save_path)
        st.session_state.messages = []  # new doc → new conversation
        st.toast(f"Loaded: {save_path.name}", icon="📄")

    # Build/obtain cached index
    with st.spinner("Indexing document (first time only)…"):
        retriever = build_index(save_path)
else:
    st.info("Upload a PDF in the sidebar to start chatting.")

# Render previous messages (chat history)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask a question about the uploaded PDF…")
if prompt:
    # Special command to clear chat quickly
    if prompt.strip().lower() in {"/clear", "clear", "/reset"}:
        st.session_state.messages = []
        st.experimental_rerun()

    if not retriever:
        with st.chat_message("assistant"):
            st.markdown("Please upload a PDF first from the sidebar.")
    else:
        # Show user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = answer_question(retriever, prompt)
                except Exception as e:
                    reply = f"Sorry, I ran into an error: `{e}`"
            st.markdown(reply)

        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": reply})
