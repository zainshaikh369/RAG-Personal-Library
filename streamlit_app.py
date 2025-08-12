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

# === CACHED: Build the RAG Index ===
@st.cache_resource
def build_index(pdf_path: Path):
    pdf_loader = PyMuPDFReader()
    documents = pdf_loader.load(file_path=pdf_path)

    file_name = pdf_path.stem
    for doc in documents:
        doc.metadata = {"file_name": file_name}

    title_text = documents[0].text.split("\f")[0]
    title_node = TextNode(text=title_text, metadata={"file_name": file_name})
    title_node_with_score = NodeWithScore(node=title_node, score=1.0)

    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    retriever = index.as_retriever(similarity_top_k=10)

    class CustomRetriever:
        def __init__(self, base_retriever, inject_node=None):
            self.retriever = base_retriever
            self.inject_node = inject_node

        def retrieve(self, query):
            chunks = self.retriever.retrieve(query)
            return [self.inject_node] + chunks if self.inject_node else chunks

    return CustomRetriever(retriever, inject_node=title_node_with_score)

# === Query Processing ===
def process_question(custom_retriever, question: str) -> str:
    llm = Ollama(model="phi", request_timeout=180)
    query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=get_response_synthesizer(llm=llm)
    )
    response = query_engine.query(question)
    return str(response)

# === Streamlit UI ===
st.set_page_config(page_title="PDF Q&A with Local LLM", layout="centered")
st.title("📄 RAG-based Personal Library")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
user_question = st.text_input("Ask a question about the document:")
response_placeholder = st.empty()

if st.button("Get Answer"):
    if uploaded_file is None or user_question.strip() == "":
        st.warning("Please upload a PDF and enter a question.")
    else:
        save_path = Path("data") / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.success(f"PDF saved as: {save_path.name}")

        # Build index and process question
        with st.spinner("Thinking..."):
            retriever = build_index(save_path)
            answer = process_question(retriever, user_question)

        response_placeholder.success(answer)
