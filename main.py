import os
import time
import warnings
from llama_index.core import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.readers.file import PyMuPDFReader

# === Silence warnings ===
warnings.filterwarnings("ignore")

# === Config ===
PDF_DIR = "./data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "phi"  # lightweight + high quality for RAG
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 10
TIMEOUT = 180
RETRIES = 3

# === Load PDF File Dynamically ===
pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
if not pdf_files:
    raise FileNotFoundError(f"❌ No PDF files found in '{PDF_DIR}'.")

file_path = os.path.join(PDF_DIR, pdf_files[0])
file_name = os.path.splitext(os.path.basename(file_path))[0]
print(f"📄 Loading file: {file_path}")

# === Load full PDF and attach filename as metadata ===
pdf_loader = PyMuPDFReader()
documents = pdf_loader.load(file_path=file_path)

for doc in documents:
    doc.metadata = {"file_name": file_name}

# === Extract Page 1 for always-included context ===
title_text = documents[0].text.split("\f")[0]
title_node = TextNode(text=title_text)
title_node.metadata = {"file_name": file_name}
title_node_with_score = NodeWithScore(node=title_node, score=1.0)

# === Set Local Embeddings ===
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
print(f"🔍 Using local embedding model: {EMBEDDING_MODEL_NAME}")

# === Chunking ===
parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
nodes = parser.get_nodes_from_documents(documents)

# === Build vector index ===
index = VectorStoreIndex(nodes)
print("📦 Vector index created.")

# === Start Ollama LLM ===
try:
    llm = Ollama(model=OLLAMA_MODEL, request_timeout=TIMEOUT)
except Exception as e:
    raise RuntimeError("❌ Could not connect to Ollama. Run: `ollama run phi`") from e

# === Set up retriever and synthesizer ===
base_retriever = index.as_retriever(similarity_top_k=TOP_K)
response_synthesizer = get_response_synthesizer(llm=llm)

# === Custom retriever with title injection (quiet version) ===
class CustomRetriever:
    def __init__(self, retriever, inject_node=None):
        self.retriever = retriever
        self.inject_node = inject_node

    def retrieve(self, query):
        nodes = self.retriever.retrieve(query)
        if self.inject_node:
            nodes = [self.inject_node] + nodes
        return nodes

custom_retriever = CustomRetriever(base_retriever, inject_node=title_node_with_score)

query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

# === Q&A loop ===
print("\n🤖 Ask questions about your PDF (type 'exit' to quit):\n")

while True:
    query = input("📝 Your question: ").strip()
    if query.lower() in ("exit", "quit"):
        print("👋 Exiting.")
        break

    for attempt in range(RETRIES):
        try:
            response = query_engine.query(query)
            print(f"\n💡 Answer:\n{response}\n")
            break
        except Exception as e:
            print(f"⚠️ Error (attempt {attempt + 1}): {e}")
            time.sleep(2)
    else:
        print("❌ All retries failed. Try again or simplify your query.")
