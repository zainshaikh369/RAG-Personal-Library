# 🧠 RAG-Based PDF Question-Answering Assistant

This project is a local Retrieval-Augmented Generation (RAG) assistant that lets you ask natural language questions about your PDF documents using locally hosted large language models (LLMs) via [Ollama](https://ollama.com).

Everything runs **offline** — your files never leave your machine.

---

## 🔍 Features

- 📄 Parse and process any PDF file
- 🧩 Sentence-level chunking for accurate retrieval
- 🔍 Local embeddings using `MiniLM-L6-v2`
- 🧠 LLM inference using lightweight models via Ollama (e.g., `phi`, `gemma`)
- 🔐 100% private and offline
- 🧵 Interactive CLI interface (Streamlit UI coming soon)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-pdf-assistant.git
cd rag-pdf-assistant
```

### 2. Set up a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add a PDF

Put your `.pdf` file in the `/data` folder. Only one file is loaded at a time (for now).

### 5. Start the model with Ollama

In a separate terminal:

```bash
ollama run phi
```

Or try other models like `gemma:2b`, `mistral`, etc.

### 6. Run the app

```bash
python main.py
```

---

## 💡 Example

```text
🤖 Ask questions about your PDF (type 'exit' to quit):

📝 Your question: What is the main idea of this document?

💡 Answer:
[LLM-generated response based on retrieved context]
```

---

## ⚙️ Supported Models via Ollama

You can easily switch between models by editing this line in `main.py`:

```python
OLLAMA_MODEL = "phi"  # Options: "phi", "gemma:2b", "gemma:7b", "mistral"
```

| Model        | Parameters | Notes                        |
|--------------|------------|------------------------------|
| `phi`        | ~2.7B      | Fast, accurate, very light   |
| `gemma:2b`   | 2B         | Google open-weight model     |
| `gemma:7b`   | 7B         | Higher quality, slower       |
| `mistral`    | 7B         | General-purpose, strong RAG  |

---

## 📦 Dependencies

See `requirements.txt` for exact packages.

Main ones:

- `llama-index`
- `llama-index-llms-ollama`
- `llama-index-embeddings-huggingface`
- `sentence-transformers`
- `PyMuPDF`

---

## 📁 Project Structure

```
├── main.py               # Main CLI app
├── requirements.txt
├── README.md
├── data/                 # Place your PDFs here
├── venv/                 # Your virtual environment (not tracked)
```

---

## ✍️ Author

**Zain Shaikh**  
📍 San Jose, CA  
📧 zainshaikh369@gmail.com  
🔗 [linkedin.com/in/zain-shaikh-a127351b2](https://www.linkedin.com/in/zain-shaikh-a127351b2)

---

## 🛡️ Disclaimer

This tool runs locally and does not send any data to external APIs. Please ensure that you have the right to process any documents you use with this assistant.

---

## 📌 Next Steps

- [ ] Add multi-document support  
- [ ] Build a Streamlit-based chat UI  
- [ ] Save/load vector index for faster reuse  
