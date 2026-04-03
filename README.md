# 📚 RAG Chatbot Application

## 🚀 Project Description

This project implements a **Retrieval-Augmented Generation (RAG) based chatbot** that answers user queries using information extracted from multiple research papers (PDFs).

The system combines:

* **Semantic search using FAISS (vector database)**
* **Embedding model: sentence-transformers/all-MiniLM-L6-v2**
* **Local LLM: LLaMA3 via Ollama**
* **Conversational memory (last 4 interactions)**
* **FastAPI backend + Streamlit chat UI**

The chatbot retrieves relevant content from documents and generates accurate, context-aware answers.

---

## 🧠 Key Features

* 📄 PDF ingestion and preprocessing
* ✂️ Smart text chunking
* 🔎 Semantic retrieval using FAISS
* 🤖 Local LLM-based answer generation (Ollama)
* 🧠 Conversational memory (last 4 interactions)
* 🌐 FastAPI backend for API communication
* 💬 Streamlit-based chat interface
* 📊 Evaluation support using RAGAS

---

## 📁 Project Structure

```bash
rag-app/
│
├── data/
│   └── pdf/                  # Input PDF files
│
├── faiss_index/               # Contains vectors for retrival
│                   
│
├── src/
│   ├── ingestion.py          # PDF loading & chunking
│   ├── vectorstore.py        # FAISS setup (save/load)
│   ├── llm.py                # LLM (Ollama integration)
│   ├── memory.py             # Conversation memory
│   ├── chain.py              # RAG chain
│   ├── evaluation.py         # RAGAS evaluation
│
├── api/
│   └── main.py               # FastAPI backend with laoding all modules
│
├── ui/
│   └── app.py                # Streamlit UI application
│
├── config.py                 # Configurations
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Setup Instructions (Using UV)

### 1️⃣ Install UV (if not installed)

```bash
pip install uv
```

---

### 2️⃣ Create Virtual Environment

```bash
uv venv
```

---

### 3️⃣ Activate Virtual Environment

#### Windows:

```bash
.venv\Scripts\activate
```

#### Mac/Linux:

```bash
source .venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

## 🤖 Setup Ollama (LLM)

1. Install Ollama from official website
2. Pull model:

```bash
ollama pull llama3
```

3. (Optional) Run once to verify:

```bash
ollama run llama3
```

---

## ▶️ Running the Application

### 🔹 Start Backend (Terminal 1)

```bash
(venv) rag-app> uvicorn api.main:app --reload
```

---

### 🔹 Start UI (Terminal 2)

```bash
(venv) rag-app> streamlit run ui/app.py
```

---

## 🌐 Access the Application

Open in browser:

```
http://localhost:8501
```

---

## 🔄 How It Works

1. PDFs are loaded and converted into text
2. Text is split into chunks
3. Chunks are converted into embeddings
4. Embeddings are stored in FAISS
5. User query is embedded
6. Relevant chunks are retrieved
7. LLM generates answer using retrieved context
8. Memory maintains last 4 interactions

---

## ⚠️ Important Notes

* Always run commands from **project root (`rag-app`)**
* Ensure Ollama is running before asking queries
* FAISS index is stored locally (`faiss_index/`)
* First run may take time (model download)

---

## 🧩 Tech Stack

* LangChain
* FAISS (Vector Database)
* Sentence Transformers (Embeddings)
* Ollama (LLaMA3 - Local LLM)
* FastAPI (Backend API)
* Streamlit (Chat UI)

---

## 🎯 Future Improvements

* Add streaming responses
* Improve UI (chat bubbles)
* Use better embedding & LLM models 
* Add reranking
* Deploy to cloud

---

## 🏁 Conclusion

This project demonstrates a complete **end-to-end RAG system**, integrating document retrieval, semantic search, and conversational AI into a scalable architecture.

---

## 🙌 Author

Developed as part of Generative AI training assignment.

<img width="956" height="473" alt="image" src="https://github.com/user-attachments/assets/24b7168e-4078-48bb-98d8-de16d9ebb69f" />
