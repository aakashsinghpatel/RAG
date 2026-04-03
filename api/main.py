from src.ingestion import load_documents, split_documents
from src.vectorstore import create_or_load_vectorstore,get_retriever
from src.llm import load_llm
from src.memory import get_memory
from src.chain import build_chain

# Import to setup API
from fastapi import FastAPI
from pydantic import BaseModel



app = FastAPI()

class QueryRequest(BaseModel):
    question: str

# Lead all neccesary module at start of server
@app.on_event("startup")
def startup_event():
    # Initialize once (IMPORTANT)
    global llm_chain
    print("\n🔹 STEP 1: Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    print("\n🔹 STEP 2: Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("\n🔹 STEP 3: Creating / Loading FAISS...")
    vectorstore = create_or_load_vectorstore(chunks)
    print("Vectorstore ready")

    print("\n🔹 STEP 4: Creating retriever...")
    retriever = get_retriever(vectorstore)
    print(f"Retriever ready with {vectorstore.index.ntotal} vectors")

    print("\n🔹 STEP 5: Loading LLM...")
    llm = load_llm()
    print(f"LLM loaded {llm}")

    print("\n🔹 STEP 6: Setting memory...")
    memory = get_memory()
    print(f"Memory ready with window size {memory}")

    print("\n🔹 STEP 7: Building RAG chain...")
    SESSION_ID = "test_session"
    llm_chain = build_chain(llm, retriever, session_id=SESSION_ID)
    print(f"\n✅ RAG Chatbot Ready! llm_chain:  {llm_chain}")
    print("✅ App initialized once!")

# Method for /chat API call 
@app.post("/chat")
def chat(request: QueryRequest):
    result = llm_chain({
        "question": request.question
    })
    return {"answer": result["answer"]}