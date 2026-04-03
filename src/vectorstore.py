from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import config

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

def create_or_load_vectorstore(chunks):
    embeddings = get_embeddings()
    print(f"Embeddings model loaded {embeddings}")
    # ✅ If already saved → load
    faiss_file = os.path.join(config.FAISS_PATH, "index.faiss")
    if os.path.exists(faiss_file):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            config.FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # ✅ Save locally
        vectorstore.save_local(config.FAISS_PATH)
        print("FAISS index saved locally!")
    print(f"Vectorstore ready with {vectorstore.index.ntotal} vectors")
    return vectorstore


def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_kwargs={"k": config.TOP_K}
    )