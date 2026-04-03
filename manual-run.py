def main():
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
    result = llm_chain({"question": "what is transformer?"})
    answer = result["answer"]
    print(f"\n✅ Raw chain result: {result}")
    print(f"\n✅ FINAL ANSWER:\n{answer}")

    print("\n📚 SOURCE DOCUMENTS:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"\n--- Source {i+1} ---")
        print(doc.page_content[:300])

    while True:
        query = input("Ask: ")
        if query.lower() == "exit":
            break

        result = llm_chain({"question": query})
        print("\nAnswer:", result["answer"])
        print("-" * 50)

# uncomment below method to run as standalone script (python -m api.main)
# if __name__ == "__main__":
#     main()