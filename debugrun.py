import os
os.environ["TRANSFORMERS_NO_TORCH"] = "1"
from src.ingestion import load_documents, split_documents
from src.vectorstore import create_or_load_vectorstore, get_retriever
from src.llm import load_llm
from src.memory import get_memory
from src.chain import build_chain
from src.memory import add_to_memory, get_history
from langchain_core.messages import HumanMessage, AIMessage



def main():
    print("\n🔹 STEP 1: Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")
    # print(f"First page preview:\n{docs[0].page_content[:500]}") if docs else print("No documents loaded")

    print("\n🔹 STEP 2: Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("\n🔹 STEP 3: Creating / Loading FAISS...")
    vectorstore = create_or_load_vectorstore(chunks)
    print("Vectorstore ready")

    print("\n🔹 STEP 4: Creating retriever...")
    retriever = get_retriever(vectorstore)
    print(f"Retriever ready with {vectorstore.index.ntotal} vectors")

    # print("\n🔹 STEP 5: Testing retrieval...")
    # test_query = "What is transformer?"
    # retrieved_docs = retriever.invoke(test_query)

    # print(f"\nTop {len(retrieved_docs)} retrieved chunks:")
    # for i, doc in enumerate(retrieved_docs):
    #     print(f"\n--- Chunk {i+1} ---")
    #     print(doc.page_content[:300])  # preview

    #  from heer correct
    print("\n🔹 STEP 6: Loading LLM...")
    llm = load_llm()
    print(f"LLM loaded {llm}")

    # print(llm.invoke("hello world") )
    # print(f"\n  LLM Output: {llm.invoke('hello world').content}")
    # # Example prompt
#     prompt = """Context: Figure 1: The Transformer - model architecture.
# The Transformer follows this overall architecture using stacked self-attention and point-wise, fully
# connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,
# respectively.
# 3.1
# Encoder and Decoder Stacks
# Encoder:
# The encoder is composed of a stack of N = 6 identical layers. Each layer has two
# sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-
# wise fully connected feed-forward network. We employ a residual connection [11] around each of
# the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
# LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
# itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
# layers, produce outputs of dimension dmodel = 512.
# Decoder:

# the next output timestep, and so on. As such, a Transformer decoder (without an encoder)
# can be used as a language model (LM), i.e. a model trained solely for next-step prediction
# (Liu et al., 2018; Radford et al., 2018; Al-Rfou et al., 2019). This constitutes the second
# model structure we consider. A schematic of this architecture is shown in Figure 4, middle.
# In fact, early work on transfer learning for NLP used this architecture with a language
# modeling objective as a pre-training method (Radford et al., 2018).
# Language models are typically used for compression or sequence generation (Graves,
# 2013). However, they can also be used in the text-to-text framework simply by concatenating
# the inputs and targets. As an example, consider the case of English to German translation:
# If we have a training datapoint with input sentence “That is good.” and target “Das ist
# gut.”, we would simply train the model on next-step prediction over the concatenated input

# fully-visible masking on a portion of the input sequence.
# at the timestep corresponding to the classification token is then used to make a prediction
# for classifying the input sequence.
# The self-attention operations in the Transformer’s decoder use a “causal” masking pattern.
# When producing the ith entry of the output sequence, causal masking prevents the model
# from attending to the jth entry of the input sequence for j > i. This is used during training
# so that the model can’t “see into the future” as it produces its output. An attention matrix
# for this masking pattern is shown in Figure 3, middle.
# The decoder in an encoder-decoder Transformer is used to autoregressively produce an
# output sequence. That is, at each output timestep, a token is sampled from the model’s
# predicted distribution and the sample is fed back into the model to produce a prediction for
# the next output timestep, and so on. As such, a Transformer decoder (without an encoder)
# Question: What is transformer?
# Answer:
#     """
#     print(llm.invoke(prompt) )
#     print(f"\nLLM Output for prompt: {llm.invoke('hello world').content}")

    #  till heer correct

    # print(f"\nTesting LLM with prompt: '{prompt}'")
    # Generate output
    # output = llm("hello world")
    # print(f"LLM Output: {output}")
    # Extract the generated text
    # generated_text = output.generations[0][0].text
    # print(f"LLM Output: {generated_text}")
    # The output is a list of dictionaries
    # 
    print("\n🔹 STEP 7: Setting memory...")
    memory = get_memory()
    print(f"Memory ready with window size {memory}")

    print("\n🔹 STEP 8: Building RAG chain...")
    # qa_chain = build_chain(llm, retriever,memory)
    # qa_chain = build_chain(llm, retriever)


    # print("\n🔹 STEP 9: Running full RAG query...")
    # result = qa_chain.invoke({"question": test_query})

    # print(f"\n✅ FINAL ANSWER:\n{result}")
    # # print("\n🔹 STEP 6: Building RAG chain...")

    # -------------
    SESSION_ID = "test_session"
    qa_chain = build_chain(llm, retriever, session_id=SESSION_ID)
    print(f"RAG chain ready: {qa_chain}")
    
    # print("\n🔹 STEP 7: Running query...")
    # test_query = "What is transformer?"
    # add_to_memory(SESSION_ID, "user", test_query)
    # add_to_memory(SESSION_ID, "ai", "hello response from LLM")
    # memory = get_memory(SESSION_ID)
    # print(f"Current memory messages: {memory.messages[0].content}")
    # print(f"Current memory messages: {isinstance(memory.messages[0], HumanMessage)}")
    # history_messages = get_history(SESSION_ID)
    # print("history_messages", history_messages)
    # history_text = "\n".join([
    #         f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
    #         for msg in history_messages
    #     ])
    # print(f"Formatted history_text:\n{history_text}")
    

    result = qa_chain({"question": "what is transformer?"})
    answer = result["answer"]
    print(f"\n✅ Raw chain result: {result}")
    print(f"\n✅ FINAL ANSWER:\n{answer}")

    # print("\n📚 SOURCE DOCUMENTS:")
    # for i, doc in enumerate(result["source_documents"]):
    #     print(f"\n--- Source {i+1} ---")
    #     print(doc.page_content[:300])


if __name__ == "__main__":
    main()