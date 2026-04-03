from langchain_core.prompts import PromptTemplate
from src.memory import add_to_memory, get_history
from langchain_core.messages import HumanMessage


def build_chain(llm, retriever, session_id="default"):

    prompt = PromptTemplate.from_template("""
    You are a knowledgeable and professional AI assistant.

    Follow these guidelines while answering:
    - Provide clear, structured, and well-explained answers
    - Use bullet points where helpful
    - Keep explanations concise but informative
    - Maintain a professional and helpful tone
    - If referencing previous conversation, connect it naturally
    - Do NOT make up information if context is insufficient

    Use the following information to answer the question:

    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer:
    - Start with a brief introduction
    - Then provide a structured explanation (use bullet points if needed)
    - End with a helpful closing line
    """)

    def qa_chain(inputs):
        # Step 1: get data
        question = inputs["question"]
        print(f"Received question: '{question}' for session_id: '{session_id}'")
        # Step 2: retrieve docs
        docs = retriever.invoke(question)
        print(f"Retrieved {len(docs)} documents for question: '{question}'")
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 3: prepare memory text
        history_messages = get_history(session_id)
        print("history_messages", history_messages)

        history_text = "\n".join([
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in history_messages
        ])

        print(f"Formatted history_text:\n{history_text}")


        # Step 4: format prompt
        formatted_prompt = prompt.format(
            question=question,
            context=context,
            chat_history=history_text
        )
        print(f"Formatted prompt for LLM:\n{formatted_prompt}")
        # Step 5: call LLM
        response = llm.invoke(formatted_prompt)
        response_text= response.content
        print(f"\n✅Raw LLM response: '{response}' with type {type(response)}")
        print(f"LLM response(response_text): '{response_text}'")
        # Step 6: save memory
        add_to_memory(session_id, "user", question)
        add_to_memory(session_id, "ai", response_text)

        return {
            "answer": response_text,
            "source_documents": docs
        }

    return qa_chain