from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
)
from langchain_community.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper

llm = ChatOllama(
    model="llama3",
    temperature=0
)

ragas_llm = LangchainLLMWrapper(llm)

questions = [
    "What is the transformer architecture?",
]

def run_ragas_evaluation(results):
    """
    results = [
        {
            "question": "",
            "answer": "",
            "contexts": [],
            "ground_truth": ""
        }
    ]
    """

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    print("\nRunning RAGAS Evaluation...\n")

    scores = evaluate(
        dataset,
        metrics=[context_precision],
        llm=ragas_llm 
    )

    print("\n📊 RAGAS Scores:\n")
    print(scores)

    return scores
# Method to run the evaluation
def  run_evaluation(qa_chain):
    results = []

    for q in questions:
        print("\n" + "="*50)
        print("QUESTION:", q)

        result = qa_chain({"question": q})

        answer = result["answer"]
        contexts = [doc.page_content for doc in result["source_documents"]]

        print("\nANSWER:\n", answer)

        results.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": answer
        })

    return run_ragas_evaluation(results)