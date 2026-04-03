# from ragas import evaluate
# from datasets import Dataset

# def evaluate_model(questions, answers, contexts):

#     dataset = Dataset.from_dict({
#         "question": questions,
#         "answer": answers,
#         "contexts": contexts,
#         "ground_truth": answers  # basic version
#     })

#     result = evaluate(dataset)
#     return result