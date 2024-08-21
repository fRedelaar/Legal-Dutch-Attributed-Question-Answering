import argparse
import pandas as pd
from scripts.helpers.pyserini_BM25 import retrieve_docs_for_questions_BM25
from scripts.helpers.TREC_EVAL_recall_at_k import evaluate_and_write_results


def retrieve_relevant_docs(k, retriever_type, QA):
    """
    Retrieve relevant documents based on the provided settings.

    Parameters:
    k (int): Number of documents to retrieve.
    retriever_type (str): Type of retriever to use.
    QA (pd.DataFrame): DataFrame containing the questions.

    Returns:
    Tuple of relevant documents and the QA DataFrame.
    """
    if retriever_type == "pyserini_bm25":
        relevant_docs = retrieve_docs_for_questions_BM25(QA, k)
        print(relevant_docs)
    else:
        raise ValueError(f"Invalid retriever type: {retriever_type}. Supported types: pyserini_bm25")

    return relevant_docs, QA

def main():
    parser = argparse.ArgumentParser(description="Script to retrieve and generate answers.")
    parser.add_argument("--retriever", type=str, default="pyserini_bm25",
                        help="Select the document retriever to use. Currently supports 'pyserini_bm25'.")
    args = parser.parse_args()

    configurations = [
        {"k": 3, "retriever_type": args.retriever, "recall_metric": "recall_3"},
        {"k": 5, "retriever_type": args.retriever, "recall_metric": "recall_5"},
        {"k": 10, "retriever_type": args.retriever, "recall_metric": "recall_10"},
        # More configurations can be added here
    ]

    # Load data only once
    QA = pd.read_csv("../../datasets/QA_set/QA_set_102_largecorpus.csv")

    for config in configurations:
        print(f"Results {config['retriever_type']}, using {config['recall_metric']}")
        relevant_docs, QA_loaded = retrieve_relevant_docs(k=config['k'], retriever_type=config['retriever_type'], QA=QA)
        evaluate_and_write_results(relevant_docs, QA_loaded, recall_metric=config['recall_metric'], k=config['k'], modelname="BM25")

if __name__ == "__main__":
    main()
