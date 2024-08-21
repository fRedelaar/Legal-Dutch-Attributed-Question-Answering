import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import numpy as np
import torch
import os
from scipy.spatial.distance import cosine
from scripts.helpers.TREC_EVAL_recall_at_k import evaluate_and_write_results

def retrieve_docs_for_questions_legal_bert(qa_df, k):
    # Load the CSV file
    csv_file_path = "../../../../datasets/knowledge_corpus/large_corpus/knowledge_corpus_large.csv"
    df = pd.read_csv(csv_file_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Extract the text content from the DataFrame
    df['combined_text'] = df.apply(lambda row: f"{row['DOC_ID']} {row['text']}", axis=1)
    documents = df['combined_text'].tolist()

    # Initialize the Legal BERT model and tokenizer
    model_id = 'Gerwin/legal-bert-dutch-english'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)

    # Use a pipeline for feature extraction
    pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=0 if device == 'cuda' else -1)

    # Check if embeddings file exists
    embeddings_file = 'legal_bert_embeddings_nometadata_largecorpus.npy'
    if os.path.exists(embeddings_file):
        # Load embeddings from file
        print("Embedding file found!")
        doc_embeddings = np.load(embeddings_file)
    else:
        # Encode the documents
        print("No embedding file found. Creating new one.")
        doc_embeddings = [np.mean(pipe(doc)[0], axis=0) for doc in documents]
        doc_embeddings = np.array(doc_embeddings)
        np.save(embeddings_file, doc_embeddings)

    relevant_docs = []
    for idx, row in qa_df.iterrows():
        query = row['question']
        query_embedding = np.mean(pipe(query)[0], axis=0)

        # Compute cosine similarity between query and each document
        cosine_scores = np.array([1 - cosine(query_embedding, doc_embedding) for doc_embedding in doc_embeddings])

        # Get the top k results
        top_results = np.argsort(cosine_scores)[-k:][::-1]
        doc_ids = [df.iloc[idx]['DOC_ID'] for idx in top_results]
        relevant_docs.append(doc_ids)

    return relevant_docs

def retrieve_relevant_docs(k, QA):
    relevant_docs = retrieve_docs_for_questions_legal_bert(QA, k)
    print(relevant_docs)
    return relevant_docs, QA

def main():
    parser = argparse.ArgumentParser(description="Script to retrieve and generate answers.")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve per query.")
    args = parser.parse_args()

    configurations = [
        {"k": 5, "recall_metric": "recall_5"},
        # {"k": 10, "recall_metric": "recall_10"},
    ]

    # Load data only once
    QA = pd.read_csv("../../../../datasets/QA_set/QA_set_102_largecorpus.csv")

    results_dir = "../../../../experiment_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for config in configurations:
        print(f"Results using {config['recall_metric']}")
        relevant_docs, QA_loaded = retrieve_relevant_docs(k=config['k'], QA=QA)
        evaluate_and_write_results(relevant_docs, QA_loaded, recall_metric=config['recall_metric'], k=config['k'], modelname="legal-bert-dutch")

if __name__ == "__main__":
    main()
