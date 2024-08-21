import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import torch
import os
import scipy
from scipy.spatial.distance import cosine
from typing import List
from tqdm import tqdm
from scripts.helpers.TREC_EVAL_recall_at_k import evaluate_and_write_results

def set_cache_folder(folder: str):
    os.makedirs(folder, exist_ok=True)

def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]

def get_splade_embeddings(docs: List[str], tokenizer, model, device: str) -> np.ndarray:
    tokens = tokenizer(docs, return_tensors='pt', padding=True, truncation=True).to(device)
    output = model(**tokens)
    vecs = torch.max(torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1)[0].squeeze().detach().cpu().numpy()
    del output
    del tokens
    if device != 'cpu':
        torch.cuda.synchronize()
    return vecs

def retrieve_docs_for_questions_splade(qa_df, k, csv_file_path, model_id, cache_folder, embeddings_file):
    df = pd.read_csv(csv_file_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df['combined_text'] = df.apply(lambda row: f"{row['DOC_ID']} {row['text']}", axis=1)
    documents = df['combined_text'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)

    if os.path.exists(embeddings_file):
        print("Embedding file found!")
        doc_embeddings = scipy.sparse.load_npz(embeddings_file)
    else:
        print("No embedding file found. Creating new one.")
        vecs = []
        for chunk in tqdm(split(documents, chunk_size=5)):
            vecs.append(get_splade_embeddings(chunk, tokenizer, model, device))
        embeddings = np.vstack(vecs)
        doc_embeddings = scipy.sparse.csr_matrix(embeddings)
        scipy.sparse.save_npz(embeddings_file, doc_embeddings)

    relevant_docs = []
    for idx, row in qa_df.iterrows():
        query = row['question']
        query_embedding = get_splade_embeddings([query], tokenizer, model, device)

        l2_norm_matrix = scipy.sparse.linalg.norm(doc_embeddings, axis=1)
        l2_norm_query = scipy.linalg.norm(query_embedding)
        cosine_similarity = doc_embeddings.dot(query_embedding.T) / (l2_norm_matrix * l2_norm_query)

        top_results = np.argsort(cosine_similarity)[-k:][::-1]
        doc_ids = [df.iloc[idx]['DOC_ID'] for idx in top_results]
        relevant_docs.append(doc_ids)

    return relevant_docs

def retrieve_relevant_docs(k, QA, csv_file_path, model_id, cache_folder, embeddings_file):
    relevant_docs = retrieve_docs_for_questions_splade(QA, k, csv_file_path, model_id, cache_folder, embeddings_file)
    return relevant_docs, QA

def main():
    parser = argparse.ArgumentParser(description="Script to retrieve and generate answers.")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve per query.")
    args = parser.parse_args()

    configurations = [
        # {"k": 3, "recall_metric": "recall_3"},
        # {"k": 5, "recall_metric": "recall_5"},
        {"k": 10, "recall_metric": "recall_10"},
    ]

    QA = pd.read_csv("../../../../datasets/QA_set/QA_set_102_largecorpus.csv")

    results_dir = "../../../../experiment_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    cache_folder = "/storage/llm/cache"
    set_cache_folder(cache_folder)

    csv_file_path = "../../../../datasets/knowledge_corpus/large_corpus/knowledge_corpus_large.csv"
    model_id = 'naver/splade-cocondenser-ensembledistil'
    embeddings_file = 'splade_embeddings_no_metadata_largecorpus.npz'

    for config in configurations:
        print(f"Results using {config['recall_metric']}")
        relevant_docs, QA_loaded = retrieve_relevant_docs(k=config['k'], QA=QA, csv_file_path=csv_file_path, model_id=model_id, cache_folder=cache_folder, embeddings_file=embeddings_file)
        evaluate_and_write_results(relevant_docs, QA_loaded, recall_metric=config['recall_metric'], k=config['k'], modelname="splade-cocondenser-ensembledistil")

if __name__ == "__main__":
    main()
