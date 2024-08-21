import argparse
import time
import pandas as pd
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer, util
import openai
from scripts.helpers.TREC_EVAL_recall_at_k import evaluate_and_write_results
from scripts.helpers.helper import load_txt_data
from scripts.helpers.transform_found_docIDs_to_docs import transform_found_doc_id_to_docs
from API_key import SECRET_KEY, ORGANIZATION_KEY
import sys
import re


openai.api_key = SECRET_KEY
openai.organization = ORGANIZATION_KEY

class PrintLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


def retrieve_docs_for_questions_e5_large(qa_df, k):
    # Load the CSV file
    csv_file_path = "../../datasets/knowledge_corpus/large_corpus/knowledge_corpus_large.csv"
    df = pd.read_csv(csv_file_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Extract the text content from the DataFrame
    df['combined_text'] = df.apply(lambda row: f"{row['DOC_ID']}  {row['text']}", axis=1)
    documents = df['combined_text'].tolist()

    # Initialize the SBERT model
    model = SentenceTransformer('intfloat/multilingual-e5-large')

    # Check if embeddings file exists
    embeddings_file = 'multilingual-e5-large_no_metadata_largecorpus.npy'
    if os.path.exists(embeddings_file):
        # Load embeddings from file
        print("Embedding file found!")
        doc_embeddings = np.load(embeddings_file)
    else:
        # Encode the documents in batches
        print("No embedding file found. Creating new one.")
        batch_size = 32
        doc_embeddings = []

        for i in range(0, len(documents), batch_size):
            batch_embeddings = model.encode(documents[i:i + batch_size])
            doc_embeddings.extend(batch_embeddings)

        doc_embeddings = np.array(doc_embeddings)
        np.save(embeddings_file, doc_embeddings)

    relevant_docs = []
    for idx, row in qa_df.iterrows():
        query = row['question']
        query_embedding = model.encode(query)

        # Compute cosine similarity between query and each document
        cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        cosine_scores = cosine_scores.numpy()

        # Get the top k results
        top_results = np.argsort(cosine_scores)[-k:][::-1]
        print(top_results)
        doc_ids = [df.iloc[idx]['DOC_ID'] for idx in top_results]
        relevant_docs.append(doc_ids)

    return relevant_docs

def retrieve_relevant_docs(k):
    # Load data
    QA = pd.read_csv("../../datasets/QA_set/QA_set_102_largecorpus.csv")
    relevant_docs = retrieve_docs_for_questions_e5_large(QA, k)
    print(relevant_docs)
    return relevant_docs, QA

def generate_prompt_and_model_responses(QA, triple_nested_docs, generator_type, k, run_number):
    QA['model_answer'] = ""
    QA['model_attribution'] = ""

    for index, row in QA.iterrows():
        time.sleep(1)
        question = row['question']
        prompt = load_txt_data('../../prompts/one-shot-short.txt')
        docs_formatted = "\n".join(
            ["\n".join(doc for doc in double_nested_docs) for double_nested_docs in triple_nested_docs[index]]
        )
        prompt += f"\nQuestion: {question}\nPotential relevant documents:\n{docs_formatted}"

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Question: ", question)

        while True:
            try:
                gpt_api = openai.ChatCompletion.create(
                    model=generator_type,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0,
                    max_tokens=1000,
                    top_p=1
                )
                break
            except Exception as e:
                print(f"API call failed: {e}. Retrying in 10 seconds...")
                time.sleep(10)

        answer = gpt_api['choices'][0]['message']['content']
        answer = answer.replace("\n", " ")

        print("Answer: ", answer)

        # Extract model answer and DOC IDS using regex
        answer_pattern = re.compile(r'ANSWER:\s*(.*?)(?:DOC IDS:|$)')
        doc_ids_pattern = re.compile(r'DOC IDS:\s*(.*)')

        model_answer_match = answer_pattern.search(answer)
        doc_ids_match = doc_ids_pattern.search(answer)

        model_answer = model_answer_match.group(1).strip() if model_answer_match else ""
        model_attribution = doc_ids_match.group(1).strip() if doc_ids_match else ""

        QA.at[index, 'model_answer'] = model_answer
        QA.at[index, 'model_attribution'] = model_attribution

    output_filename = f"../../datasets/model_results/BIG_K3_QA_{generator_type}_answers_k{k}_set_run_{run_number}.csv"
    QA.to_csv(output_filename, index=False)
    print(f"Results written to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to retrieve and generate answers.")
    parser.add_argument("--k", type=int, default=3, help="Number of documents to retrieve per query.")
    args = parser.parse_args()

    models = ["gpt-4o"]

    # models = ["gpt-4o", "gpt-4-turbo-2024-04-09", "gpt-4-0613", "gpt-3.5-turbo-0125"]

    ks = [3]

    for model in models:
        for k in ks:
            for run in range(1, 11):
                # Create logger for each run, for post analysis
                now = time.localtime()
                date_time_str = time.strftime("%d_%m_%Y_%H_%M_%S", now)
                filename = f"SHORT_PROMPT_k3_LOGGER_RETRIEVEDBY_me5large_only_text_LLM_{model}_run_{run}_DATE_{date_time_str}.txt"
                sys.stdout = PrintLogger(filename)

                print(f"Processing run {run} with {model} for k={k}, using multilingual-e5-large retriever")
                relevant_docs, QA = retrieve_relevant_docs(k)
                triple_nested_docs = transform_found_doc_id_to_docs(relevant_docs)
                generate_prompt_and_model_responses(QA, triple_nested_docs, generator_type=model, k=k, run_number=run)
                evaluate_and_write_results(relevant_docs, QA, recall_metric=f'recall_{k}', k=k, modelname={model})

                # Close the logger for the current run
                sys.stdout.log.close()
                sys.stdout = sys.__stdout__