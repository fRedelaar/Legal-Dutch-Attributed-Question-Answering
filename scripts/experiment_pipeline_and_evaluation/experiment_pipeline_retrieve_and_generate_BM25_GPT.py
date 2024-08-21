import argparse
import time
import pandas as pd
import openai
from scripts.helpers.pyserini_BM25 import retrieve_docs_for_questions_BM25
from scripts.helpers.helper import load_txt_data
from scripts.helpers.TREC_EVAL_recall_at_k import evaluate_and_write_results
from API_key import SECRET_KEY, ORGANIZATION_KEY
from scripts.helpers.transform_found_docIDs_to_docs import transform_found_doc_id_to_docs

openai.api_key = SECRET_KEY
openai.organization = ORGANIZATION_KEY

def retrieve_relevant_docs(k, retriever_type):
    # Load data
    QA = pd.read_csv("../../datasets/QA_set/QA_set_102_largecorpus.csv")
    if retriever_type == "pyserini_bm25":
        relevant_docs_BM25 = retrieve_docs_for_questions_BM25(QA, k)
        print(relevant_docs_BM25)
    else:
        raise ValueError("Invalid retriever type. Supported types: pyserini_bm25")
    return relevant_docs_BM25, QA

def generate_prompt_and_model_responses(QA, triple_nested_docs, generator_type, k):
    QA['model_answer'] = ""
    QA['model_attribution'] = ""

    for index, row in QA.iterrows():
        time.sleep(10)
        question = row['question']
        prompt = load_txt_data('../../prompts/one-shot.txt')
        docs_formatted = "\n".join(
            ["\n".join(doc for doc in double_nested_docs) for double_nested_docs in triple_nested_docs[index]]
        )
        prompt += f"\nQuestion: {question}\nPotential relevant documents:\n{docs_formatted}"

        gpt_api = openai.ChatCompletion.create(
            model=generator_type,
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            max_tokens=1000,
            top_p=1
        )
        model_answer = ""
        model_attribution = ""
        for message in gpt_api['choices'][0]['message']['content'].split('\n'):
            if message.startswith("ANSWER:"):
                model_answer = message.replace("ANSWER:", "").strip()
            elif message.startswith("DOC IDS:"):
                model_attribution = message.replace("DOC IDS:", "").strip()

        QA.at[index, 'model_answer'] = model_answer
        QA.at[index, 'model_attribution'] = model_attribution

    output_filename = f"../../datasets/model_results/QA_{generator_type}_answers_k{k}_QA_5.csv"
    QA.to_csv(output_filename, index=False)
    print(f"Results written to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to retrieve and generate answers.")
    parser.add_argument("--retriever", type=str, choices=["pyserini_bm25"], default="pyserini_bm25")
    args = parser.parse_args()

    models = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo"]
    ks = [5, 10]  # Example K values

    for model in models:
        for k in ks:
            print(f"Processing with {model} for k={k}")
            relevant_docs, QA = retrieve_relevant_docs(k, retriever_type=args.retriever)
            triple_nested_docs = transform_found_doc_id_to_docs(relevant_docs)
            generate_prompt_and_model_responses(QA, triple_nested_docs, generator_type=model, k=k)
            evaluate_and_write_results(relevant_docs, QA, recall_metric=f'recall_{k}', k=k)
