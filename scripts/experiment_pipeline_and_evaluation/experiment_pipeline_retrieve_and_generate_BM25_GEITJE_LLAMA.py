import argparse
import time
import pandas as pd
import torch
from transformers import pipeline, Conversation
from helpers import retrieve_docs_for_questions_BM25
from helpers import load_txt_data
from helpers import evaluate_and_write_results
from helpers import transform_found_doc_id_to_docs
import re
import sys


class PrintLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

now = time.localtime()
date_time_str = time.strftime("%d_%m_%Y_%H_%M_%S", now)

# Boolean to switch between models
GEITJE = True

# Dynamically set the model name and output filename based on the GEITJE variable
model_name = "BramVanroy/GEITje-7B-ultra" if GEITJE else "ReBatch/Llama-3-8B-dutch"
output_filename_suffix = "GEITJE" if GEITJE else "LLAMA"
filename = f"output-{output_filename_suffix}-max_length5000_temp0.2_QAv2.0_{date_time_str}.txt"
sys.stdout = PrintLogger(filename)

# Device configuration
device = torch.device("cuda")

# Model parameters
model_kwargs = {
    "load_in_8bit": True,
}
max_length = 5000
temperature = 0.2

# Load the model
chatbot = pipeline("conversational", model=model_name, model_kwargs=model_kwargs, device_map="auto",
                   max_length=max_length, temperature=temperature)


def retrieve_relevant_docs(k, retriever_type):
    # Load data
    QA = pd.read_csv("../../datasets/QA_set/QA_set_102_largecorpus.csv")

    if retriever_type == "pyserini_bm25":
        relevant_docs_BM25 = retrieve_docs_for_questions_BM25(QA, k)
        print(relevant_docs_BM25)
    else:
        raise ValueError("Invalid retriever type. Supported types: pyserini_bm25")
    return relevant_docs_BM25, QA


def generate_prompt_and_model_responses(QA, triple_nested_docs, k):
    QA['model_answer'] = ""
    QA['model_attribution'] = ""

    for index, row in QA.iterrows():
        question = row['question']

        prompt = load_txt_data('../../prompts/one-shot-NL.txt')

        docs_formatted = "\n".join(
            ["\n".join(doc for doc in double_nested_docs) for double_nested_docs in triple_nested_docs[index]]
        )
        prompt += f"\nQuestion: {question}\nPotential relevant documents:\n{docs_formatted}"


        print("@@@@@@@@@@@@@@@@@@@@@@@@ QUESTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(question)

        print("@@@@@@@@@@@@@@@@@@@@@@@@ RELEVANT DOCS @@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(docs_formatted)

        start_messages = [
            {"role": "system",
             "content": "Je bent een chatbot genaamd Henk en een juridisch assistent. Jouw taak wordt straks zeer precies uitgelegd."},
            {"role": "user", "content": prompt}
        ]

        conversation = Conversation(start_messages)
        conversation = chatbot(conversation)
        response = conversation.messages[-1]["content"]
        print("@@@@@@@@@@@@@@@@@@@@@@@@ RESPONSE @@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(response)

        model_answer = ""
        model_attribution = ""

        answer_started = False
        for line in response.split('\n'):
            if line.startswith("ANTWOORD:"):
                model_answer = line.replace("ANTWOORD:", "").strip()
                answer_started = True
            elif line.startswith("Antwoord:"):
                model_answer = line.replace("Antwoord:", "").strip()
                answer_started = True
            elif answer_started:
                if line.startswith("DOC IDS:"):
                    break
                model_answer += "\n" + line.strip()

        # Extract all mentions of DOC followed by four digits
        model_attribution = ", ".join(re.findall(r"DOC\d{4}", response))

        QA.at[index, 'model_answer'] = model_answer
        QA.at[index, 'model_attribution'] = model_attribution

    output_filename = f"../../datasets/model_results/QA_{output_filename_suffix}_answers_k{k}_QA_v2_5.csv"

    QA.to_csv(output_filename, index=False)
    print(f"Results written to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to retrieve and generate answers.")
    parser.add_argument("--retriever", type=str, choices=["pyserini_bm25"], default="pyserini_bm25")
    args = parser.parse_args()

    ks = [5]  # Example K values

    for k in ks:
        print(f"Processing for k={k}")
        relevant_docs, QA = retrieve_relevant_docs(k, retriever_type=args.retriever)
        triple_nested_docs = transform_found_doc_id_to_docs(relevant_docs)
        generate_prompt_and_model_responses(QA, triple_nested_docs, k=k)
        evaluate_and_write_results(relevant_docs, QA, recall_metric=f'recall_{k}', k=k)

sys.stdout.log.close()
