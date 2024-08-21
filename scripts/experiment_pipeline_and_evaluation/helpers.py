import pandas as pd
from pyserini.search.lucene import LuceneSearcher
import pandas as pd
import pytrec_eval
import datetime


def load_txt_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


def calculate_hit_rate(retrieved_docs, ground_truth_docs):
    return int(any(doc in ground_truth_docs for doc in retrieved_docs))

def evaluate_and_write_results(relevant_docs, QA, recall_metric, k):
    # Prepare ground truth and runs for pytrec_eval
    gt = {}
    run = {}
    hit_rate_scores = []

    for index, row in QA.iterrows():
        qid = str(row['question_id'])
        ground_truth_docs = [doc.strip() for doc in row['human_attribution'].split(',')]
        gt[qid] = {doc: 1 for doc in ground_truth_docs}

        retrieved_docs = [doc.strip() for doc in relevant_docs[index]]
        run[qid] = {doc: 1 if doc in ground_truth_docs else 0 for doc in retrieved_docs}

        # Calculate hit rate
        hit_rate = calculate_hit_rate(retrieved_docs[:k], ground_truth_docs)
        hit_rate_scores.append(hit_rate)

        print(f"Query {qid} - Ground Truth Docs: {ground_truth_docs}, Retrieved Docs: {retrieved_docs}, Hit Rate@{k}: {hit_rate}")

    # Evaluate using pytrec_eval with dynamic recall metric
    evaluator = pytrec_eval.RelevanceEvaluator(gt, {recall_metric})
    results = evaluator.evaluate(run)

    print(f"{recall_metric} results of retriever:", results)

    # Convert the results dictionary to a list of dictionaries for easier DataFrame creation
    data_for_df = [{'question_id': qid, recall_metric: scores[recall_metric], f'hitrate_{k}': hit_rate_scores[i]} for i, (qid, scores) in enumerate(results.items())]

    results_df = pd.DataFrame(data_for_df)
    average_recall = results_df[recall_metric].mean()
    average_hit_rate = results_df[f'hitrate_{k}'].mean()
    print(f"Average {recall_metric}: {average_recall}")
    print(f"Average Hit Rate@{k}: {average_hit_rate}")

    # Append a row for the average recall and average hit rate
    avg_recall_hitrate_row = pd.DataFrame([{'question_id': 'Average', recall_metric: average_recall, f'hitrate_{k}': average_hit_rate}])
    results_df = pd.concat([results_df, avg_recall_hitrate_row], ignore_index=True)

    file_name = f"../../experiment_results/Retriever_TREC_eval_{recall_metric}_hitrate_{k}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_results.csv"
    results_df.to_csv(file_name, index=False)
    print("Results written to a CSV file:", file_name)

def retrieve_docs_for_questions_BM25(qa_df, k):

    # Initialize searcher
    searcher = LuceneSearcher('../../datasets/knowledge_corpus/kc_pyserini_index')


    # Set the BM25 parameters
    searcher.set_bm25(1.2, 0.75)

    relevant_docs = []
    for idx, row in qa_df.iterrows():
        query = row['question']
        hits = searcher.search(query, k=k)
        doc_ids = [hit.docid for hit in hits]
        relevant_docs.append(doc_ids)

    return relevant_docs



def transform_found_doc_id_to_docs(relevant_docs):
    # Load the knowledge corpus CSV
    knowledge_corpus = pd.read_csv("../../datasets/knowledge_corpus/knowledge_corpus/knowledge_corpus.csv")


    # Initialize an empty list to store transformed documents
    triple_nested_docs = []

    # Iterate through relevant_docs and fetch corresponding information from the knowledge corpus
    for doc_ids in relevant_docs:
        double_nested_docs = []
        for doc_id in doc_ids:
            # Fetch row from knowledge corpus based on DOC_ID
            doc_row = knowledge_corpus[knowledge_corpus['DOC_ID'] == doc_id].iloc[0]

            # Concatenate DOC_ID, Law_name, hoofdstuk, and text
            doc_text = f"DOC_ID: {doc_row['DOC_ID']}, law_name: {doc_row['law_name']}, " \
                       f"hoofdstuk: {doc_row['hoofdstuk']}, hoofdstuk_titel: {doc_row['hoofdstuk_titel']}, " \
                       f"afdeling: {doc_row['afdeling']}, afdeling_titel: {doc_row['afdeling_titel']} " \
                       f"paragraaf: {doc_row['paragraaf']}, paragraaf_titel: {doc_row['paragraaf_titel']} " \
                       f"artikel: {doc_row['artikel']}, text: {doc_row['text']} "

            # Append to double_nested_docs list
            double_nested_docs.append([doc_text])

        # Append double_nested_docs list to triple_nested_docs
        triple_nested_docs.append(double_nested_docs)

    # print(triple_nested_docs)
    return triple_nested_docs
