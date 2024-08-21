import pandas as pd
import pytrec_eval
import datetime

def calculate_hit_rate(retrieved_docs, ground_truth_docs):
    return int(any(doc in ground_truth_docs for doc in retrieved_docs))

def evaluate_and_write_results(relevant_docs, QA, recall_metric, k, modelname):
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

    print("@@@@@@@@@@@@@@@@@@@@")
    print(modelname)
    print("@@@@@@@@@@@@@@@@@@@@")


    file_name = f"../../experiment_results/Retriever_{modelname}_{recall_metric}_{k}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_results.csv"
    results_df.to_csv(file_name, index=False)
    print("Results written to a CSV file:", file_name)