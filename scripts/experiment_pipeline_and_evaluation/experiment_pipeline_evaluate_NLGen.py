import pandas as pd
import json
import time
import os
import datetime
import evaluate
from evaluate import load
import openai
from API_key import SECRET_KEY, ORGANIZATION_KEY
from scripts.helpers.helper import load_txt_data

openai.api_key = SECRET_KEY
openai.organization = ORGANIZATION_KEY


def calculate_hit_rate(results):
    """
    Calculate the hit rate of the generator's attribution answers.

    Parameters:
    results (pd.DataFrame): DataFrame containing results.

    Returns:
    float: Average hit rate.
    """
    total_hit = 0
    total_queries = len(results)

    for index, row in results.iterrows():
        human_attribution = int(float(row["human_attribution"]))
        model_attribution = int(float(row["model_attribution"]))

        hit = 1 if human_attribution == model_attribution else 0
        total_hit += hit

        # Debug print to trace values
        print(f"Row {index} - Human: {human_attribution}, Model: {model_attribution}, Hit: {hit}")

    average_hit_rate = total_hit / total_queries if total_queries > 0 else 0
    return average_hit_rate

def calculate_precision_recall_generator(results):
    """
    Calculate precision and recall of the generator's attribution answers.

    Parameters:
    results (pd.DataFrame): DataFrame containing results.

    Returns:
    tuple: Average precision and recall.
    """
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for index, row in results.iterrows():
        human_attribution = int(float(row["human_attribution"]))
        model_attribution = int(float(row["model_attribution"]))

        true_positives = 1 if human_attribution == model_attribution else 0
        false_positives = 1 if model_attribution != human_attribution else 0
        false_negatives = 1 if human_attribution != model_attribution else 0

        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        # Debug print to trace values
        print(f"Row {index} - TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")

    average_precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0
    average_recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0

    return average_precision, average_recall



def calculate_mauve_rouge_meteor_bertscore_bleu(results):
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')

    reference_texts = results['human_answer'].tolist()
    generated_texts = results['model_answer'].tolist()

    rouge_results = rouge.compute(predictions=generated_texts, references=reference_texts)
    meteor_results = meteor.compute(predictions=generated_texts, references=reference_texts, beta=40, alpha=0.1)

    score_results = {
        "rouge_results": rouge_results,
        "meteor_results": round(meteor_results.get('meteor', 0), 4),
    }

    return score_results

def calculate_G_EVAL_scores(results):
    """
    Calculate the G-EVAL scores.

    Parameters:
    results (pd.DataFrame): DataFrame containing results.

    Returns:
    dict: G-EVAL scores dictionary.
    """
    COH_G_EVAL_responses = []
    CON_G_EVAL_responses = []
    FLU_G_EVAL_responses = []
    REL_G_EVAL_responses = []

    G_EVAL_PROMPT_COH = load_txt_data("../../prompts/coherence.txt")
    G_EVAL_PROMPT_CON = load_txt_data("../../prompts/consistency.txt")
    G_EVAL_PROMPT_FLU = load_txt_data("../../prompts/fluency.txt")
    G_EVAL_PROMPT_REL = load_txt_data("../../prompts/relevance.txt")

    for index, row in results.iterrows():
        print("Working currently on index:", index, " --- row: ", row)
        question = row["question"]
        human_answer = row["human_answer"]
        model_answer = row["model_answer"]

        coh_prompt = G_EVAL_PROMPT_COH + f"Model Answer: {model_answer}"
        con_prompt = G_EVAL_PROMPT_CON + f"Human Answer: {human_answer}\n Model Answer: {model_answer}"
        flu_prompt = G_EVAL_PROMPT_FLU + f"Model Answer: {model_answer}"
        rel_prompt = G_EVAL_PROMPT_REL + f"Human Answer: {question}\n Model Answer: {model_answer}"

        time.sleep(1)
        try:
            coh_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": coh_prompt}],
                temperature=0,
                max_tokens=5,
                top_p=1
            )

            time.sleep(1)
            con_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": con_prompt}],
                temperature=0,
                max_tokens=5,
                top_p=1
            )

            time.sleep(1)
            flu_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": flu_prompt}],
                temperature=0,
                max_tokens=10,
                top_p=1
            )

            time.sleep(1)
            rel_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": rel_prompt}],
                temperature=0,
                max_tokens=5,
                top_p=1
            )
        except Exception as e:
            print(f"Error encountered: {e}")
            time.sleep(10)
            continue

        try:
            COH_G_EVAL_response = int(coh_response['choices'][0]['message']['content'])
        except ValueError:
            COH_G_EVAL_response = 0  # or another default value
        COH_G_EVAL_responses.append(COH_G_EVAL_response)

        try:
            CON_G_EVAL_response = int(con_response['choices'][0]['message']['content'])
        except ValueError:
            CON_G_EVAL_response = 0  # or another default value
        CON_G_EVAL_responses.append(CON_G_EVAL_response)

        try:
            FLU_G_EVAL_response = int(flu_response['choices'][0]['message']['content'])
        except ValueError:
            FLU_G_EVAL_response = 0  # or another default value
        FLU_G_EVAL_responses.append(FLU_G_EVAL_response)

        try:
            REL_G_EVAL_response = int(rel_response['choices'][0]['message']['content'])
        except ValueError:
            REL_G_EVAL_response = 0  # or another default value
        REL_G_EVAL_responses.append(REL_G_EVAL_response)

    COH_G_EVALUATION = sum(COH_G_EVAL_responses) / len(COH_G_EVAL_responses)
    CON_G_EVALUATION = sum(CON_G_EVAL_responses) / len(CON_G_EVAL_responses)
    FLU_G_EVALUATION = sum(FLU_G_EVAL_responses) / len(FLU_G_EVAL_responses)
    REL_G_EVALUATION = sum(REL_G_EVAL_responses) / len(REL_G_EVAL_responses)

    g_eval_results = {
        "COH_G_EVALUATION": COH_G_EVALUATION,
        "CON_G_EVALUATION": CON_G_EVALUATION,
        "FLU_G_EVALUATION": FLU_G_EVALUATION,
        "REL_G_EVALUATION": REL_G_EVALUATION,
    }

    responses_df = pd.DataFrame({
        "COH_G_EVALUATION": COH_G_EVAL_responses,
        "CON_G_EVALUATION": CON_G_EVAL_responses,
        "FLU_G_EVALUATION": FLU_G_EVAL_responses,
        "REL_G_EVALUATION": REL_G_EVAL_responses,
    })

    responses_filename = f"../../experiment_results/experiment_responses_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    responses_df.to_csv(responses_filename, index=False)
    print(f"Responses saved to {responses_filename}")

    return g_eval_results

def main():
    csv_files = [
        "../../../../results/NLGenresults.csv",
    # adjust to your own filename
    ]

    for file_path in csv_files:
        results = pd.read_csv(file_path)

        # Print column names to check if 'human_attribution' exists
        print("Column names:", results.columns)

        # Proceed if 'human_attribution' exists in the DataFrame
        if 'human_attribution' in results.columns:
            average_hit_rate = calculate_hit_rate(results)
            average_precision, average_recall = calculate_precision_recall_generator(results)
            hug_scores = calculate_mauve_rouge_meteor_bertscore_bleu(results)
            scores_G_EVAL = calculate_G_EVAL_scores(results)

            to_write = {
                'average_precision': average_precision,
                'average_hit_rate': average_hit_rate,
                'average_recall': average_recall,
                'hug_scores': hug_scores,
                'scores_G_EVAL': scores_G_EVAL
            }

            filename = f"../../../../experiment_results/NLGen_dataset_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
            with open(filename, 'w') as file:
                file.write("Experiment Results:\n")
                for key, value in to_write.items():
                    file.write(f"{key}: {json.dumps(value, indent=2)}\n")

            print(f"Experiment results saved to {filename}")
        else:
            print(f"'human_attribution' column not found in {file_path}")

if __name__ == "__main__":
    main()
