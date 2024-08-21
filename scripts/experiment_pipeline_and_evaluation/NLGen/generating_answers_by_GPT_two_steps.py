from API_key import SECRET_KEY, ORGANIZATION_KEY
import openai
import csv
import time
from datasets import load_dataset
from tqdm import tqdm

openai.api_key = SECRET_KEY
openai.organization = ORGANIZATION_KEY

# Load the dataset
ds = load_dataset("din0s/msmarco-nlgen")['dev']

# Define the number of rows to process
num_rows = 1000

# Prepare the data to store in the CSV
data = []

# Preprocess the data and create the prompts
for i in tqdm(range(num_rows), desc="Processing requests"):
    example = ds[i]
    query = example['query']
    passages = example['passages']

    # Ensure we have up to 10 passages, pad with empty if fewer
    passages = passages[:10] + [{'passage_text': 'N/A'}] * (10 - len(passages))

    # Create a numbered list of passages
    passage_texts = "\n".join([f"{index + 1}. {p['passage_text']}" for index, p in enumerate(passages)])

    # Create the instruction prompt for selecting the most relevant passage
    prompt_selection = f"""
    Given the following query and passages, select the most relevant passage number. Only base your decision on one passage.

    Query: {query}

    Passages:
    {passage_texts}

    Your response should be in the format: 'Selected Passage: [Passage number here]'. Make sure to always write only a number!
    """

    # Attempt to make the first API call for selecting the most relevant passage
    while True:
        try:
            response_selection = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt_selection}],
                temperature=0.3,
                max_tokens=50,
                top_p=1
            )
            response_content = response_selection.choices[0].message['content'].strip()
            print(f"Response Content: {response_content}")  # Debug print
            if "Selected Passage:" in response_content:
                selected_passage_number = response_content.split("Selected Passage:")[1].strip()
                break
            elif response_content.isdigit():
                selected_passage_number = response_content
                break
            else:
                raise ValueError("Response does not contain 'Selected Passage:' or a valid passage number")
        except (openai.error.OpenAIError, ValueError, IndexError) as e:
            print(f"Error: {e}. Retrying in 10 seconds...")
            time.sleep(10)

    # Create the instruction prompt for generating the answer based on the selected passage
    selected_passage_text = passages[int(selected_passage_number) - 1]['passage_text']

    prompt_answer = f"""
    Given the following query and the most relevant passage, generate an answer.

    Query: {query}

    Passage: {selected_passage_text}

    Your answer should be in the format: 'Answer: [Your answer here]'.
    """

    # Attempt to make the second API call for generating the answer
    while True:
        try:
            response_answer = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt_answer}],
                temperature=0.3,
                max_tokens=300,
                top_p=1
            )
            answer = response_answer.choices[0].message['content']
            break
        except openai.error.OpenAIError as e:
            print(f"Error: {e}. Retrying in 10 seconds...")
            time.sleep(10)

    # Extract the answer text
    try:
        answer_text = answer.split("Answer:")[1].strip()
    except IndexError:
        answer_text = "N/A"

    # Add the answer and selected passage number to the data list
    data.append([answer_text, selected_passage_number])

# Define the CSV file name
csv_file = "gpt_answers.csv"

# Write the data to the CSV file
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["answer", "passage"])
    writer.writerows(data)

print(f"Data has been written to {csv_file}")
