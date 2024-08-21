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

    # Create the instruction prompt
    prompt = f"""
    You will be given a task to generate an answer on the most relevant passage. First an example follows.

    Example Query: what is a omurice omelet

    Passages:
    1. In one common rendition, the rice is fried with ketchup, and extra ketchup is squeezed on top as a garnish. In another popular one, seen in the Kyoto video, the chef uses demi-glace (a rich, veal stock–based sauce) to both fry the rice and top the omelette. Japanese mayo is often also squeezed on top.
    2. 오무라이스. 1  Omurice is a contemporary Asian dish consisting of an omelette made with fried rice. Its name derives from the combination of the English words omelette and rice. 2  A relatively simple dish, it typically calls for rice fried with ketchup, chicken and onions wrapped in a thin sheet of fried egg.
    3. In cuisine, an omelette or omelet is a dish made from beaten eggs quickly fried with butter or oil in a frying pan (without stirring as in scrambled egg). It is quite common for the omelette to be folded around a filling such as cheese, chives, vegetables, meat (often ham or bacon), or some combination of the above. Whole eggs or sometimes only egg whites are beaten with a small amount of milk or cream, or even water.
    4. How to Make Omelet Rice (Omurice) -Stir frying rice (You will need a wok) Dice carrot, onion, red capsicums, ham and crab stick. Mix the tomato sauce (3 tbsp) and worcestershire sauce in a bowl. Pre heat the wok on high heat for 10 seconds and add some oil. Add all diced ingredients and saute for 1 minute. Reduce the heat to half. Add the steamed rice and the mixed sauce.
    5. For those unfamiliar with omurice, it's a Japanese invention that combines an omelette with fried rice. You'll often hear it referred to as omuraisu (a contraction of the words omuretsu and raisu, the Japanese pronunciations of omelette and rice), or omumeshi, which fully translates rice into Japanese.
    6. In it, a chef in Kyoto makes a plate of omurice with a deftness and perfection of technique that may be unrivaled. He starts by frying rice in a carbon steel skillet, tossing it every which way until each grain is coated in a sheen of demi-glace and oil.
    7. Omurice is a contemporary Asian dish consisting of an omelette made with fried rice. Its name derives from the combination of the English words omelette and rice. Omurice is said to have originated from Japan and it became a popular dish at a western-style restaurant in Tokyo's Ginza district around the turn of the 19th century.
    8. Recipe 16 - Omurice. Today in Moto's Kitchen we're going to learn how to make Omurice! This popular dish, notorious in maid cafe's, is an interesting and delicious take on the western omelette. With a base of fried rice, chicken and ketchup, the dish is topped with a simple egg omelette. Eat this tasty dish anytime throughout the day!
    9. Another way to change this up is to top the finished omurice with Hayashi sauce or Japanese curry. Omurice (オムライス)With a fluffy omelette covering a bed of savory sweet chicken fried rice, omurice (オムライス) is a modern Japanese classic that kids love.Marc Matsumoto.
    10. A cut-open omurice with ketchup. Omurice or omu-rice (オムライス, Omu-raisu) is an example of yōshoku (a Western-influenced style of Japanese cuisine) consisting of an omelette made with fried rice and usually topped with ketchup. With omu and raisu being contractions of the words omelette and rice, the name is an example of wasei-eigo.

    Answer: An omurice omelet is a contemporary Asian dish consisting of an omelette made with fried rice.
    Selected Passage: 7

    Given the following query and passages, generate an answer based on the most relevant passage. Only base your answer on one passage.

    Query: {query}

    Passages:
    {passage_texts}

    Your answer should be in the format: 'Answer: [Your answer here]
    Selected Passage: [Passage number here]'. Make sure to always write only a number!
    """

    # print(prompt)

    # Attempt to make the API call, retrying if there's an error
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                top_p=1
            )
            answer = response.choices[0].message['content']
            break
        except openai.error.OpenAIError as e:
            print(f"Error: {e}. Retrying in 10 seconds...")
            time.sleep(10)

    # Extract the answer and selected passage number
    try:
        answer_text = answer.split("Answer:")[1].split("\nSelected Passage:")[0].strip()
        selected_passage = answer.split("Selected Passage:")[1].strip()
    except IndexError:
        answer_text = "N/A"
        selected_passage = "N/A"

    # Add the answer and selected passage to the data list
    data.append([answer_text, selected_passage])

# Define the CSV file name
csv_file = "gpt_answers.csv"

# Write the data to the CSV file
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["answer", "passage"])
    writer.writerows(data)

print(f"Data has been written to {csv_file}")
