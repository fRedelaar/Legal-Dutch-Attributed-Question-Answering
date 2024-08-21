import pandas as pd


def transform_found_doc_id_to_docs(relevant_docs):
    # Load the knowledge corpus CSV
    knowledge_corpus = pd.read_csv("../../datasets/knowledge_corpus/large_corpus/knowledge_corpus_large.csv")

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
