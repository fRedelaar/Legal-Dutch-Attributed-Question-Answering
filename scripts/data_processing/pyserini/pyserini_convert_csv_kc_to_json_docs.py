import os
import pandas as pd
import json
import numpy as np


def convert_kc_csv_to_json(csv_file, json_dir):
    """
    Converts a knowledge corpus CSV to JSON files and stores them in a directory.

    Parameters:
        csv_file (str): Path to the CSV file.
        json_dir (str): Directory where JSON files will be stored.
    """

    # Create the directory if it does not exist
    os.makedirs(json_dir, exist_ok=True)

    # Read the CSV file into a data frame with UTF-8 encoding
    kc_df = pd.read_csv(csv_file, encoding='utf-8')

    # Convert each row in the DataFrame to a JSON file
    for index, row in kc_df.iterrows():
        # Handle empty or NaN values for additional headers
        afdeling_titel = row.get('afdeling_titel')  # Using .get() to handle KeyError
        if pd.isna(afdeling_titel) or afdeling_titel == "":
            afdeling_titel = None

        paragraaf = row.get('paragraaf')
        if pd.isna(paragraaf) or paragraaf == "":
            paragraaf = None

        paragraaf_titel = row.get('paragraaf_titel')
        if pd.isna(paragraaf_titel) or paragraaf_titel == "":
            paragraaf_titel = None

        artikel = row.get('artikel')
        if pd.isna(artikel) or artikel == "":
            artikel = None

        subparagraaf_titel = row.get('subparagraaf_titel')
        if pd.isna(subparagraaf_titel) or subparagraaf_titel == "":
            subparagraaf_titel = None

        titel_titel = row.get('titel_titel')
        if pd.isna(titel_titel) or titel_titel == "":
            titel_titel = None

        article_name = row.get('article_name')
        if pd.isna(article_name) or article_name == "":
            article_name = None

        doc = {
            'id': row['DOC_ID'],
            # 'law_id': row['law_id'],
            # 'law_name': row['law_name'],
            # 'hoofdstuk': row['hoofdstuk'],
            # 'hoofdstuk_titel': row['hoofdstuk_titel'],
            # 'afdeling': row['afdeling'],
            # 'afdeling_titel': afdeling_titel,
            # 'paragraaf': paragraaf,
            # 'paragraaf_titel': paragraaf_titel,
            # 'subparagraaf_titel': subparagraaf_titel,
            # 'titel_titel': titel_titel,
            # 'artikel': artikel,
            # 'article_name': article_name,
            'contents': row['text']
        }
        file_path = os.path.join(json_dir, f'{row["DOC_ID"]}.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False)

    print("###### DONE CONVERTING KC TO JSON DOCS! #####")

def main():
    convert_kc_csv_to_json("../../../datasets/knowledge_corpus/large_corpus/knowledge_corpus_large.csv",
                           "../../../datasets/knowledge_corpus/kc_pyserini_docs")

if __name__ == "__main__":
    main()