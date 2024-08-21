import os
import pandas as pd

def read_law_articles(directory):
    """
    Read law articles from all CSV files in a directory.

    Parameters:
    directory (str): Directory containing CSV files.

    Returns:
    pd.DataFrame: Concatenated DataFrame.
    """
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    dfs = [pd.read_csv(path) for path in file_paths]
    return pd.concat(dfs, ignore_index=True)

def split_long_text(row, max_words=150):
    """
    Split long article into smaller chunks of articles.

    Parameters:
    row (pd.Series): Row of a DataFrame.
    max_words (int): Maximum number of words per chunk. Default is 150.

    Returns:
    pd.DataFrame: DataFrame with split articles.
    """
    text = row['text']
    if pd.notna(text) and len(text.split()) > max_words:
        words = text.split()
        split_texts = [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
        return pd.concat([row.to_frame().T.assign(text=split_text) for split_text in split_texts], ignore_index=True)
    else:
        return pd.DataFrame([row])

def lowercase_text(row):
    row['text'] = row['text'].lower()
    return row

def main():
    input_directory = '../../datasets/knowledge_corpus/large_corpus/law_articles_CSV_format/'
    output_file = '../../datasets/knowledge_corpus/large_corpus/knowledge_corpus_large.csv'

    # Read the CSV files
    law_articles = read_law_articles(input_directory)

    # Apply the function to each row and concatenate the results
    law_articles_split = pd.concat(law_articles.apply(split_long_text, axis=1).tolist(), ignore_index=True)

    # Lowercase the text in the 'text' column
    law_articles_split = law_articles_split.apply(lowercase_text, axis=1)

    # Add a new column for IDs
    law_articles_split['DOC_ID'] = ['DOC{:04d}'.format(i) for i in range(1, len(law_articles_split) + 1)]

    # Reorder the columns
    reordered_columns = ['DOC_ID', 'law_id', 'law_name', 'hoofdstuk', 'hoofdstuk_titel', 'afdeling', 'afdeling_titel', 'paragraaf', 'paragraaf_titel', 'subparagraaf_titel', 'titel_titel', 'artikel', 'article_name', 'text']
    law_articles_split = law_articles_split[reordered_columns]

    # Save the concatenated dataframe with split text to a new CSV file
    law_articles_split.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
