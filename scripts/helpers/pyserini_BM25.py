import pandas as pd
from pyserini.search.lucene import LuceneSearcher


def retrieve_docs_for_questions_BM25(qa_df, k):

    # Initialize searcher
    searcher = LuceneSearcher('../../datasets/knowledge_corpus/kc_pyserini_index')


    # Set the BM25 parameters, if needed (the default might work well)
    searcher.set_bm25(1.2, 0.75)

    relevant_docs = []
    for idx, row in qa_df.iterrows():
        query = row['question']
        hits = searcher.search(query, k=k)
        doc_ids = [hit.docid for hit in hits]
        relevant_docs.append(doc_ids)

    return relevant_docs

