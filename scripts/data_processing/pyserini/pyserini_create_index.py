import subprocess
import sys

def create_pyserini_index(input_dir, index_dir):
    """
    Creates an index using Pyserini.

    Parameters:
        input_dir (str): Directory containing the JSON files.
        index_dir (str): Directory where the index will be stored.
    """
    python_executable = sys.executable  # Get path to the Python interpreter
    command = [
        python_executable, '-m', 'pyserini.index',
        '-collection', 'JsonCollection',
        '-generator', 'DefaultLuceneDocumentGenerator',
        '-threads', '4',
        '-input', input_dir,
        '-index', index_dir,
        '-storePositions',
        '-storeDocvectors',
        '-storeRaw'
    ]
    subprocess.run(command)
    print("###### DONE INDEXING JSON DOCS! #####")

def main():
    create_pyserini_index('../../../datasets/knowledge_corpus/kc_pyserini_docs',
                          '../../../datasets/knowledge_corpus/kc_pyserini_index')

if __name__ == "__main__":
    main()