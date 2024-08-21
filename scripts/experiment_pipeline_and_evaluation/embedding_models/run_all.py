import subprocess
import os

# Define the subfolders where the scripts are located
subfolders = ['dragon', 'featureExtraction', 'SentenceTransformers']

# List of Python files to run sequentially with their respective folders
files_to_run = [
    ('SentenceTransformers', 'distiluse-base-multilingual-cased-v2.py'),
    ('SentenceTransformers', 'paraphrase-multilingual-MiniLM-L12-v2.py'),
    ('SentenceTransformers', 'multilingual-e5-small.py'),
    ('SentenceTransformers', 'multilingual-e5-base.py'),
    ('SentenceTransformers', 'multilingual-e5-large.py'),
    ('SentenceTransformers', 'allnli-GroNLP-bert-base-dutch-cased.py'),
    ('dragon', 'fb-dragon.py'),
    ('featureExtraction', 'splade-cocondenser-ensembledistil.py'),
    ('featureExtraction', 'legal-bert-dutch-english.py'),
]

# Enter your Python interpreter location here
python_interpreter = ""

# Check if the Python interpreter exists
if not os.path.isfile(python_interpreter):
    raise FileNotFoundError(f"Python interpreter not found at: {python_interpreter}")

# Loop through each file and run it
for folder, file in files_to_run:
    file_path = os.path.join(folder, file)
    if not os.path.isfile(file_path):
        print(f"Script file not found: {file_path}")
        continue

    print(f"Running script: {file_path}")

    try:
        result = subprocess.run([python_interpreter, file_path], check=True, capture_output=True, text=True)
        print(f"Output of {file_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {file_path}:\n{e.stderr}")