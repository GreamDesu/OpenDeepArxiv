import os
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ARXIV_PDF_FOLDER = os.getenv("ARXIV_PDF_FOLDER", "arxiv_downloads")
    EMBEDDING_FILE = os.getenv("EMBEDDING_FILE", "bert_embeddings.npy")
    FILENAME_FILE = os.getenv("FILENAME_FILE", "pdf_filenames.csv")
    SIMILARITY_MATRIX_FILE = os.getenv(
        "SIMILARITY_MATRIX_FILE", "bert_similarity_matrix.csv")
    FINAL_CSV_PATH = os.getenv("FINAL_CSV_PATH", "filtered_papers_final.csv")
    MAX_RESULTS = 200
    SIMILARITY_THRESHOLD = 0.95
    MAX_CLUSTERS = 10
    # New configurable UMAP parameters
    UMAP_N_NEIGHBORS = 10
    UMAP_MIN_DIST = 0.1
    UMAP_N_COMPONENTS = 3

    # New percentage for removing farthest papers
    REMOVE_FARTHEST_PERCENTAGE = 0.20

    # Optionally: tokenizer maximum length for embedding
    TOKENIZER_MAX_LENGTH = 512