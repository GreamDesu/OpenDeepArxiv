import os
import json
import numpy as np
import pandas as pd
import logging
import umap
import torch
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
from transformers import AutoTokenizer, AutoModel
import openai
from .config import Config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class PaperFilter:
    def __init__(self, topic, pdf_metadata_df, pdf_folder="arxiv_downloads",
                 embedding_file=Config.EMBEDDING_FILE,
                 filename_file=Config.FILENAME_FILE,
                 similarity_matrix_file=Config.SIMILARITY_MATRIX_FILE,
                 final_csv_path=Config.FINAL_CSV_PATH,
                 final_openai_csv_path="openai_filtered_papers.csv"):
        # Use the current working directory so that the pdf folder matches the download location.
        base_dir = os.getcwd()
        self.topic = topic
        self.pdf_metadata_df = pdf_metadata_df
        self.pdf_folder = os.path.join(base_dir, pdf_folder)
        self.embedding_file = os.path.join(base_dir, embedding_file)
        self.filename_file = os.path.join(base_dir, filename_file)
        self.similarity_matrix_file = os.path.join(
            base_dir, similarity_matrix_file)
        self.final_csv_path = os.path.join(base_dir, final_csv_path)
        self.final_openai_csv_path = os.path.join(
            base_dir, final_openai_csv_path)
        self.SEED = 42
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        # Use API key from config instead of hardcoding it.
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def pdf_to_text(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text("text") for page in doc)
            return text
        except Exception as e:
            logging.error(f"Error reading {pdf_path}: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path, max_chars=100000):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
                if len(text) >= max_chars:
                    break
            return text[:max_chars]
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def get_embedding(self, text):
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=Config.TOKENIZER_MAX_LENGTH)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            embedding = torch.mean(last_hidden_state, dim=1).numpy()
            return embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return np.zeros((1, 768))

    def process_papers(self):
        logging.info(f"Looking for PDFs in: {self.pdf_folder}")
        if not os.path.exists(self.pdf_folder):
            logging.error(f"Folder not found: {self.pdf_folder}")
        else:
            logging.info(f"Files in folder: {os.listdir(self.pdf_folder)}")

        pdf_files = [os.path.join(self.pdf_folder, f)
                     for f in os.listdir(self.pdf_folder) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logging.info(f"No PDF files found in folder: {self.pdf_folder}")
        else:
            logging.info(
                f"Found {len(pdf_files)} PDF files in folder: {self.pdf_folder}")

        texts, filenames, embeddings = [], [], []
        for pdf in tqdm(pdf_files, desc="Processing PDFs"):
            text = self.pdf_to_text(pdf)
            if text:
                texts.append(text)
                filenames.append(os.path.basename(pdf))
                embeddings.append(self.get_embedding(text))
        try:
            embeddings = np.vstack(embeddings)
        except Exception as e:
            logging.error(f"Error stacking embeddings: {e}")
            embeddings = np.array([])
        return filenames, embeddings

    def save_embeddings(self, filenames, embeddings):
        try:
            np.save(self.embedding_file, embeddings)
            pd.DataFrame(filenames).to_csv(
                self.filename_file, index=False, header=False)
            logging.info(
                f"Embeddings saved to '{self.embedding_file}' and filenames saved to '{self.filename_file}'.")
        except Exception as e:
            logging.error(f"Error saving embeddings: {e}")

    def load_embeddings(self):
        try:
            embeddings = np.load(self.embedding_file)
            filenames = pd.read_csv(self.filename_file, header=None)[
                0].tolist()
            logging.info(
                f"Loaded {len(filenames)} filenames and embeddings of shape {embeddings.shape}.")
            return filenames, embeddings
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")
            return [], np.array([])

    def compute_similarity(self, filenames, embeddings):
        try:
            similarity_matrix = cosine_similarity(embeddings)
            df_sim = pd.DataFrame(
                similarity_matrix, index=filenames, columns=filenames)
            return df_sim
        except Exception as e:
            logging.error(f"Error computing similarity: {e}")
            return pd.DataFrame()

    def remove_highly_similar_papers(self, similarity_matrix, threshold=Config.SIMILARITY_THRESHOLD):
        to_keep = set(similarity_matrix.index)
        removed_count = 0

        for i, paper1 in enumerate(similarity_matrix.index):
            if paper1 not in to_keep:
                continue
            for paper2 in similarity_matrix.index[i + 1:]:
                if paper2 in to_keep and similarity_matrix.loc[paper1, paper2] > threshold:
                    to_keep.remove(paper2)
                    removed_count += 1
        filtered_filenames = list(to_keep)
        logging.info(
            f"Removed {removed_count} highly similar papers (threshold={threshold}).")
        return filtered_filenames

    def reduce_with_umap(self, embeddings,
                         n_neighbors=Config.UMAP_N_NEIGHBORS,
                         min_dist=Config.UMAP_MIN_DIST,
                         n_components=Config.UMAP_N_COMPONENTS):

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            metric='cosine', n_components=n_components, random_state=self.SEED)
        return reducer.fit_transform(embeddings)

    def auto_elbow_method(self, data, max_k=Config.MAX_CLUSTERS):
        distortions = []
        K_range = range(1, max_k + 1)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.SEED, n_init=10)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
        distortions = np.array(distortions)
        distortions = (distortions - distortions.min()) / \
            (distortions.max() - distortions.min())
        d1 = np.diff(distortions)
        d2 = np.diff(d1)
        concavity = np.sign(d2.mean())
        extrema = argrelextrema(
            distortions, np.less if concavity == -1 else np.greater)[0]
        optimal_k = extrema[0] + 1 if len(extrema) > 0 else max_k // 2
        logging.info(f"Optimal number of clusters determined: {optimal_k}")
        return optimal_k

    def remove_farthest_papers(self, reduced_embeddings, filtered_filenames, labels, percentage=Config.REMOVE_FARTHEST_PERCENTAGE):
        centroids = np.array(
            [reduced_embeddings[labels == i].mean(axis=0) for i in np.unique(labels)])
        distances = np.linalg.norm(
            reduced_embeddings - centroids[labels], axis=1)
        threshold_index = int(len(distances) * (1 - percentage))
        sorted_indices = np.argsort(distances)
        keep_indices = sorted_indices[:threshold_index]
        kept_filenames = [filtered_filenames[i] for i in keep_indices]
        removed_indices = sorted_indices[threshold_index:]
        removed_filenames = [filtered_filenames[i] for i in removed_indices]
        kept_embeddings = reduced_embeddings[keep_indices]
        removed_embeddings = reduced_embeddings[removed_indices]
        kept_labels = labels[keep_indices]
        removed_labels = labels[removed_indices]
        return kept_filenames, removed_filenames, kept_embeddings, removed_embeddings, kept_labels, removed_labels, centroids

    def query_openai(self, text):
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content":
                     f"""You are an expert researcher experienced in evaluating research papers. Given the user's topic query and the content of a paper, determine if the paper is relevant to the topic.
User query: {self.topic}.
Respond with exactly one word: 'True' if the paper is relevant, or 'False' if it is not. Do not include any extra text."""},
                    {"role": "user", "content": f"Here is a paper:\n{text}\n\nIs this paper related to the topic?"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "query_schema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "verdict": {
                                    "description": "Is paper relevant to the user's query?",
                                    "type": "boolean"
                                }
                            }
                        }
                    }
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error querying OpenAI: {e}")
            return "error"

    def plot_3d_umap_with_centroids_and_removed(self, kept_embeddings, kept_labels,
                                                removed_embeddings, removed_labels,
                                                centroids, center_of_centroids):
        try:
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            unique_labels = np.unique(
                np.concatenate([kept_labels, removed_labels]))
            n_labels = len(unique_labels)
            cmap = plt.get_cmap('coolwarm', n_labels)
            ax.scatter(kept_embeddings[:, 0], kept_embeddings[:, 1], kept_embeddings[:, 2],
                       c=kept_labels, cmap=cmap, s=80, alpha=0.9, edgecolor='k', label="Kept Papers")
            ax.scatter(removed_embeddings[:, 0], removed_embeddings[:, 1], removed_embeddings[:, 2],
                       c=removed_labels, cmap=cmap, s=60, alpha=0.5, edgecolor='grey', label="Removed Papers (20%)")
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       color='red', s=200, marker='X', edgecolor='black', linewidth=1.5, label="Cluster Centroids")
            ax.scatter(center_of_centroids[0], center_of_centroids[1], center_of_centroids[2],
                       color='navy', s=250, marker='P', edgecolor='black', linewidth=1.5, label="Center of Centroids")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            ax.set_title("3D UMAP Visualization with Clustering")
            ax.grid(True)
            ax.view_init(elev=20, azim=45)
            ax.legend(loc='best')
            plt.tight_layout()
            # Save plot relative to the same folder
            save_path = os.path.join(os.path.dirname(
                __file__), "3d_umap_clustering.png")
            plt.savefig(save_path, dpi=300)
            logging.info(f"3D UMAP clustering plot saved as '{save_path}'")
        except Exception as e:
            logging.error(f"Error generating 3D plot: {e}")

    def run(self):
        logging.info("Generating and saving embeddings.")
        filenames, embeddings = self.process_papers()
        self.save_embeddings(filenames, embeddings)
        similarity_matrix = self.compute_similarity(filenames, embeddings)
        similarity_matrix.to_csv(self.similarity_matrix_file)
        logging.info(
            f"Similarity matrix saved to '{self.similarity_matrix_file}'.")
        filtered_filenames = self.remove_highly_similar_papers(
            similarity_matrix)
        filtered_indices = [filenames.index(f) for f in filtered_filenames]
        filtered_embeddings = embeddings[filtered_indices]
        reduced_embeddings = self.reduce_with_umap(
            filtered_embeddings, n_components=3)
        n_clusters = self.auto_elbow_method(reduced_embeddings, max_k=10)
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self.SEED, n_init=10)
        labels = kmeans.fit_predict(reduced_embeddings)
        centroids = kmeans.cluster_centers_
        center_of_centroids = np.mean(centroids, axis=0)
        (kept_filenames,
         removed_filenames,
         kept_embeddings,
         removed_embeddings,
         kept_labels,
         removed_labels,
         _) = self.remove_farthest_papers(reduced_embeddings, filtered_filenames, labels)
        self.plot_3d_umap_with_centroids_and_removed(kept_embeddings, kept_labels,
                                                     removed_embeddings, removed_labels,
                                                     centroids, center_of_centroids)
        pd.DataFrame(kept_filenames, columns=["Filename"]).to_csv(
            self.final_csv_path, index=False)
        logging.info(
            f"Final filtered papers saved to '{self.final_csv_path}'.")
        results = []
        for filename in tqdm(kept_filenames, desc="Classifying papers with OpenAI", unit="paper"):
            pdf_path = os.path.join(self.pdf_folder, filename)
            extracted_text = self.extract_text_from_pdf(pdf_path)
            response = self.query_openai(extracted_text)
            try:
                verdict = json.loads(response)['verdict']
            except Exception:
                verdict = False
            results.append((filename, verdict))
        verdicts = pd.DataFrame(results, columns=["Filename", "isRelevant"])
        verdicts.to_csv(self.final_openai_csv_path, index=False)
        logging.info(
            f"Filtered papers with OpenAI classification saved to '{self.final_openai_csv_path}'.")
        relevant_verdicts = verdicts[verdicts["isRelevant"] == True]
        merged_df = pd.merge(relevant_verdicts, self.pdf_metadata_df,
                             left_on="Filename", right_on="pdf_name", how="inner")
        merged_df = merged_df.drop(columns=["pdf_name"]).rename(
            columns={"Filename": "filename"})
        merged_df.to_csv(self.final_csv_path, index=False)
        logging.info(f"Merged dataframe saved to '{self.final_csv_path}'.")
        return merged_df
