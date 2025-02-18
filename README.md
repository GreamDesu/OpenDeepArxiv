# OpenDeepArxiv

OpenDeepArxiv is an open-sourced project designed to streamline the process of searching for research papers on arXiv, filtering based on relevance, and generating comprehensive PDF reports complete with summaries. This pipeline leverages state-of-the-art summarization techniques to help researchers quickly grasp the latest findings in their fields.

## Overview

The project is organized into several stages:
- **ArXiv Search:** Retrieves papers using custom queries constructed with OpenAI. It handles API communication and downloads PDFs.
- **Similarity Filtering:** Processes paper metadata and applies text similarity measures to shortlist the most relevant publications.
- **PDF Report Generation:** Generates a final PDF report combining paper metadata with concise summaries generated for each relevant paper.

## How It Works

### ArXiv Search Procedure
- Constructs a custom query using a prompt and OpenAI.
- Communicates with the arXiv API to retrieve search results.
- Downloads available PDFs and stores metadata for further processing.

### Filtering Procedure
- Computes text embeddings for retrieved papers.
- Compares these embeddings against the user-defined topic.
- Removes papers below a configurable similarity threshold and trims outlier entries based on the percentage setting.

### Summarization Procedure
- Utilizes a summarization pipeline to process filtered papers.
- Leverages advanced models (e.g. OpenAI’s) to generate concise summaries.
- Combines summaries with metadata into a comprehensive PDF report.

## Why OpenDeepArxiv

- **Automated Research:** The end-to-end pipeline automates paper retrieval, relevance filtering, and report generation.
- **Advanced Filtering:** Uses modern embedding techniques for smart similarity filtering, ensuring only the most pertinent papers are selected.
- **Customizable:** Configuration via environment variables makes it easy to tailor search parameters, filters, and report formatting.
- **Open Source Collaboration:** Contributions are welcome; the package is designed to evolve with community feedback and improvements.

## Features

- Automated search and download of arXiv papers.
- Intelligent filtering using similarity thresholds.
- Comprehensive and visually appealing PDF reports.
- Easy configuration via environment variables.

## Requirements

- Python 3.7+
- Libraries: argparse, logging, pandas, tqdm, arxiv, openai, dotenv, feedparser, requests, etc.
- An OpenAI API key (set in the `.env` file)
- A virtual environment is recommended.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/GreamDesu/OpenDeepArxiv.git
   cd OpenDeepArxiv
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root with the following content:
   ```properties
   OPENAI_API_KEY="your_openai_api_key_here"
   ARXIV_PDF_FOLDER="arxiv_downloads"
   EMBEDDING_FILE=bert_embeddings.npy
   FILENAME_FILE=pdf_filenames.csv
   SIMILARITY_MATRIX_FILE=bert_similarity_matrix.csv
   FINAL_CSV_PATH=filtered_papers_final.csv
   ```
5. Run the pipeline:
   ```bash
   python main.py --topic "machine learning"
   ```

## Pipeline Workflow

1. **ArXiv Search:** Initiated from `main.py`, it uses `ArxivSearch` to create a search query and download papers.
2. **Similarity Filtering:** The `PaperFilter` processes the metadata and applies similarity thresholds as configured in `config.py`.
3. **Report Generation:** The `SummarizationPipeline` generates a PDF report with summaries of the selected papers.

### Detailed Pipeline Process

1. **User Query:**  
   - The user starts by providing a topic or research question. This input represents their area of interest and drives the entire workflow.

2. **LLM Query Formulation:**  
   - The user query is sent to a large language model (LLM).  
   - The LLM leverages its background knowledge and embedded instructions (including details from arXiv API documentation and module code) to generate precise search parameters.  
   - This process converts the vague user interest into a structured query that can include filters like subject categories, date ranges, and sorting priorities.

3. **Downloading Papers:**  
   - The generated query is handed off to the ArXiv API.  
   - Relevant papers are retrieved based on the custom query parameters.  
   - Each paper’s metadata is saved locally, and its corresponding PDF is downloaded if available.

4. **Rough Filtering (First-Level Filtering):**  
   - The system computes text embeddings for each paper’s abstract or full text using advanced models (e.g., BERT).  
   - At this stage, the pipeline removes papers that are either:
     - Overly similar, preventing redundancy (i.e., duplicate or near-duplicate content).
     - Irrelevant to the user-specified topic based on a similarity threshold defined in the configuration.
   - This stage helps in reducing the overall pool to a more manageable and relevant subset.

5. **Fine Filtering (Second-Level Filtering):**  
   - The refined list from the rough filtering is then passed to the LLM once again.  
   - The LLM reviews each paper’s content against the original user query and its own prior knowledge.  
   - This step further eliminates any papers that, despite passing the rough filtering, do not sufficiently match the research focus.  
   - This dual-step filtering ensures high precision in the paper selection process.

6. **Summarization:**  
   - For each paper that survives the filtering stages, a summarization pipeline is activated.  
   - The LLM processes the paper’s content to generate concise yet comprehensive summaries.  
   - This step is crucial to ensure that all useful information is preserved even as the content is compressed to a digestible format.

7. **Report Generation:**  
   - The system concatenates the summaries along with the original metadata (title, authors, publication year, etc.) to form a comprehensive PDF report.  
   - This report provides users with an in-depth overview of the field, highlighting state-of-the-art research and key insights in a readable format.
   - The final report serves as a valuable resource for quickly familiarizing oneself with the research domain.

## Project Structure

```
OpenDeepArxiv/
├── __init__.py           # Package initialization and exports
├── __main__.py           # Entry point for running the package as a module
├── .env                  # Environment variables (e.g., API keys)
├── arxiv_api_instructions.md
├── arxiv_downloads/       # Directory to save downloaded PDFs
├── arxiv_implementation_instructions.py
├── arxiv_search.py        # Module for constructing queries and downloading papers
├── config.py              # Configuration settings for the project
├── main.py                # Main pipeline entry point with CLI support
├── README.md              # Project documentation (this file)
├── similarity_filters.py  # Module for paper filtering and similarity analysis
├── summarization.py       # Module for generating summaries and reports
├── taxonomy.md            # Taxonomy details used by the OpenAI query generator
└── ...                    # Other files and notebooks
```

## Usage

### Command Line

The pipeline can be executed directly from the command line. For example:

```bash
python -m OpenDeepArxiv --topic "Diffusion models in robotics"
```

Alternatively, you can also run the pipeline using the `main.py` script:

```bash
python main.py --topic "Diffusion models in robotics"
```

### As a Python Package

You can import the `main` function from the package in your own scripts:

```python
from OpenDeepArxiv import main

topic = "Diffusion models in robotics"
main(topic)
```

### In a Jupyter Notebook

Import the package directly into your notebook:

```python
from OpenDeepArxiv import main

topic = "Diffusion models in robotics"
main(topic)
```

## Customization

- **Modularity:** The project is broken into modules (`arxiv_search.py`, `similarity_filters.py`, and `summarization.py`) to ease maintenance and future enhancements.
- **Logging & Error Handling:** Replace print statements with logging and check error handling for robust pipeline execution.
- **Extensibility:** Further customization can be added, for example, to support asynchronous processing or integration with additional APIs.

## Contributing

Contributions are welcome! This project is open-sourced and intended to promote collaborative improvements. If you’d like to contribute:
- Fork the repository and make your changes.
- Submit a pull request with your improvements or bug fixes.
- Open issues for bugs or feature requests.

Your input helps make OpenDeepArxiv more robust and user-friendly.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

OpenDeepArxiv is maintained by [GreamDesu](https://github.com/GreamDesu).