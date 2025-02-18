import os
import json
import re
import time
import logging
import pandas as pd
import arxiv
from openai import OpenAI
from .config import Config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class ArxivSearch:
    def __init__(self, topic):
        self.topic = topic
        # Use the current working directory to resolve the PDF folder
        self.pdf_folder = os.path.join(os.getcwd(), Config.ARXIV_PDF_FOLDER)
        os.makedirs(self.pdf_folder, exist_ok=True)  # Ensure the PDF folder exists
        base_dir = os.path.dirname(__file__)
        self.user_manual_path = os.path.join(base_dir, "arxiv_api_instructions.md")
        self.init_file_path = os.path.join(base_dir, "arxiv_implementation_instructions.py")
        self.taxonomy_file_path = os.path.join(base_dir, "taxonomy.md")

        # Initialize OpenAI client using API key from config
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

        try:
            with open(self.user_manual_path, "r", encoding="utf-8") as f:
                self.user_manual = f.read()
            with open(self.init_file_path, "r", encoding="utf-8") as f:
                self.init_code = f.read()
            with open(self.taxonomy_file_path, "r", encoding="utf-8") as f:
                self.taxonomy_file = f.read()
        except Exception as e:
            logging.error(f"Error reading required files: {e}")
            raise

    def create_search_query(self):
        user_prompt = (f"I don't know anything about {self.topic} and would like a state-of-the-art overview. "
                       f"Today is {time.strftime('%Y-%m-%d')}. Please select the most relevant papers.")
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": ''' 
You are the best researcher in the world. Your task is to determine search parameters for arXiv.
You have the arXiv API instructions and Python library implementation.
Generate query parameters for Python using the "arxiv" library.
                    '''},
                    {"role": "user", "content": f"User prompt: {user_prompt}"},
                    {"role": "system", "content": f"arXiv API instructions:\n{self.user_manual}"},
                    {"role": "system", "content": f"Python library code:\n{self.init_code}"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "query_schema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "query": {"description": "Query string", "type": "string"},
                                "sort_by": {"description": "Sort criterion", "type": "string",
                                            "enum": ["arxiv.SortCriterion.Relevance", "arxiv.SortCriterion.LastUpdatedDate", "arxiv.SortCriterion.SubmittedDate"]},
                                "sort_order": {"description": "Sort order", "type": "string",
                                               "enum": ["arxiv.SortOrder.Descending", "arxiv.SortOrder.Ascending"]},
                                "date_range": {
                                    "description": "Date range",
                                    "type": "object",
                                    "properties": {
                                        "start": {"description": "Start date", "type": "string"},
                                        "end": {"description": "End date", "type": "string"}
                                    }
                                },
                                "categories": {
                                    "description": "Categories",
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "max_results": {"description": "Max results", "type": "string"}
                            }
                        }
                    }
                }
            )
            query_json = json.loads(response.choices[0].message.content)
            with open("arxiv_query.json", "w") as json_file:
                json.dump(query_json, json_file, indent=4)
            logging.info("Query successfully saved to arxiv_query.json")
        except Exception as e:
            logging.error(f"Error generating search query: {e}")
            raise

    def download_papers(self):
        try:
            with open("arxiv_query.json", "r") as json_file:
                query_data = json.load(json_file)
        except Exception as e:
            logging.error(f"Error reading query JSON: {e}")
            raise

        query = query_data.get("query", "")
        max_results = Config.MAX_RESULTS
        sort_by_str = query_data.get("sort_by", "arxiv.SortCriterion.Relevance")
        sort_by = getattr(arxiv.SortCriterion, sort_by_str.split(".")[-1], arxiv.SortCriterion.Relevance)
        sort_order_str = query_data.get("sort_order", "arxiv.SortOrder.Descending")
        sort_order = getattr(arxiv.SortOrder, sort_order_str.split(".")[-1], arxiv.SortOrder.Descending)

        id_list = query_data.get("id_list", [])
        categories = query_data.get("categories", [])
        date_range = query_data.get("date_range", {})

        logging.info(f"Query: {query}")
        logging.info(f"Max Results: {max_results}")
        logging.info(f"Sort By: {sort_by}")
        logging.info(f"Sort Order: {sort_order}")
        logging.info(f"Categories: {categories}")
        logging.info(f"Date Range: {date_range}")

        try:
            if id_list:
                search = arxiv.Search(
                    id_list=id_list,
                    max_results=max_results,
                    sort_by=sort_by,
                    sort_order=sort_order
                )
                query = f"Fetching specific papers by ID: {id_list}"
            else:
                if categories:
                    category_filter = " AND ".join(f"cat:{cat}" for cat in categories)
                    query = f"({query} AND ({category_filter}))" if query else f"({category_filter})"
                if date_range:
                    query = f"({query} AND submittedDate:[{date_range['start']} TO {date_range['end']}])"
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=sort_by,
                    sort_order=sort_order
                )
            logging.info(f"Final Search Query: {query}")
            logging.info(f"Converted arxiv.Search Query: {search}")
        except Exception as e:
            logging.error(f"Error constructing arXiv search: {e}")
            raise

        client_arxiv = arxiv.Client()
        pdf_records = []
        try:
            for paper in client_arxiv.results(search):
                paper_id = paper.entry_id.split("/")[-1]
                filename = f"{paper_id}.pdf"

                match = re.match(r"(\d{2})(\d{2})\.\d{5}", paper_id)
                pub_year = int("20" + match.group(1)) if match else None

                pdf_path = os.path.join(self.pdf_folder, filename)
                if os.path.exists(pdf_path):
                    logging.warning(f"File {filename} already downloaded.")
                else:
                    if match and pub_year < 2020:
                        logging.warning(f"Skipping {filename} due to publication year < 2020.")
                    else:
                        logging.info(f"Found: {paper.title} ({pub_year}) - checking PDF availability...")
                        try:
                            paper.download_pdf(filename=pdf_path)
                        except Exception as e:
                            logging.error(f"Error downloading {filename}: {e}")
                            continue
                        logging.info(f"Downloaded: {filename}")

                authors = ", ".join(author.name for author in paper.authors)
                pdf_records.append({
                    "pdf_name": filename,
                    "title": paper.title,
                    "year": pub_year,
                    "authors": authors,
                })
        except Exception as e:
            logging.error(f"Error during arXiv paper download loop: {e}")

        try:
            pdf_metadata_df = pd.DataFrame(pdf_records)
            pdf_metadata_df.to_csv('pdf_metadata.csv', index=False)
            logging.info("PDF metadata DataFrame saved.")
        except Exception as e:
            logging.error(f"Error saving metadata DataFrame: {e}")
            raise
        return pdf_metadata_df

    def run(self):
        self.create_search_query()
        return self.download_papers()