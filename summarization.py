import os
import logging
from tqdm import tqdm
import openai
import markdown
from xhtml2pdf import pisa
from .config import Config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class SummarizationPipeline:
    def __init__(self, topic, merged_df, pdf_folder):
        self.topic = topic
        self.merged_df = merged_df.copy()  # ensure a local copy

        # Use the current working directory for a consistent base dir
        base_dir = os.getcwd()
        self.pdf_folder = os.path.join(base_dir, pdf_folder)
        self.md_summary_filepath = os.path.join(base_dir, "summaries.md")
        self.md_report_filepath = os.path.join(base_dir, "final_report.md")
        self.pdf_filepath = os.path.join(base_dir, "final_report.pdf")

        # Use the API key from Config
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

    def generate_summary(self, text, summary_length="detailed"):
        prompt = (
            f"You are an AI trained in analyzing research papers. "
            f"Summarize the following scientific text in a detailed manner while maintaining key insights. "
            f"Summary should be {summary_length} and avoid unnecessary details."
        )
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"\n\nScientific Paper: {text}"}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path):
        # Replace with your actual PDF text extraction logic (e.g., PyPDF2, pdfminer.six)
        logging.info(f"Extracting text from {pdf_path}")
        return ""

    def generate_summaries_and_save_markdown(self):
        self.merged_df["summary"] = ""
        for idx, row in tqdm(self.merged_df.iterrows(), total=self.merged_df.shape[0], desc="Generating summaries"):
            if row.get("isRelevant"):
                pdf_path = os.path.join(self.pdf_folder, row["filename"])
                text = self.extract_text_from_pdf(pdf_path)
                summary_text = self.generate_summary(text)
                self.merged_df.at[idx, "summary"] = summary_text

        contents = ""
        for _, row in self.merged_df[self.merged_df["isRelevant"]].iterrows():
            contents += f"# {row['title']}\n\n"
            contents += f"**Authors:** {row['authors']}\n\n"
            contents += f"**Year:** {row['year']}\n\n"
            contents += f"{row['summary']}\n\n"
            contents += "---\n\n"

        try:
            with open(self.md_summary_filepath, "w", encoding="utf-8") as f:
                f.write(contents)
            logging.info(f"Markdown summaries saved to '{self.md_summary_filepath}'")
        except Exception as e:
            logging.error(f"Error saving markdown summaries: {e}")
        return self.md_summary_filepath

    def generate_markdown_report(self, summaries):
        prompt = (
            f"User query: {self.topic} \n\n"
            f"Paper Summaries:\n{summaries}\n\n"
            "Based on these summaries, generate a single, structured report in markdown format that overviews the current state-of-the-art in the field. "
            "Cover key insights, trends, and future directions. Include references to papers and explain abbreviations."
        )
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content":
                     "You are a scientific research expert who synthesizes research findings into a coherent, markdown formatted report."},
                    {"role": "user", "content": prompt}
                ]
            )
            report = response.choices[0].message.content.strip()
            with open(self.md_report_filepath, "w", encoding="utf8") as f:
                f.write(report)
            logging.info(f"Structured markdown report saved to '{self.md_report_filepath}'")
            return self.md_report_filepath
        except Exception as e:
            logging.error(f"Error generating markdown report: {e}")
            return ""

    def convert_markdown_to_pdf(self, md_filepath):
        try:
            with open(md_filepath, "r", encoding="utf8") as md_file:
                md_text = md_file.read()
            html_text = markdown.markdown(md_text)
            with open(self.pdf_filepath, "w+b") as pdf_file:
                pisa_status = pisa.CreatePDF(html_text, dest=pdf_file)
            if pisa_status.err == 0:
                logging.info(f"PDF file successfully created at '{self.pdf_filepath}'")
            else:
                logging.error("An error occurred during PDF generation")
            return self.pdf_filepath
        except Exception as e:
            logging.error(f"Error converting markdown to PDF: {e}")
            return ""

    def run(self):
        with tqdm(total=3, desc="Summarization Pipeline", unit="step") as pbar:
            summary_md = self.generate_summaries_and_save_markdown()
            pbar.update(1)

            try:
                with open(summary_md, "r", encoding="utf8") as f:
                    summaries = f.read()
            except Exception as e:
                logging.error(f"Error reading summaries: {e}")
                summaries = ""
            md_report = self.generate_markdown_report(summaries)
            pbar.update(1)

            pdf_file = self.convert_markdown_to_pdf(md_report)
            pbar.update(1)

        return pdf_file