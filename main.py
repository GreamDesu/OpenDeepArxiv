import argparse
import logging
from tqdm import tqdm
from .arxiv_search import ArxivSearch
from .similarity_filters import PaperFilter
from .summarization import SummarizationPipeline

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def main(topic):
    steps = ["arXiv search", "similarity filtering", "PDF report generation"]
    with tqdm(total=len(steps), desc="Overall Pipeline", unit="step") as pbar:
        logging.info("Running arXiv search...")
        searcher = ArxivSearch(topic)
        pdf_metadata_df = searcher.run()
        pbar.update(1)

        logging.info("Running similarity filtering...")
        paper_filter = PaperFilter(topic, pdf_metadata_df)
        merged_df = paper_filter.run()
        pbar.update(1)

        logging.info("Generating final PDF report...")
        pipeline = SummarizationPipeline(
            topic, merged_df, pdf_folder=paper_filter.pdf_folder)
        pdf_report = pipeline.run()
        pbar.update(1)

    logging.info(f"Generated PDF report at: {pdf_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the DeepOpenArxiv Pipeline.")
    parser.add_argument("--topic", required=True,
                        help="Research topic for search and summarization")
    args = parser.parse_args()
    main(args.topic)
