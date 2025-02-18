import argparse
from .main import main

def run():
    parser = argparse.ArgumentParser(
        description="Run the DeepOpenArxiv Pipeline as a package."
    )
    parser.add_argument("--topic", required=True,
                        help="Research topic for search and summarization")
    args = parser.parse_args()
    main(args.topic)

if __name__ == "__main__":
    run()