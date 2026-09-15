"""Central paths for RedactedGPT.

By default everything lives under ./data. Override with an environment variable:

    export REDACTEDGPT_DATA=/path/to/your/data
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("REDACTEDGPT_DATA", os.path.join(REPO_ROOT, "data"))
MODEL_ROOT = os.path.join(DATA_ROOT, "Models")

# Subdirectories the pipeline reads and writes. Names match the original project
# layout so the training scripts run unmodified.
SUBDIRS = ["PDFs", "Images", "Textfiles", "CSVs", "Models", "Tokenizer Files"]
for _d in SUBDIRS:
    os.makedirs(os.path.join(DATA_ROOT, _d), exist_ok=True)
