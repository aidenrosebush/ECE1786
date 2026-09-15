# RedactedGPT

Can a language model guess what's under the black bars? This project fine-tunes **BERT**,
**DeBERTa**, and **GPT-2** to predict redacted words in declassified CIA documents, and compares
how the three architectures handle the task.

## Results

Best validation **top-10 accuracy** — the redacted word appears in the model's ten highest-ranked
predictions — after fine-tuning on masked tokens drawn from the document corpus.

| Model | Checkpoint | Best top-10 accuracy |
|-------|-----------|---------------------:|
| **BERT** | `bert-base-cased` | **93.9%** |
| DeBERTa | `microsoft/deberta-base` | 90.8% |
| GPT-2 | `gpt2` | 67.7% |

**Why BERT wins.** Predicting a hidden word from text on *both* sides of it is exactly the masked
language modelling objective BERT was pretrained on. GPT-2 is autoregressive — it only sees
preceding context — so it is solving a harder version of the problem with less information, and the
gap of roughly 26 points reflects that mismatch rather than a difference in model quality.

Loss and accuracy curves for all three models are produced by `src/plot_results.py`.

## How it works

1. **`src/data_processing.py`** — converts source PDFs to page images, OCRs them with Tesseract,
   cleans the extracted text, and builds masked-token training CSVs. Redaction markers in the
   original documents become the prediction targets.
2. **`src/train_bert.py`**, **`train_deberta.py`**, **`train_gpt2.py`** — fine-tune each model on
   those CSVs via the HuggingFace `Trainer`, logging validation loss and top-10 accuracy.
3. **`src/plot_results.py`** — renders the training curves used above.

## Setup

System dependencies for the OCR pipeline:

```bash
# Debian / Ubuntu
sudo apt-get install poppler-utils tesseract-ocr
# macOS
brew install poppler tesseract
```

Python dependencies:

```bash
pip install -r requirements.txt
```

## Running

All paths resolve from `config.py`, which defaults to `./data`. Override it if your corpus lives
elsewhere:

```bash
export REDACTEDGPT_DATA=/path/to/data
```

Then:

```bash
python src/data_processing.py     # PDFs -> OCR text -> training CSVs
python src/train_bert.py          # fine-tune BERT
python src/train_deberta.py       # fine-tune DeBERTa
python src/train_gpt2.py          # fine-tune GPT-2
python src/plot_results.py        # training curves
```

Training was run on GPU; the fine-tuning scripts are impractical on CPU.

## Data

Five sample declassified CIA documents are included in `data/PDFs/` so the pipeline can be run end
to end. The full training corpus — a larger set from the
[CIA FOIA Reading Room](https://www.cia.gov/readingroom/) plus page images of the 9/11 Commission
Report — is not in the repo for size reasons. See [`data/README.md`](data/README.md).

Model checkpoints and generated intermediates are gitignored.

## Notes

Developed as a course project for ECE1786 (Creative Applications of Natural Language Processing),
University of Toronto, Fall 2023. Originally written as Colab notebooks and since converted to
standalone scripts.
