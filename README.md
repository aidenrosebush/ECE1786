# ECE1786
Repository for ECE1786 Course Project (Fall 2023).

The final presentation for this project can be found here: https://docs.google.com/presentation/d/1RQ0xDc4as7fq5tDBYuLwA_7QouyAC9TLtLHVCYxo0aY/edit?usp=sharing

See BERTModel.ipynb, GPTModel.ipynb, and deBERTA.ipynb for training code with each of the three models we compared.

Achieved 93% accuracy guessing words in declassified CIA documents using bert-base-cased from Huggingface. We compare to deberta-base and gpt2, highlighting the advantage of bidirectional models and model size with respect to training data and context for predictions. 

This project is an example of mixed fine-tuning and training from scratch, as many of the words in the documents were not represented in the stock huggingface models. We show that with enough training time, it is possible for a model to learn a new context for existing tokens and accurately train new embeddings at the same time. 

We demonstrate that with enough context, LLMs could guess valuable redacted information from government documents. The study of safeguarding redacted documents from such tactics is an ongoing field of study which complements this work. See [1], [2] and [3] for some examples. 

Data processing was done largely manually in data_processing.ipynb as many of the scanned or photocopied documents also had unpredictable errors and illegible handwriting. View the complete set of documents used for training by following the links in [4] and [5]. We used Pdf2image [6] and Pytesseract [7] for automatic conversion of the PDF files to text data for us to process.

All code to be run on Google Colab, with Drive locations mentioned as where Drive is mounted. Not included here is the official 9/11 commission report used for background training of the models. It can be found at https://www.9-11commission.gov/report/911Report.pdf. 

[1]
I. Pilán, P. Lison, L. Øvrelid, A. Papadopoulou, D. Sánchez, and M. Batet, “The Text Anonymization Benchmark (TAB): A dedicated corpus and evaluation framework for text anonymization,” Comput. Linguist. Assoc. Comput. Linguist., vol. 48, no. 4, pp. 1053–1101, 2022.

[2]
E. Eder, U. Krieg-Holz, and U. Hahn, “CodE Alltag 2.0 --- A Pseudonymized German-Language Email Corpus,” in Proceedings of the Twelfth Language Resources and Evaluation Conference, 2020, pp. 4466–4477.

[3]
M. Friedrich, A. Köhn, G. Wiedemann, and C. Biemann, “Adversarial learning of privacy-preserving text representations for DE-identification of medical records,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 5829–5839.

[4]
“Freedom of information act electronic reading room,” Cia.gov. [Online]. Available: https://www.cia.gov/readingroom/.

[5]
The 9/11 Commission Report, https://www.9-11commission.gov/report/911Report.pdf

[6]
“Pdf2image,” PyPI. [Online]. Available: https://pypi.org/project/pdf2image/. 

[7]
“Pytesseract,” PyPI. [Online]. Available: https://pypi.org/project/pytesseract/. 







 
