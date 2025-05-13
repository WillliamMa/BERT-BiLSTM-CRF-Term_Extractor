# TERMEX: A Domain-Agnostic Pipeline for Scientific Terminology Extraction Using BERT-Based Sequence Labeling

TERMEX is a full-stack pipeline for extracting single-word and multi-word technical terms from scientific research papers. It combines rule-based linguistic preprocessing with a fine-tuned BERT-BiLSTM-CRF model for sequence labeling.

## Features

- PDF-to-text conversion using `pdfminer`
- Sentence segmentation and noun phrase extraction with `spaCy`
- BIO tagging with fuzzy matching
- Data cleaning with domain-specific rules
- BERT-based model training and warm-start fine-tuning
- Evaluation and visualization tools

## Pipeline Overview

```
PDF -> Cleaned Text -> Sentence & NP Extraction -> BIO Tagging -> Cleaned Data -> BERT-Compatible Tensors -> Training -> Evaluation
```

## Installation

```bash
git clone https://github.com/WillliamMa BERT-BiLSTM-CRF-Term-Extractor.git
cd BERT-BiLSTM-CRF-Term_Extractor
pip install -r requirements.txt
```

## Usage

### Preprocessing

```bash
python extract.py
python preprocess.py
python generate_bio_tags.py
python clean_data_bio.py
python preprocess_bert_data.py
```

### Model Training

```bash
python train_bert.py
```

### Evaluation

Results will be printed per epoch and can be visualized from logs or integrated into a test script.

## Project Structure

- `extract.py`: Extract text from PDFs
- `preprocess.py`: Clean and segment text, extract noun phrases
- `generate_bio_tags.py`: Annotate BIO tags via fuzzy matching
- `clean_data_bio.py`: Clean noisy tags and fix labeling issues
- `preprocess_bert_data.py`: Convert data to BERT-ready tensor format
- `train_bert.py`: Model training and evaluation script
- `bert_bilstm_crf.py`: Main model architecture

## Requirements

See `requirements.txt`

## License

MIT License