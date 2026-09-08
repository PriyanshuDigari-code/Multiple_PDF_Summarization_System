# AI Multi-PDF Summarizer

A PDF summarization project using HTML, CSS, Vanilla JavaScript, FastAPI, TF-IDF, Sentence Transformers, BERT embeddings and cosine similarity.

## Folder Structure

```text
pdf-summarizer/
├── index.html
├── style.css
├── script.js
└── backend/
    ├── main.py
    └── requirements.txt
```

## How it works

1. Select multiple PDF files in the browser.
2. JavaScript sends the files to FastAPI using `fetch()`.
3. PyMuPDF extracts the PDF text.
4. The text is cleaned and split into sentences.
5. TF-IDF gives each sentence an importance score.
6. Sentence Transformer creates sentence embeddings.
7. BERT creates another semantic representation.
8. The three scores are combined.
9. Cosine similarity removes repeated sentences.
10. The most important original sentences become the extractive summary.
