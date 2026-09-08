# AI Multi-PDF Summarizer

A simple college-level PDF summarization project using HTML, CSS, Vanilla JavaScript, FastAPI, TF-IDF, Sentence Transformers, BERT embeddings and cosine similarity.

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

## Run the backend

Open a terminal in the `backend` folder:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn main:app --reload
```

The API will run at `http://127.0.0.1:8000`.

## Run the frontend

Open `index.html` in a browser. If your browser blocks local requests, use VS Code Live Server.

## API

- `GET /health` checks whether FastAPI is running.
- `POST /summarize` accepts multiple PDF files and returns summaries.

## Important

The first run can be slow because the Sentence Transformer and BERT models need to download and load. The models are loaded once when FastAPI starts.
