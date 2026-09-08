# AI Multi-PDF Summarizer

An AI-powered multi-PDF extractive summarization system built using HTML, CSS, Vanilla JavaScript, FastAPI, PyMuPDF, TF-IDF, NumPy, and Cosine Similarity.

## Features

1. Upload multiple PDF files at once
2. Extract text from PDF documents
3. Clean and process extracted text
4. Split documents into sentences
5. Use TF-IDF to calculate sentence importance
6. Use cosine similarity to measure sentence relevance
7. Remove highly similar and repeated sentences
8. Generate an extractive summary for each PDF
9. Generate a combined summary from all uploaded PDFs
10. Extract important keywords from each document
11. Display original and summary word counts
12. Display the number of pages in each PDF

## Technologies Used

### Frontend

- HTML
- CSS
- Vanilla JavaScript

### Backend

- Python
- FastAPI
- Uvicorn

### NLP / Machine Learning

- PyMuPDF
- TF-IDF
- Cosine Similarity
- NumPy
- Scikit-learn

## Folder Structure

```text
Multiple_PDF_Summarization_System/
│
├── backend/
│   ├── main.py
│   └── requirements.txt
│
├── index.html
├── script.js
├── style.css
├── README.md
└── .gitignore
```

## How It Works

1. Select one or multiple PDF files in the browser.
2. JavaScript sends the PDF files to the FastAPI backend.
3. PyMuPDF extracts text from each PDF.
4. The extracted text is cleaned and divided into sentences.
5. TF-IDF calculates the importance of words and sentences.
6. Cosine similarity measures the relevance of sentences to the document.
7. Important sentences are selected to create an extractive summary.
8. Highly similar sentences are removed to reduce repetition.
9. Keywords and document statistics are generated.
10. A combined summary is created from all uploaded PDFs.

## API Endpoints

### GET `/`

Checks whether the backend is running.

### GET `/health`

Health-check endpoint.

### POST `/summarize`

Accepts multiple PDF files and returns:

- Individual document summaries
- Combined summary
- Keywords
- Page count
- Original word count
- Summary word count

## Project Purpose

The project demonstrates how traditional NLP techniques such as TF-IDF and cosine similarity can be used to build a practical extractive PDF summarization system.

## Deployment

The frontend can be deployed using GitHub Pages and the FastAPI backend can be deployed using Render.
