import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Multi-PDF Summarizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text(file_bytes):
    document = pymupdf.open(stream=file_bytes, filetype="pdf")
    text = ""
    pages = len(document)
    for page in document:
        text += str(page.get_text()) + " "
    document.close()
    return text, pages


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 20
    ]


def normalize(values):
    minimum = np.min(values)
    maximum = np.max(values)
    if maximum == minimum:
        return np.ones(len(values))
    return (values - minimum) / (maximum - minimum)


def create_summary(sentences):
    """Pure mathematical extractive summary using TF-IDF Matrix Similarity."""
    if len(sentences) <= 3:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences)
        dense_matrix = matrix.toarray()  
    except ValueError:
        return sentences[:3]

    tfidf_scores = dense_matrix.sum(axis=1)

    doc_centroid = dense_matrix.mean(axis=0).reshape(1, -1)
    similarity_scores = cosine_similarity(dense_matrix, doc_centroid).flatten()

    final_scores = 0.5 * normalize(tfidf_scores) + 0.5 * normalize(
        similarity_scores
    )

    number_to_select = max(3, min(len(sentences), int(len(sentences) * 0.3)))
    ranked = np.argsort(final_scores)[::-1]

    selected = []
    for index in ranked:
        if len(selected) >= number_to_select:
            break

        too_similar = False
        if selected:
            sims = cosine_similarity(
                dense_matrix[index].reshape(1, -1), dense_matrix[selected]
            ).flatten()
            if np.max(sims) > 0.60:
                too_similar = True

        if not too_similar:
            selected.append(index)

    selected.sort()
    return [sentences[idx] for idx in selected]


def get_keywords(text, number=8):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=number)
    try:
        vectorizer.fit_transform([text])
        return list(vectorizer.get_feature_names_out())
    except ValueError:
        return []


@app.get("/")
def read_root():
    return {
        "status": "Server running efficiently",
        "message": "Backend is running. Visit /docs to test the API."
    }

@app.get("/health")
def health():
    return {"status": "Backend running efficiently"}

@app.post("/summarize")
async def summarize(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")

    documents: list[dict] = []
    all_sentences: list[str] = []

    for file in files:
        filename = file.filename or "unknown.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"{filename} is not a PDF."
            )

        try:
            file_bytes = await file.read()
            text, pages = extract_text(file_bytes)
            text = clean_text(text)

            if not text:
                raise HTTPException(
                    status_code=400,
                    detail=f"No readable text found in {filename}.",
                )

            sentences = split_sentences(text)
            if not sentences:
                raise HTTPException(
                    status_code=400,
                    detail=f"Not enough text found in {filename}.",
                )

            summary_sentences = create_summary(sentences)
            summary = " ".join(summary_sentences)

            documents.append(
                {
                    "filename": filename,
                    "pages": pages,
                    "original_words": len(text.split()),
                    "summary_words": len(summary.split()),
                    "keywords": get_keywords(text),
                    "summary": summary,
                }
            )
            all_sentences.extend(summary_sentences)

        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error))

    combined: list[str] = []
    if all_sentences:
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            # FIXED: Added type: ignore to bypass false-positive spmatrix warning
            matrix = vectorizer.fit_transform(all_sentences).toarray()  # type: ignore
            for i, sentence in enumerate(all_sentences):
                if not combined:
                    combined.append(sentence)
                    continue

                indices = [all_sentences.index(s) for s in combined]
                similarities = cosine_similarity(
                    matrix[i].reshape(1, -1), matrix[indices]
                ).flatten()

                if np.max(similarities) < 0.60:
                    combined.append(sentence)
        except ValueError:
            combined = all_sentences

    return {"documents": documents, "combined_summary": " ".join(combined)}
