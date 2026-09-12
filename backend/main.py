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
    document = pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    )

    text_parts = []

    for page in document:
        page_text = page.get_text()

        if page_text:
            text_parts.append(page_text)

    pages = len(document)

    document.close()

    return "\n".join(text_parts), pages

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]

def normalize(values):

    values = np.asarray(values)

    minimum = np.min(values)
    maximum = np.max(values)

    if maximum == minimum:
        return np.ones(len(values))

    return (values - minimum) / (maximum - minimum)

def create_summary(sentences):

    if not sentences:
        return []

    
    if len(sentences) <= 3:
        return sentences

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:

        matrix = vectorizer.fit_transform(
            sentences
        ).toarray()

    except ValueError:

        return sentences[:3]

    
    tfidf_scores = matrix.sum(axis=1)

    document_centroid = matrix.mean(axis=0).reshape(1, -1)

    similarity_scores = cosine_similarity(
        matrix,
        document_centroid
    ).flatten()

    final_scores = (
        0.5 * normalize(tfidf_scores)
        +
        0.5 * normalize(similarity_scores)
    )

    number_to_select = max(
        3,
        min(
            len(sentences),
            int(len(sentences) * 0.30)
        )
    )

    ranked_indices = np.argsort(
        final_scores
    )[::-1]

    selected_indices = []

    for index in ranked_indices:

        if len(selected_indices) >= number_to_select:
            break

        if not selected_indices:

            selected_indices.append(index)
            continue

        similarities = cosine_similarity(
            matrix[index].reshape(1, -1),
            matrix[selected_indices]
        ).flatten()

        if np.max(similarities) < 0.60:

            selected_indices.append(index)

    selected_indices.sort()

    return [
        sentences[index]
        for index in selected_indices
    ]

def get_keywords(text, number=8):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=number
    )

    try:

        vectorizer.fit_transform([text])

        return list(
            vectorizer.get_feature_names_out()
        )

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

    return {
        "status": "Backend running efficiently"
    }

@app.post("/summarize")
async def summarize(
    files: list[UploadFile] = File(...)
):

    if not files:

        raise HTTPException(
            status_code=400,
            detail="No PDF files uploaded."
        )

    documents = []

    all_original_sentences = []


    for file in files:

        filename = file.filename or "unknown.pdf"

        if not filename.lower().endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail=f"{filename} is not a PDF."
            )

        try:

            file_bytes = await file.read()


            text, pages = extract_text(
                file_bytes
            )

            text = clean_text(text)

            if not text:

                raise HTTPException(
                    status_code=400,
                    detail=f"No readable text found in {filename}."
                )

            sentences = split_sentences(text)

            if not sentences:

                raise HTTPException(
                    status_code=400,
                    detail=f"Not enough text found in {filename}."
                )

            summary_sentences = create_summary(
                sentences
            )

            summary = " ".join(
                summary_sentences
            )

            all_original_sentences.extend(
                sentences
            )

            documents.append(
                {
                    "filename": filename,
                    "pages": pages,
                    "original_words": len(
                        text.split()
                    ),
                    "summary_words": len(
                        summary.split()
                    ),
                    "keywords": get_keywords(text),
                    "summary": summary
                }
            )

        except HTTPException:

            raise

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=str(error)
            )

    combined_summary = ""

    if all_original_sentences:

        try:

            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            matrix = vectorizer.fit_transform(
                all_original_sentences
            ).toarray()


            tfidf_scores = matrix.sum(axis=1)

            document_centroid = matrix.mean(
                axis=0
            ).reshape(1, -1)

            similarity_scores = cosine_similarity(
                matrix,
                document_centroid
            ).flatten()


            final_scores = (
                0.5 * normalize(tfidf_scores)
                +
                0.5 * normalize(similarity_scores)
            )

            number_to_select = max(
                3,
                min(
                    len(all_original_sentences),
                    int(
                        len(all_original_sentences) * 0.20
                    )
                )
            )

            ranked_indices = np.argsort(
                final_scores
            )[::-1]

            selected_indices = []

            for index in ranked_indices:

                if len(selected_indices) >= number_to_select:
                    break

                if not selected_indices:

                    selected_indices.append(index)
                    continue

                similarities = cosine_similarity(
                    matrix[index].reshape(1, -1),
                    matrix[selected_indices]
                ).flatten()

                if np.max(similarities) < 0.60:

                    selected_indices.append(index)


            selected_indices.sort()

            combined_sentences = [
                all_original_sentences[index]
                for index in selected_indices
            ]

            combined_summary = " ".join(
                combined_sentences
            )

        except ValueError:

            combined_summary = " ".join(
                all_original_sentences[:10]
            )

    return {
        "documents": documents,
        "combined_summary": combined_summary
    }