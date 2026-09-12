import re
from typing import List

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
    """
    Extract readable text from a PDF.
    """

    document = pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    )

    text_parts = []

    for page in document:

        page_text = page.get_text("text")

        if page_text:
            text_parts.append(page_text)

    pages = len(document)

    document.close()

    return "\n".join(text_parts), pages


def clean_text(text):
    """
    Clean common PDF formatting problems.
    """

    text = re.sub(
        r"(\w+)-\s*\n\s*(\w+)",
        r"\1\2",
        text
    )

    text = re.sub(
        r"([a-z]{3,})\s*\n\s*([a-z]{2,})",
        r"\1 \2",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def is_noisy_sentence(sentence):
    """
    Detect sentences that contain mostly numbers,
    chart values, or other PDF extraction noise.
    """

    words = sentence.split()

    if not words:
        return True

    numeric_tokens = 0

    for word in words:

        cleaned = word.strip(
            ".,:;()[]{}$%+-"
        )

        if cleaned.replace(".", "", 1).isdigit():
            numeric_tokens += 1

   
    if len(words) >= 3:

        numeric_ratio = numeric_tokens / len(words)

        if numeric_ratio > 0.60:
            return True

    noisy_patterns = [
        "series 1",
        "item 1",
        "figure 1",
        "chart 1",
        "table 1",
    ]

    lower_sentence = sentence.lower()

    for pattern in noisy_patterns:

        if lower_sentence.strip() == pattern:
            return True

    return False


def split_sentences(text):
    """
    Split cleaned PDF text into meaningful sentences.
    """

    raw_sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = []

    for sentence in raw_sentences:

        sentence = sentence.strip()

        if len(sentence) < 25:
            continue

        if is_noisy_sentence(sentence):
            continue

        alphabetic_characters = sum(
            char.isalpha()
            for char in sentence
        )

        if alphabetic_characters < 15:
            continue

        sentences.append(sentence)

    return sentences


def normalize(values):

    values = np.asarray(values)

    minimum = np.min(values)
    maximum = np.max(values)

    if maximum == minimum:
        return np.ones(len(values))

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def rank_sentences(sentences):
    """
    Calculate importance score for every sentence.
    """

    if not sentences:
        return []

    if len(sentences) == 1:
        return [(0, 1.0)]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:

        matrix = vectorizer.fit_transform(
            sentences
        ).toarray()

    except ValueError:

        return [
            (index, 1.0)
            for index in range(len(sentences))
        ]


    tfidf_scores = matrix.sum(axis=1)


    centroid = matrix.mean(
        axis=0
    ).reshape(1, -1)

    similarity_scores = cosine_similarity(
        matrix,
        centroid
    ).flatten()


    final_scores = (
        0.5 * normalize(tfidf_scores)
        +
        0.5 * normalize(similarity_scores)
    )

    ranked_indices = np.argsort(
        final_scores
    )[::-1]

    return [
        (int(index), float(final_scores[index]))
        for index in ranked_indices
    ]


# =========================================================
# CREATE INDIVIDUAL SUMMARY
# =========================================================

def create_summary(sentences):
    """
    Create an extractive summary for one PDF.
    """

    if not sentences:
        return []

    if len(sentences) <= 3:
        return sentences

    ranked = rank_sentences(sentences)

    # Approximately 30% of sentences
    number_to_select = max(
        3,
        int(len(sentences) * 0.30)
    )

    number_to_select = min(
        number_to_select,
        len(sentences)
    )

    # We need the TF-IDF matrix again
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:

        matrix = vectorizer.fit_transform(
            sentences
        ).toarray()

    except ValueError:

        return sentences[:number_to_select]

    selected_indices = []

    for index, score in ranked:

        if len(selected_indices) >= number_to_select:
            break

        # First sentence
        if not selected_indices:

            selected_indices.append(index)
            continue

        similarities = cosine_similarity(
            matrix[index].reshape(1, -1),
            matrix[selected_indices]
        ).flatten()

        # Avoid highly repetitive sentences
        if np.max(similarities) < 0.60:

            selected_indices.append(index)

    # Restore original document order
    selected_indices.sort()

    return [
        sentences[index]
        for index in selected_indices
    ]


# =========================================================
# GET KEYWORDS
# =========================================================

def get_keywords(text, number=8):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=number,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b"
    )

    try:

        matrix = vectorizer.fit_transform(
            [text]
        )

        if matrix.shape[1] == 0:
            return []

        keywords = list(
            vectorizer.get_feature_names_out()
        )

        return keywords

    except ValueError:

        return []


# =========================================================
# COMBINED SUMMARY
# =========================================================

def create_combined_summary(
    document_sentences
):
    """
    Create a combined summary while guaranteeing
    that every uploaded PDF contributes information.

    document_sentences example:

    [
        [PDF1 sentence1, PDF1 sentence2, ...],
        [PDF2 sentence1, PDF2 sentence2, ...]
    ]
    """

    if not document_sentences:
        return ""

    # Remove empty documents
    valid_documents = [
        sentences
        for sentences in document_sentences
        if sentences
    ]

    if not valid_documents:
        return ""

    # -----------------------------------------------------
    # SPECIAL CASE: only one PDF
    # -----------------------------------------------------

    if len(valid_documents) == 1:

        summary_sentences = create_summary(
            valid_documents[0]
        )

        return " ".join(summary_sentences)

    # -----------------------------------------------------
    # Calculate total number of sentences
    # -----------------------------------------------------

    total_sentences = sum(
        len(sentences)
        for sentences in valid_documents
    )

    # Around 20% of all sentences
    total_to_select = max(
        4,
        int(total_sentences * 0.20)
    )

    total_to_select = min(
        total_to_select,
        total_sentences
    )

    # -----------------------------------------------------
    # We guarantee at least 2 sentences from each PDF
    # when possible.
    # -----------------------------------------------------

    selected_global_sentences = []

    # Store sentences as:
    #
    # (document_number, sentence)
    #
    all_sentences = []

    for document_number, sentences in enumerate(
        valid_documents
    ):

        for sentence in sentences:

            all_sentences.append(
                (
                    document_number,
                    sentence
                )
            )

    # -----------------------------------------------------
    # Rank sentences from ALL documents together
    # -----------------------------------------------------

    sentence_texts = [
        item[1]
        for item in all_sentences
    ]

    ranked = rank_sentences(
        sentence_texts
    )

    # -----------------------------------------------------
    # TF-IDF matrix for similarity checking
    # -----------------------------------------------------

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            sentence_texts
        ).toarray()

    except ValueError:

        return " ".join(
            sentence_texts[:total_to_select]
        )

    selected_indices = []

    # -----------------------------------------------------
    # FIRST:
    # Select important sentences from each document.
    #
    # This guarantees that PDF 1 and PDF 2 both
    # contribute to the combined summary.
    # -----------------------------------------------------

    for document_number in range(
        len(valid_documents)
    ):

        document_indices = [
            index
            for index, item in enumerate(
                all_sentences
            )
            if item[0] == document_number
        ]

        if not document_indices:
            continue

        # Rank this document's sentences
        document_ranked = sorted(
            document_indices,
            key=lambda index: dict(ranked).get(
                index,
                0
            ),
            reverse=True
        )

        # Try to select at least 2
        # from each document.
        sentences_needed = min(
            2,
            len(document_indices)
        )

        for index in document_ranked:

            if len(selected_indices) >= total_to_select:
                break

            if index in selected_indices:
                continue

            if not selected_indices:

                selected_indices.append(index)
                continue

            similarities = cosine_similarity(
                matrix[index].reshape(1, -1),
                matrix[selected_indices]
            ).flatten()

            if np.max(similarities) < 0.60:

                selected_indices.append(index)

            if (
                sum(
                    1
                    for selected in selected_indices
                    if all_sentences[selected][0]
                    == document_number
                )
                >= sentences_needed
            ):
                break

    # -----------------------------------------------------
    # SECOND:
    # Fill remaining slots using globally important
    # sentences.
    # -----------------------------------------------------

    for index, score in ranked:

        if len(selected_indices) >= total_to_select:
            break

        if index in selected_indices:
            continue

        if not selected_indices:

            selected_indices.append(index)
            continue

        similarities = cosine_similarity(
            matrix[index].reshape(1, -1),
            matrix[selected_indices]
        ).flatten()

        if np.max(similarities) < 0.60:

            selected_indices.append(index)

    # -----------------------------------------------------
    # Restore original PDF/sentence order
    # -----------------------------------------------------

    selected_indices.sort()

    selected_global_sentences = [
        all_sentences[index][1]
        for index in selected_indices
    ]

    return " ".join(
        selected_global_sentences
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def read_root():

    return {
        "status": "Server running efficiently",
        "message": "Backend is running. Visit /docs to test the API."
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "Backend running efficiently"
    }


# =========================================================
# SUMMARIZE
# =========================================================

@app.post("/summarize")
async def summarize(
    files: List[UploadFile] = File(...)
):

    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------

    if not files:

        raise HTTPException(
            status_code=400,
            detail="No PDF files uploaded."
        )

    documents = []

    # IMPORTANT:
    # Keep sentences separated by PDF.
    #
    # Example:
    #
    # [
    #   [PDF1 sentence1, PDF1 sentence2],
    #   [PDF2 sentence1, PDF2 sentence2]
    # ]
    #
    document_sentences = []

    # -----------------------------------------------------
    # Process every PDF
    # -----------------------------------------------------

    for file in files:

        filename = (
            file.filename
            or "unknown.pdf"
        )

        # -------------------------------------------------
        # Check extension
        # -------------------------------------------------

        if not filename.lower().endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail=f"{filename} is not a PDF."
            )

        try:

            # ---------------------------------------------
            # Read file
            # ---------------------------------------------

            file_bytes = await file.read()

            # ---------------------------------------------
            # Extract text
            # ---------------------------------------------

            text, pages = extract_text(
                file_bytes
            )

            # ---------------------------------------------
            # Clean text
            # ---------------------------------------------

            text = clean_text(text)

            if not text:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No readable text found in "
                        f"{filename}."
                    )
                )

            # ---------------------------------------------
            # Split into sentences
            # ---------------------------------------------

            sentences = split_sentences(
                text
            )

            if not sentences:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Not enough readable text "
                        f"found in {filename}."
                    )
                )

            # ---------------------------------------------
            # Individual PDF summary
            # ---------------------------------------------

            summary_sentences = create_summary(
                sentences
            )

            summary = " ".join(
                summary_sentences
            )

            # ---------------------------------------------
            # Store sentences separately
            # ---------------------------------------------

            document_sentences.append(
                sentences
            )

            # ---------------------------------------------
            # Store document result
            # ---------------------------------------------

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
                    "keywords": get_keywords(
                        text
                    ),
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

    combined_summary = create_combined_summary(
        document_sentences
    )

    return {
        "documents": documents,
        "combined_summary": combined_summary
    }