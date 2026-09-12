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
    Extract text from a PDF while preserving line breaks.
    """

    try:
        document = pymupdf.open(
            stream=file_bytes,
            filetype="pdf"
        )
    except Exception as error:
        raise ValueError(
            f"Could not open PDF: {str(error)}"
        )

    pages = len(document)

    page_texts = []

    for page in document:

        text = page.get_text(
            "text",
            sort=True
        )

        if text:
            page_texts.append(text)

    document.close()

    return "\n".join(page_texts), pages


def is_noise_line(line):
    """
    Detect common PDF extraction noise such as:
    chart numbers, series labels and standalone numbers.
    """

    line = line.strip()

    if not line:
        return True

    lower = line.lower()

    chart_labels = {
        "series 1",
        "series 2",
        "series 3",
        "series 4",
        "series 5",
        "item 1",
        "item 2",
        "item 3",
        "item 4",
        "item 5",
        "item 6",
    }

    if lower in chart_labels:
        return True

    if re.fullmatch(
        r"[\d\s.,:%$+\-()/]+",
        line
    ):
        return True

    words = line.split()

    if len(words) >= 3:

        numeric_count = 0

        for word in words:

            cleaned = word.strip(
                ".,:;()[]{}$%+-"
            )

            if re.fullmatch(
                r"\d+(?:\.\d+)?",
                cleaned
            ):
                numeric_count += 1

        ratio = numeric_count / len(words)

        if ratio >= 0.60:
            return True

    return False

def clean_text(text):
    """
    Clean common PDF extraction problems.
    """

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"(\w+)-\s*\n\s*(\w+)",
        r"\1\2",
        text
    )

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if is_noise_line(line):
            continue

        lines.append(line)

    cleaned_lines = []

    for line in lines:

        line = re.sub(
            r"[ \t]+",
            " ",
            line
        ).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(
        cleaned_lines
    )

def split_sentences(text):
    """
    Convert PDF text into meaningful content units.

    PDFs often have headings and paragraphs without
    normal sentence formatting, so both line boundaries
    and punctuation are considered.
    """

    lines = text.split("\n")

    sentences = []

    current = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if is_noise_line(line):
            continue

        is_heading = (
            len(line) <= 80
            and not line.endswith(
                (".", "!", "?", ",")
            )
        )

        if is_heading:

            if current:

                sentences.append(
                    current.strip()
                )

                current = ""

            if len(line) >= 20:
                sentences.append(line)

            continue

        if not current:

            current = line

        else:

            current += " " + line

        parts = re.split(
            r"(?<=[.!?])\s+",
            current
        )

        if len(parts) > 1:

            complete_parts = parts[:-1]

            current = parts[-1]

            for part in complete_parts:

                part = part.strip()

                if len(part) >= 25:

                    sentences.append(part)

    if current:

        parts = re.split(
            r"(?<=[.!?])\s+",
            current
        )

        for part in parts:

            part = part.strip()

            if len(part) >= 25:

                sentences.append(part)

    final_sentences = []

    for sentence in sentences:

        sentence = re.sub(
            r"\s+",
            " ",
            sentence
        ).strip()

        if len(sentence) < 25:
            continue

        if is_noise_line(sentence):
            continue

        alphabetic_count = sum(
            char.isalpha()
            for char in sentence
        )

        if alphabetic_count < 15:
            continue

        final_sentences.append(
            sentence
        )

    return final_sentences

def normalize(values):

    values = np.asarray(values)

    if len(values) == 0:
        return values

    minimum = np.min(values)
    maximum = np.max(values)

    if maximum == minimum:
        return np.ones(
            len(values)
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )

def rank_sentences(sentences):
    """
    Rank sentences using:
    1. TF-IDF importance
    2. Similarity to document centroid
    """

    if not sentences:
        return []

    if len(sentences) == 1:
        return [(0, 1.0)]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b"
    )

    try:

        matrix = vectorizer.fit_transform(
            sentences
        ).toarray()

    except ValueError:

        return [
            (index, 1.0)
            for index in range(
                len(sentences)
            )
        ]

    tfidf_scores = matrix.sum(
        axis=1
    )

    centroid = matrix.mean(
        axis=0
    ).reshape(1, -1)

    similarity_scores = cosine_similarity(
        matrix,
        centroid
    ).flatten()

    final_scores = (
        0.5 * normalize(
            tfidf_scores
        )
        +
        0.5 * normalize(
            similarity_scores
        )
    )

    ranked_indices = np.argsort(
        final_scores
    )[::-1]

    return [
        (
            int(index),
            float(final_scores[index])
        )
        for index in ranked_indices
    ]

def create_summary(sentences):
    """
    Create a concise extractive summary.
    """

    if not sentences:
        return []

    if len(sentences) <= 3:
        return sentences

    ranked = rank_sentences(
        sentences
    )

    number_to_select = max(
        3,
        int(len(sentences) * 0.20)
    )

    number_to_select = min(
        number_to_select,
        len(sentences)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b"
    )

    try:

        matrix = vectorizer.fit_transform(
            sentences
        ).toarray()

    except ValueError:

        return sentences[
            :number_to_select
        ]

    selected_indices = []

    for index, score in ranked:

        if len(selected_indices) >= number_to_select:
            break

        if not selected_indices:

            selected_indices.append(
                index
            )

            continue

        similarities = cosine_similarity(
            matrix[index].reshape(1, -1),
            matrix[selected_indices]
        ).flatten()

        if np.max(similarities) < 0.65:

            selected_indices.append(
                index
            )

    selected_indices.sort()

    return [
        sentences[index]
        for index in selected_indices
    ]

def get_keywords(text, number=8):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=number,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b"
    )

    try:

        vectorizer.fit_transform(
            [text]
        )

        return list(
            vectorizer.get_feature_names_out()
        )

    except ValueError:

        return []

def create_combined_summary(
    document_sentences
):
    """
    Create one summary from all uploaded PDFs.

    Every PDF gets a chance to contribute important
    information to the final combined summary.
    """

    if not document_sentences:
        return ""

    valid_documents = [
        sentences
        for sentences in document_sentences
        if sentences
    ]

    if not valid_documents:
        return ""


    if len(valid_documents) == 1:

        summary = create_summary(
            valid_documents[0]
        )

        return " ".join(summary)

    all_sentences = []

    for document_index, sentences in enumerate(
        valid_documents
    ):

        for sentence in sentences:

            all_sentences.append(
                {
                    "document": document_index,
                    "text": sentence
                }
            )

    sentence_texts = [
        item["text"]
        for item in all_sentences
    ]

    if not sentence_texts:
        return ""

    ranked = rank_sentences(
        sentence_texts
    )

    total_sentences = len(
        sentence_texts
    )

    number_to_select = max(
        4,
        int(total_sentences * 0.15)
    )

    number_to_select = min(
        number_to_select,
        total_sentences
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b"
    )

    try:

        matrix = vectorizer.fit_transform(
            sentence_texts
        ).toarray()

    except ValueError:

        return " ".join(
            sentence_texts[
                :number_to_select
            ]
        )

    selected_indices = []

    for document_index in range(
        len(valid_documents)
    ):

        candidates = []

        for sentence_index, score in ranked:

            if (
                all_sentences[
                    sentence_index
                ]["document"]
                == document_index
            ):
                candidates.append(
                    sentence_index
                )

        if not candidates:
            continue

        for index in candidates:

            if index in selected_indices:
                continue

            if not selected_indices:

                selected_indices.append(
                    index
                )

                break

            similarities = cosine_similarity(
                matrix[index].reshape(
                    1,
                    -1
                ),
                matrix[selected_indices]
            ).flatten()

            if np.max(similarities) < 0.65:

                selected_indices.append(
                    index
                )

                break

    for index, score in ranked:

        if len(selected_indices) >= number_to_select:
            break

        if index in selected_indices:
            continue

        if not selected_indices:

            selected_indices.append(
                index
            )

            continue

        similarities = cosine_similarity(
            matrix[index].reshape(
                1,
                -1
            ),
            matrix[selected_indices]
        ).flatten()

        if np.max(similarities) < 0.65:

            selected_indices.append(
                index
            )

    selected_indices.sort()

    combined_sentences = [
        all_sentences[index]["text"]
        for index in selected_indices
    ]

    return " ".join(
        combined_sentences
    )

@app.get("/")
def read_root():

    return {
        "status": "Server running efficiently",
        "message": (
            "Backend is running. "
            "Visit /docs to test the API."
        )
    }

@app.get("/health")
def health():

    return {
        "status": "Backend running efficiently"
    }

@app.post("/summarize")
async def summarize(
    files: List[UploadFile] = File(...)
):

    if not files:

        raise HTTPException(
            status_code=400,
            detail="No PDF files uploaded."
        )

    documents = []

    document_sentences = []

    for file in files:

        filename = (
            file.filename
            or "unknown.pdf"
        )

        if not filename.lower().endswith(
            ".pdf"
        ):

            raise HTTPException(
                status_code=400,
                detail=f"{filename} is not a PDF."
            )

        try:

            file_bytes = await file.read()

            if not file_bytes:

                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} is empty."
                )

            text, pages = extract_text(
                file_bytes
            )

            text = clean_text(
                text
            )

            if not text:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No readable text found "
                        f"in {filename}."
                    )
                )

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

            summary_sentences = create_summary(
                sentences
            )

            summary = " ".join(
                summary_sentences
            )

            document_sentences.append(
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
                detail=(
                    f"Error processing "
                    f"{filename}: {str(error)}"
                )
            )

    combined_summary = create_combined_summary(
        document_sentences
    )

    return {
        "documents": documents,
        "combined_summary": combined_summary
    }