import os
import re

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please add it to your .env file."
    )

genai.configure(api_key=GEMINI_API_KEY)

llm = genai.GenerativeModel("gemini-2.5-flash")


# ============================================================
# 2. CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 5


# ============================================================
# 3. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean text by removing unnecessary spaces and formatting.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 4. CREATE CHUNKS
# ============================================================

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split long text into overlapping chunks.
    """

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# 5. PREPARE DOCUMENTS
# ============================================================

def prepare_documents(df):
    """
    Convert the uploaded dataframe into RAG documents.

    Expected columns:
        Disease
        Section
        Content
    """

    required_columns = ["Disease", "Section", "Content"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    documents = []

    for _, row in df.iterrows():

        disease = clean_text(row["Disease"])
        section = clean_text(row["Section"])
        content = clean_text(row["Content"])

        if not content:
            continue

        chunks = create_chunks(content)

        for chunk in chunks:

            documents.append(
                {
                    "disease": disease,
                    "section": section,
                    "content": chunk
                }
            )

    return documents


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    """
    Load Sentence Transformer embedding model.
    """

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    return model


# ============================================================
# 7. CREATE FAISS INDEX
# ============================================================

def create_faiss_index(documents, embedding_model):
    """
    Create FAISS vector index from documents.
    """

    texts = [
        document["content"]
        for document in documents
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# 8. BUILD COMPLETE RAG SYSTEM
# ============================================================

def build_rag(df):
    """
    Build the complete RAG system.

    Returns:
        {
            "documents": documents,
            "embedding_model": embedding_model,
            "index": index
        }
    """

    print("Preparing documents...")

    documents = prepare_documents(df)

    print(
        f"Total chunks created: {len(documents)}"
    )

    print("Loading embedding model...")

    embedding_model = load_embedding_model()

    print("Creating FAISS index...")

    index = create_faiss_index(
        documents,
        embedding_model
    )

    print("RAG system loaded successfully.")

    return {
        "documents": documents,
        "embedding_model": embedding_model,
        "index": index
    }


# ============================================================
# 9. LOAD RAG FROM CSV
# ============================================================

def load_rag(data_path="who_dataset.csv"):
    """
    Load CSV and build RAG system.

    Useful for testing the backend directly.
    """

    df = pd.read_csv(data_path)

    return build_rag(df)


# ============================================================
# 10. DETECT QUERY TOPIC
# ============================================================

def detect_query_topic(query):
    """
    Detect the main topic of the user query.
    """

    query = query.lower()

    topic_keywords = {

        "symptoms": [
            "symptom",
            "symptoms",
            "sign",
            "signs",
            "feel",
            "feeling"
        ],

        "causes": [
            "cause",
            "causes",
            "reason",
            "risk factor",
            "risk factors"
        ],

        "prevention": [
            "prevent",
            "prevention",
            "avoid",
            "protect"
        ],

        "diagnosis": [
            "diagnosis",
            "diagnose",
            "test",
            "tests",
            "testing",
            "diagnostic"
        ],

        "treatment": [
            "treatment",
            "treat",
            "therapy",
            "medicine",
            "medication",
            "management"
        ],

        "complications": [
            "complication",
            "complications",
            "effect",
            "effects"
        ]
    }

    for topic, keywords in topic_keywords.items():

        for keyword in keywords:

            if keyword in query:
                return topic

    return None


# ============================================================
# 11. DETECT DISEASE
# ============================================================

def detect_disease(query, documents):
    """
    Detect whether a known disease is mentioned
    in the user's query.
    """

    query_lower = query.lower()

    diseases = set()

    for document in documents:

        disease = document["disease"]

        if disease:
            diseases.add(disease)

    # Sort longest disease names first
    # to avoid partial matching problems.
    diseases = sorted(
        diseases,
        key=len,
        reverse=True
    )

    for disease in diseases:

        if disease.lower() in query_lower:
            return disease

    return None


# ============================================================
# 12. CALCULATE SECTION SCORE
# ============================================================

def calculate_section_score(
    section,
    query_topic
):
    """
    Give additional score to sections
    relevant to the user's query topic.
    """

    if not query_topic:
        return 0.0

    section = section.lower()

    section_mapping = {

        "symptoms": [
            "symptom",
            "sign"
        ],

        "causes": [
            "cause",
            "risk"
        ],

        "prevention": [
            "prevention",
            "prevent"
        ],

        "diagnosis": [
            "diagnos",
            "test"
        ],

        "treatment": [
            "treatment",
            "therapy",
            "management"
        ],

        "complications": [
            "complication"
        ]
    }

    keywords = section_mapping.get(
        query_topic,
        []
    )

    for keyword in keywords:

        if keyword in section:
            return 1.0

    return 0.0


# ============================================================
# 13. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    rag,
    top_k=TOP_K
):
    """
    Retrieve relevant documents using:

    1. Semantic similarity from FAISS
    2. Disease matching
    3. Section/topic matching
    4. Disease-first prioritization
    """

    embedding_model = rag["embedding_model"]
    index = rag["index"]
    documents = rag["documents"]

    # --------------------------------------------------
    # 1. Convert user query into an embedding
    # --------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # --------------------------------------------------
    # 2. Retrieve a larger candidate set from FAISS
    # --------------------------------------------------

    search_k = min(
        top_k * 10,
        len(documents)
    )

    similarities, indices = index.search(
        query_embedding,
        search_k
    )

    # --------------------------------------------------
    # 3. Detect disease and question topic
    # --------------------------------------------------

    detected_disease = detect_disease(
        query,
        documents
    )

    query_topic = detect_query_topic(
        query
    )

    # --------------------------------------------------
    # 4. Create candidate results
    # --------------------------------------------------

    results = []

    for similarity, index_position in zip(
        similarities[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        document = documents[index_position]

        # ----------------------------------------------
        # Disease matching
        # ----------------------------------------------

        disease_score = 0.0

        if (
            detected_disease
            and document["disease"].lower()
            == detected_disease.lower()
        ):
            disease_score = 1.0

        # ----------------------------------------------
        # Section/topic matching
        # ----------------------------------------------

        section_score = calculate_section_score(
            document["section"],
            query_topic
        )

        # ----------------------------------------------
        # Semantic similarity
        # ----------------------------------------------

        semantic_similarity = float(
            similarity
        )

        # ----------------------------------------------
        # Final ranking score
        # ----------------------------------------------

        ranking_score = (
            semantic_similarity
            + (0.20 * disease_score)
            + (0.10 * section_score)
        )

        results.append(
            {
                "disease": document["disease"],
                "section": document["section"],
                "content": document["content"],
                "similarity": semantic_similarity,
                "disease_score": disease_score,
                "section_score": section_score,
                "ranking_score": ranking_score
            }
        )

    # --------------------------------------------------
    # 5. Disease-first filtering
    # --------------------------------------------------

    if detected_disease:

        same_disease_results = [
            result
            for result in results
            if result["disease"].lower()
            == detected_disease.lower()
        ]

        # If enough documents exist for the detected disease,
        # use only those documents.
        if len(same_disease_results) >= top_k:
            results = same_disease_results

    # --------------------------------------------------
    # 6. Sort by final ranking score
    # --------------------------------------------------

    results = sorted(
        results,
        key=lambda x: x["ranking_score"],
        reverse=True
    )

    # --------------------------------------------------
    # 7. Return top K documents
    # --------------------------------------------------

    return results[:top_k]


# ============================================================
# 14. GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    retrieved_documents
):
    """
    Generate an answer using only retrieved context.
    """

    if not retrieved_documents:

        return (
            "I could not find relevant information "
            "in the provided knowledge base."
        )

    # --------------------------------------------------------
    # Create context
    # --------------------------------------------------------

    context_parts = []

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context_parts.append(
            f"""
Source {i}
Disease: {document["disease"]}
Section: {document["section"]}

Content:
{document["content"]}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an educational healthcare information assistant.

Answer the user's question using ONLY the information
provided in the retrieved context below.

Do not use outside knowledge.

If the answer is not available in the context,
clearly say:

"The provided information does not contain
the answer to this question."

Do not invent or assume medical information.

Keep the answer clear, concise, and easy to understand.

User Question:
{query}

Retrieved Context:
{context}

Answer:
"""

    try:

        response = llm.generate_content(
            prompt
        )

        answer = response.text.strip()

    except Exception as e:

        answer = (
            f"Error generating answer: {str(e)}"
        )

    return answer


# ============================================================
# 15. BACKEND TEST
# ============================================================

if __name__ == "__main__":

    print("\nLoading dataset...\n")

    rag = load_rag(
        "who_dataset.csv"
    )

    query = "How is diabetes diagnosed?"

    print(
        f"\nQuestion: {query}\n"
    )

    retrieved_documents = retrieve_documents(
        query,
        rag,
        top_k=5
    )

    print("\nRetrieved Documents:\n")

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        print(
            f"{i}. "
            f"{document['disease']} | "
            f"{document['section']} | "
            f"Similarity: "
            f"{document['similarity']:.4f} | "
            f"Ranking: "
            f"{document['ranking_score']:.4f}"
        )

    print("\nGenerating answer...\n")

    answer = generate_answer(
        query,
        retrieved_documents
    )

    print("ANSWER:")
    print(answer)