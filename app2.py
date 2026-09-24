import streamlit as st
import pandas as pd

from rag_pipeline import (
    build_rag,
    retrieve_documents,
    generate_answer
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🏥 Educational RAG-Based Healthcare Assistant"
)

st.write(
    """
This application uses Retrieval-Augmented Generation (RAG)
to retrieve relevant information from the uploaded healthcare
dataset and generate an answer using Gemini.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "📂 Upload Healthcare Dataset"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


# ============================================================
# CACHE RAG SYSTEM
# ============================================================

@st.cache_resource
def build_cached_rag(dataframe):

    return build_rag(dataframe)


# ============================================================
# CHECK FILE UPLOAD
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload a healthcare CSV file from the sidebar."
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read CSV file: {e}"
    )

    st.stop()


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "Disease",
    "Section",
    "Content"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"""
The uploaded CSV is missing these required columns:

{missing_columns}

Required columns are:
{required_columns}
"""
    )

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.sidebar.success(
    "Dataset uploaded successfully!"
)

st.sidebar.write(
    f"Rows: {len(df)}"
)

st.sidebar.write(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# BUILD RAG SYSTEM
# ============================================================

with st.spinner(
    "Building RAG system..."
):

    try:

        rag = build_cached_rag(
            df
        )

    except Exception as e:

        st.error(
            f"Error building RAG system: {e}"
        )

        st.stop()


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader(
    "💬 Ask a Healthcare Question"
)

question = st.text_input(
    "Enter your question",
    placeholder="Example: What are the symptoms of dengue?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🔍 Ask Question"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # --------------------------------------------------------
    # Retrieve documents
    # --------------------------------------------------------

    with st.spinner(
        "Retrieving relevant information..."
    ):

        retrieved_documents = retrieve_documents(
            question,
            rag,
            top_k=5
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.spinner(
        "Generating answer..."
    ):

        answer = generate_answer(
            question,
            retrieved_documents
        )

    # --------------------------------------------------------
    # Display answer
    # --------------------------------------------------------

    st.subheader(
        "🤖 Answer"
    )

    st.write(
        answer
    )

    # --------------------------------------------------------
    # Medical disclaimer
    # --------------------------------------------------------

    st.warning(
        """
        ⚠️ Educational Information Only

        This application is for educational purposes only
        and is not a substitute for professional medical
        advice, diagnosis, or treatment.
        """
    )

    # ========================================================
    # SOURCES
    # ========================================================

    st.subheader(
        "📚 Retrieved Sources"
    )

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        with st.expander(
            f"{i}. "
            f"{document['disease']} - "
            f"{document['section']}"
        ):

            st.write(
                "**Disease:**",
                document["disease"]
            )

            st.write(
                "**Section:**",
                document["section"]
            )

            st.write(
                "**Content:**"
            )

            st.write(
                document["content"]
            )

            st.write(
                f"**Semantic Similarity:** "
                f"{document['similarity']:.4f}"
            )

            


# ============================================================
# DATASET PREVIEW
# ============================================================

with st.expander(
    "📊 View Uploaded Dataset"
):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )