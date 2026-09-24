# 🏥 Educational RAG-Based Healthcare Assistant

An educational healthcare question-answering application built using **Retrieval-Augmented Generation (RAG)**. The system retrieves relevant information from a healthcare knowledge base using **semantic search with FAISS** and generates grounded responses using **Google Gemini**.

The application is developed with **Python and Streamlit** and is designed to answer healthcare-related questions using only the information available in the provided knowledge base.

> ⚠️ **Disclaimer:** This project is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 📌 Project Overview

Large Language Models can generate useful answers, but they may sometimes produce information that is not present in the application's knowledge source.

To address this problem, I built a **Retrieval-Augmented Generation (RAG) pipeline** that first retrieves relevant healthcare information from a knowledge base and then provides that retrieved context to Gemini for answer generation.

The system follows:

**User Question → Query Embedding → FAISS Retrieval → Re-ranking → Relevant Context → Gemini → Grounded Answer**

---

## 🎯 Problem Statement

Healthcare information is available in large amounts of textual data, but finding relevant information from this data manually can be time-consuming.

A simple LLM-based chatbot can also generate responses that are not grounded in the application's knowledge source.

The objective of this project is to build an educational question-answering system that:

* Retrieves relevant healthcare information from a structured knowledge base.
* Uses semantic similarity rather than only keyword matching.
* Improves retrieval using disease and topic-based relevance.
* Generates answers using retrieved context.
* Reduces unsupported or hallucinated responses by instructing the LLM to use only the retrieved information.

---

## 🎯 Objectives

* Build an end-to-end RAG pipeline.
* Convert healthcare text into semantic embeddings.
* Store and search embeddings using FAISS.
* Retrieve the most relevant healthcare documents for a user query.
* Improve retrieval using disease and topic-based re-ranking.
* Generate grounded answers using Gemini.
* Provide a simple interactive interface using Streamlit.
* Display the retrieved sources used for generating the answer.

---

## 🗂️ Dataset

The project uses a healthcare knowledge base containing information related to diseases and healthcare topics.

The dataset contains **8,063 rows** and includes information organized around fields such as:

* Disease
* Section
* Content

The application converts the textual content into smaller chunks before generating embeddings.

---

## 🏗️ System Architecture

```text
                    Healthcare Dataset
                           │
                           ▼
                    Data Validation
                           │
                           ▼
                      Text Cleaning
                           │
                           ▼
                       Chunking
                           │
                           ▼
                Sentence Transformer
                all-MiniLM-L6-v2
                           │
                           ▼
                    Text Embeddings
                           │
                           ▼
                    FAISS Vector DB
                           │
                           │
                    User Question
                           │
                           ▼
                    Query Embedding
                           │
                           ▼
                  Semantic Retrieval
                           │
                           ▼
              Disease + Topic Re-ranking
                           │
                           ▼
                 Top Relevant Documents
                           │
                           ▼
                  Retrieved Context
                           │
                           ▼
                       Gemini
                           │
                           ▼
                    Grounded Answer
                           │
                           ▼
                    Streamlit UI
```

---

## 🔄 RAG Pipeline

### 1. Data Loading

The application accepts a healthcare CSV file through the Streamlit interface.

The uploaded dataset is validated to ensure the required columns are available.

---

### 2. Text Cleaning

The healthcare content is cleaned before processing.

The cleaning process helps remove unnecessary whitespace and prepares the text for chunking and embedding.

---

### 3. Text Chunking

Large text content is divided into smaller chunks.

The project uses:

* **Chunk size:** 500 characters
* **Chunk overlap:** 100 characters

The overlap helps preserve context between neighboring chunks.

---

### 4. Embedding Generation

Each text chunk is converted into a numerical vector using:

**Sentence Transformer — `all-MiniLM-L6-v2`**

These embeddings represent the semantic meaning of the healthcare text.

The query is also converted into an embedding using the same model.

---

### 5. FAISS Vector Search

The generated embeddings are stored in a **FAISS `IndexFlatIP`** index.

The project uses normalized embeddings with inner-product similarity, which makes the similarity score approximately equivalent to **cosine similarity**.

FAISS retrieves the most semantically similar documents for the user's question.

---

### 6. Retrieval and Re-ranking

Instead of relying only on semantic similarity, the retrieved candidates are further evaluated using additional relevance signals:

* Semantic similarity
* Disease matching
* Section/topic matching
* Topic relevance within document content

The final ranking score is calculated using weighted relevance signals.

This helps prioritize documents that are not only semantically similar but also related to the user's requested disease and topic.

---

### 7. Context Construction

The highest-ranked documents are combined into a context that is passed to Gemini.

The prompt instructs Gemini to:

* Use only the retrieved context.
* Avoid using outside knowledge.
* Avoid inventing medical information.
* Clearly state when the answer is not available in the knowledge base.

---

### 8. Answer Generation

**Gemini `gemini-2.5-flash`** generates the final response using the retrieved healthcare context.

The generated answer is then displayed in the Streamlit application.

---

## 🧪 Example

### User Question

```text
What are the risk factors for type 2 diabetes?
```

### Retrieved Source

```text
Disease: Diabetes
Section: Type 2 diabetes
```

### Retrieved Information

```text
Factors that contribute to developing type 2 diabetes
include being overweight, not getting enough exercise,
and genetics.
```

### Generated Answer

```text
Factors that contribute to developing type 2 diabetes
include being overweight, not getting enough exercise,
and genetics.
```

This demonstrates how the answer is generated from retrieved knowledge rather than relying only on the LLM's general knowledge.

---

## 🧪 Testing Performed

The RAG pipeline was tested using questions covering different diseases and healthcare topics.

| Test Question                                  | Retrieval | Grounded Answer |
| ---------------------------------------------- | --------- | --------------- |
| What are the risk factors for type 2 diabetes? | ✅         | ✅               |
| How is diabetes diagnosed?                     | ✅         | ✅               |
| How can type 2 diabetes be prevented?          | ✅         | ✅               |
| What are the complications of diabetes?        | ✅         | ✅               |
| What is the treatment for migraine?            | ✅         | ✅               |

The tests verified that relevant documents were retrieved and that generated answers were supported by the retrieved context.

---

## 🖥️ Application Features

* CSV dataset upload
* Dataset validation
* Automatic text chunking
* Semantic search
* FAISS vector retrieval
* Disease-aware retrieval
* Topic/section-based re-ranking
* Gemini-powered answer generation
* Retrieved source display
* Similarity score display
* Empty-question validation
* Healthcare information disclaimer
* Streamlit interactive interface

---

## 🛠️ Tech Stack

### Programming

* Python

### Generative AI

* Google Gemini
* Retrieval-Augmented Generation (RAG)

### Embeddings

* Sentence Transformers
* `all-MiniLM-L6-v2`

### Vector Database

* FAISS

### Data Processing

* Pandas
* NumPy

### Frontend

* Streamlit

### Configuration

* Python-dotenv

---

## 📁 Project Structure

```text
rag_project/
│
├── app2.py
├── rag_pipeline.py
├── requirements.txt
├── who_dataset.csv
├── .env
├── .gitignore
│
└── evaluation/
    └── evaluation_questions.csv
```

### File Description

| File                       | Description                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| `app2.py`                  | Streamlit user interface and application flow                                            |
| `rag_pipeline.py`          | Data processing, chunking, embeddings, FAISS retrieval, re-ranking and Gemini generation |
| `requirements.txt`         | Python dependencies                                                                      |
| `who_dataset.csv`          | Healthcare knowledge base                                                                |
| `.env`                     | Stores the Gemini API key locally                                                        |
| `.gitignore`               | Prevents sensitive/unnecessary files from being committed                                |
| `evaluation_questions.csv` | Evaluation questions for testing the RAG system                                          |

> `.env` should **not** be uploaded to GitHub.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd rag_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the API key

Create a `.env` file:

```text
GEMINI_API_KEY=your_gemini_api_key
```

Do not commit this file to GitHub.

### 6. Run the application

```bash
streamlit run app2.py
```

The application will open in the browser.

---

## 🔐 Security

API credentials are stored using environment variables rather than being hardcoded in the source code.

The `.env` file is excluded through `.gitignore`.

Never upload API keys or other secrets to GitHub.

---

## 🚀 Future Improvements

* Add automated RAG evaluation metrics.
* Add retrieval precision/recall evaluation.
* Add answer faithfulness evaluation.
* Improve query classification.
* Add persistent FAISS index storage.
* Add conversation history.
* Add metadata filtering before vector search.
* Add document/source citations in generated responses.
* Migrate to the latest Google GenAI SDK.
* Add more healthcare documents to the knowledge base.

---

## 📌 Key Learning Outcomes

Through this project, I worked with:

* Retrieval-Augmented Generation
* Text chunking
* Semantic embeddings
* Vector similarity search
* FAISS
* Query retrieval
* Retrieval re-ranking
* Prompt engineering
* Grounded LLM responses
* Gemini API integration
* Streamlit application development
* Environment-variable based API security

---

## 👩‍💻 Author

**Vanaja Pulluri**

Aspiring AI Developer | Data Science | Generative AI | Machine Learning

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes only**. The information generated by the application should not be used for medical diagnosis, treatment decisions, or emergency healthcare decisions.
