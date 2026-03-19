
##  How to Run Locally

Follow the steps below to set up and run the project on your local machine.

First, clone the repository and navigate into the project folder:

```bash
git clone https://github.com/esthejas/Mini-RAG.git
cd mini-rag
```
create a virtual environment to isolate dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```
pip install -r requirements.txt

create a .env file in the root directory and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=your_api_key_here
```
start the Streamlit application:
```bash
streamlit run app.py
```

##  Embedding Model and LLM Used

The system uses the `all-MiniLM-L6-v2` model from the Sentence Transformers library to generate embeddings for document chunks. This model is chosen because it is lightweight, fast, and provides strong semantic understanding for short text, making it suitable for efficient similarity search in a local environment without requiring external APIs. For answer generation, the system uses the `meta-llama/llama-3-8b-instruct` model via the OpenRouter API. This model is selected due to its strong instruction-following capabilities, good balance between performance and cost, and ability to generate concise, context-aware responses based on the provided input.

---

##  Document Chunking and Retrieval

The document processing pipeline begins by loading markdown files from a local directory and converting them into structured document objects. These documents are then split into smaller chunks using the `RecursiveCharacterTextSplitter`, with a chunk size of 300 and an overlap of 50 to preserve contextual continuity. Each chunk is converted into vector embeddings and stored in a FAISS vector database for efficient semantic search. When a user query is received, the system retrieves the most relevant document chunks using Max Marginal Relevance (MMR), which ensures diversity in results and reduces redundancy, thereby improving the overall quality of retrieved context.

---

##  Grounding to Retrieved Context

Grounding is enforced through strict prompt design and controlled system behavior. The language model is explicitly instructed to generate answers only from the retrieved document chunks and to avoid using any external knowledge. If the answer is not present in the provided context, the model responds with "I don't know," preventing hallucinations and unsupported claims. The system also ensures transparency by displaying both the retrieved document chunks and the final generated answer, making the responses explainable, trustworthy, and directly verifiable against the source content.

---
