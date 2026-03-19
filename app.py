import streamlit as st
from dotenv import load_dotenv
from rag_langchain import load_documents, split_documents, create_vectorstore, get_retriever
import os
import requests

st.set_page_config(page_title="Mini RAG Chatbot")
st.title("Mini RAG Chatbot")

@st.cache_resource
def setup():
    docs = load_documents("data")
    split_docs = split_documents(docs)
    vectorstore = create_vectorstore(split_docs)
    retriever = get_retriever(vectorstore)
    return retriever

retriever = setup()

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

import requests

def generate_answer(query, context):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Answer ONLY using the context.
If not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    data = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=data)

    print("DEBUG RESPONSE:", response.text)  # 👈 ADD THIS

    result = response.json()

    if "choices" not in result:
        return f"Error from API: {result}"

    return result["choices"][0]["message"]["content"]


query = st.text_input("Ask your question:")

if query:
    retrieved_docs = retriever.invoke(query)

    st.subheader(" Retrieved Context")
    context_text = ""
    for i, doc in enumerate(retrieved_docs):
        st.write(f"**Chunk {i+1}:**")
        st.write(doc.page_content)
        st.write("---")
        context_text += doc.page_content + "\n\n"

    #  Generate Answer
    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.
Do NOT repeat the context.
Give a clear and short answer.


Context:
{context_text}

Question:
{query}
"""

    answer = generate_answer(query, context_text)

    st.subheader(" Generated Answer")
    st.write(answer)