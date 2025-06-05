import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.arxiv_handler import fetch_arxiv_pdf
from app.ingestion.parser import extract_text_from_pdf
from app.ingestion.splitter import split_text
from app.retrieval.embedder import get_embedder
from app.retrieval.retriever import build_vectorstore
from app.agent.qa_chain import get_qa_chain

load_dotenv()

st.set_page_config(page_title="📚 AI Paper Assistant", layout="centered")
st.title("📚 AI Research Paper Assistant")
st.caption("Powered by Gemini, LangChain, FAISS & ArXiv")

query = st.text_input("🔍 Search for a research paper (topic or keywords):")
question = st.text_input("🤔 Ask a question about the paper (e.g., summary, methods, metrics):")

if st.button("📥 Fetch & Analyze Paper"):
    if not query or not question:
        st.warning("Please enter both a paper topic and a question.")
    else:
        with st.spinner("🔎 Searching and downloading from arXiv..."):
            pdf_path, title = fetch_arxiv_pdf(query)
            st.success(f"✅ Downloaded paper: *{title}*")

        with st.spinner("📄 Extracting & splitting content..."):
            raw_text = extract_text_from_pdf(pdf_path)
            chunks = split_text(raw_text)

        with st.spinner("📊 Embedding & preparing retriever..."):
            embedder = get_embedder()
            vectorstore = build_vectorstore(chunks, embedder)

        with st.spinner("💬 Asking Gemini..."):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
            qa = get_qa_chain(llm, vectorstore)
            answer = qa.run(question)

        st.markdown("### 📢 Gemini's Answer")
        st.success(answer)
