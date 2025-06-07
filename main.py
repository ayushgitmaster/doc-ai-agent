# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI

# from app.arxiv_handler import fetch_arxiv_pdf
# from app.ingestion.parser import extract_text_from_pdf
# from app.ingestion.splitter import split_text
# from app.retrieval.embedder import get_embedder
# from app.retrieval.retriever import build_vectorstore
# from app.agent.qa_chain import get_qa_chain

# load_dotenv()

# st.set_page_config(page_title="📚 AI Paper Assistant", layout="centered")
# st.title("📚 AI Research Paper Assistant")
# st.caption("Powered by Gemini, LangChain, FAISS & ArXiv")

# query = st.text_input("🔍 Search for a research paper (topic or keywords):")
# question = st.text_input("🤔 Ask a question about the paper (e.g., summary, methods, metrics):")

# if st.button("📥 Fetch & Analyze Paper"):
#     if not query or not question:
#         st.warning("Please enter both a paper topic and a question.")
#     else:
#         with st.spinner("🔎 Searching and downloading from arXiv..."):
#             pdf_path, title = fetch_arxiv_pdf(query)
#             st.success(f"✅ Downloaded paper: *{title}*")

#         with st.spinner("📄 Extracting & splitting content..."):
#             raw_text = extract_text_from_pdf(pdf_path)
#             chunks = split_text(raw_text)

#         with st.spinner("📊 Embedding & preparing retriever..."):
#             embedder = get_embedder()
#             vectorstore = build_vectorstore(chunks, embedder)

#         with st.spinner("💬 Asking Gemini..."):
#             llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
#             qa = get_qa_chain(llm, vectorstore)
#             answer = qa.invoke(question)

#         st.markdown("### 📢 Gemini's Answer")
#         st.success(answer)

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain

from app.arxiv_handler import fetch_arxiv_pdfs
from app.ingestion.parser import extract_text_from_pdf
from app.ingestion.splitter import split_text
from app.retrieval.embedder import get_embedder
from app.retrieval.retriever import build_vectorstore

load_dotenv()

# Streamlit config
st.set_page_config(page_title="Doc Q&A AI Agent Version-1", layout="centered")
st.title("📚 Doc Q&A AI Agent Version-1 ")
st.caption("🔍 Search and interact with multiple arXiv research papers using Gemini + LangChain")

# Session init
if 'docs' not in st.session_state:
    st.session_state.docs = []
    st.session_state.titles = []

# Search query
query = st.text_input("🔍 Enter a topic to search research papers:")

if st.button("📥 Fetch Papers"):
    if not query:
        st.warning("Please enter a valid topic.")
    else:
        with st.spinner("🔎 Fetching top 3 papers from arXiv..."):
            try:
                results = fetch_arxiv_pdfs(query, max_results=3)

                for pdf_path, title, paper_id in results:
                    raw_text = extract_text_from_pdf(pdf_path)
                    chunks = split_text(raw_text, metadata={"source": title})
                    st.session_state.docs.extend(chunks)
                    st.session_state.titles.append(title)

                st.success(f"✅ Loaded {len(results)} papers into memory.")
                st.info("You can now ask questions across all papers.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Show loaded paper titles
if len(st.session_state.titles) > 0:
    st.markdown("### 📄 Papers in Context:")
    for i, t in enumerate(st.session_state.titles):
        st.markdown(f"- {i+1}. *{t}*")

# QA Interface
st.markdown("---")
st.markdown("### 💬 Ask a Question Across All Papers")

question = st.text_input("Enter your question here:")

if st.button("🧠 Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("⚙️ Processing..."):
            try:
                embedder = get_embedder()
                vectorstore = build_vectorstore(
                    st.session_state.docs, embedder)

                retriever = vectorstore.as_retriever()
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
                )

                qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm, retriever=retriever, chain_type="stuff"
                )

                result = qa_chain({"question": question})
                answer = result["answer"]
                sources = result["sources"]

                st.markdown("### ✅ Gemini's Answer")
                st.success(answer)

                if sources:
                    st.markdown("### 📌 Sources")
                    for source in sources.split("\n"):
                        st.markdown(f"- {source}")
                else:
                    st.info("No sources identified for this answer.")
            except Exception as e:
                st.error(f"❌ Failed to get answer: {e}")

